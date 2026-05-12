from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

try:
    import wandb
except Exception:
    wandb = None

from analyze_eeg_directory_multi import (
    build_system_prompt,
    calculate_frequency_features,
    calculate_json_char_length,
    calculate_json_token_count,
    channel_statistics,
    load_eeg_sample,
    load_model_bundle,
    parse_args,
    parse_json_response,
    run_inference_causal,
    run_inference_qwen,
    replace_chinese_to_english,
)


def _should_use_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1 and dist.is_available()


def _init_distributed(backend: str) -> Tuple[int, int, int, bool]:
    if not _should_use_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return 0, 1, local_rank, False

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count()))))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def _load_processed_groups(
    output_file: Path,
) -> set[Tuple[str, str]]:
    processed = set()
    if not output_file.exists():
        return processed
    
    try:
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get("processed", False) and item.get("report") is not None:
                        group_path = item.get("group_path", "")
                        sample_name = item.get("sample_name", "")
                        if group_path and sample_name:
                            processed.add((group_path, sample_name))
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception as exc:
        print(f"⚠️ Error while reading processed groups: {exc}")
    
    return processed


def _distribute_groups(
    grouped_items: Sequence[Tuple[Tuple[str, str], List[Path]]],
    rank: int,
    world_size: int,
    processed_groups: set[Tuple[str, str]] | None = None,
) -> List[Tuple[int, Tuple[Tuple[str, str], List[Path]]]]:
    if processed_groups is None:
        processed_groups = set()
    
    original_indexed = list(enumerate(grouped_items, start=1))
    
    filtered_with_original_idx = [
        (original_idx, group_item) 
        for original_idx, group_item in original_indexed
        if group_item[0] not in processed_groups
    ]
    
    if world_size <= 1:
        return filtered_with_original_idx
    
    reassigned = [
        (original_idx, group_item)
        for new_idx, (original_idx, group_item) in enumerate(filtered_with_original_idx, start=1)
        if (new_idx - 1) % world_size == rank
    ]
    
    return reassigned


def _extract_group_key(
    relative_dir: Path,
    ignore_dirs: List[str] | None = None,
) -> str:
    if ignore_dirs is None:
        return str(relative_dir)
    parts = [p for p in relative_dir.parts if p not in ignore_dirs]
    return str(Path(*parts)) if parts else str(relative_dir)


def _process_group(
    slice_paths: List[Path],
    sample_base: str,
    group_path: str,
    top_k_channels: int,
    bundle,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any] | None, str | None, float]:
    slice_paths_sorted = sorted(slice_paths)
    slice_samples: List[Dict[str, Any]] = []
    slice_arrays: List[Any] = []
    channel_names: List[str] | None = None

    for slice_path in slice_paths_sorted:
        sample = load_eeg_sample(slice_path)
        x = sample.get("X")
        if not isinstance(x, np.ndarray):
            print(f"⚠️ {slice_path.name} has missing or invalid 'X' field, skipping this slice.")
            continue

        channel_names = sample.get("ch_names", [])
        slice_samples.append(sample)
        slice_arrays.append(x)

    if not slice_arrays:
        return None, "no valid slice data", 0.0

    try:
        combined_x = np.concatenate(slice_arrays, axis=1)
    except ValueError as exc:
        return None, f"failed to concatenate slices: {exc}", 0.0

    base_sample = slice_samples[0]
    aggregated_sample = dict(base_sample)
    aggregated_sample["X"] = combined_x
    aggregated_sample["ch_names"] = channel_names or base_sample.get("ch_names", [])
    aggregated_sample["sample_name"] = sample_base
    aggregated_sample["slice_count"] = len(slice_arrays)
    aggregated_sample["slice_files"] = [str(path) for path in slice_paths_sorted]
    aggregated_sample["group_path"] = group_path

    dataset_name_from_path = group_path.split('/')[0] if group_path else None
    dataset_name = (
        dataset_name_from_path 
        if dataset_name_from_path and dataset_name_from_path not in ['train', 'eval', 'val', 'test']
        else aggregated_sample.get("dataset_name", "Unknown_Dataset")
    )
    if dataset_name in ['train', 'eval', 'val', 'test']:
        dataset_name = aggregated_sample.get("dataset_name", dataset_name_from_path or "Unknown_Dataset")
    aggregated_sample["dataset_name"] = dataset_name

    x = aggregated_sample["X"]
    ch_names = aggregated_sample.get("ch_names", [])
    channel_summaries, global_stats = channel_statistics(
        x,
        ch_names,
        top_k_channels,
    )
    
    sampling_rate = aggregated_sample.get("sfreq", aggregated_sample.get("sampling_rate", 256.0))
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        sampling_rate = 256.0
    
    channel_freq_summaries, global_freq_stats = calculate_frequency_features(
        x,
        ch_names,
        sampling_rate=float(sampling_rate),
        top_k_channels=top_k_channels
    )
    
    prompt = build_system_prompt(
        aggregated_sample,
        channel_summaries,
        global_stats,
        channel_freq_summaries,
        global_freq_stats
    )

    start_time = time.perf_counter()

    max_retries = 3
    report = None
    processed = False
    output_text = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if bundle.supports_multimodal:
                output_text = run_inference_qwen(
                    model=bundle.model,
                    processor=bundle.processor,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            else:
                output_text = run_inference_causal(
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            report = parse_json_response(output_text)
            processed = True
            break
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt < max_retries:
                error_msg = f"JSON parse failed (attempt {attempt}/{max_retries}): {exc}\nModel output first 500 chars: {output_text[:500] if output_text else 'N/A'}\nModel output last 500 chars: {output_text[-500:] if output_text else 'N/A'}"
                print(f"⚠️ {error_msg}")
                print("🔄 Retrying generation...")
            else:
                error_msg = f"JSON parse failed after {max_retries} attempts: {exc}\nModel output first 500 chars: {output_text[:500] if output_text else 'N/A'}\nModel output last 500 chars: {output_text[-500:] if output_text else 'N/A'}"
                print(f"⚠️ {error_msg}")
                processed = False
                report = None

    elapsed = time.perf_counter() - start_time

    dataset_name_from_path = group_path.split('/')[0] if group_path else None
    dataset_name = (
        dataset_name_from_path 
        if dataset_name_from_path and dataset_name_from_path not in ['train', 'eval', 'val', 'test']
        else aggregated_sample.get("dataset_name", "Unknown_Dataset")
    )
    if dataset_name in ['train', 'eval', 'val', 'test']:
        dataset_name = aggregated_sample.get("dataset_name", dataset_name_from_path or "Unknown_Dataset")

    result_item = {
        "processed": processed,
        "dataset_name": dataset_name,
        "sample_name": aggregated_sample.get("sample_name", sample_base),
        "report": report,
        "group_path": group_path,
        "slice_files": aggregated_sample.get("slice_files", []),
    }

    tokenizer = bundle.tokenizer
    if report is not None:
        json_token_count = calculate_json_token_count(tokenizer, report)
        json_char_length = calculate_json_char_length(report)
    else:
        json_token_count = 0
        json_char_length = 0

    metrics_text = (
        f"JSON token count: {json_token_count}, char length: {json_char_length}, "
        f"slice_count: {aggregated_sample['slice_count']}"
    )

    return result_item, metrics_text, elapsed


def main() -> None:
    args = parse_args()
    backend = os.environ.get("EEG_DDP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size, local_rank, distributed = _init_distributed(backend)

    if args.wandb_log and rank != 0:
        os.environ["WANDB_MODE"] = "offline"

    os.environ["WANDB_API_KEY"] = args.wandb_api_key or ""
    eeg_dir = Path(args.eeg_dir)
    pkl_files = list(eeg_dir.rglob("*.pkl"))

    print(f"[Rank {rank}] Loading model components...")

    if world_size == 1 and torch.cuda.is_available():
        device_map = "auto"
        print(f"[Rank {rank}] single-process multi-GPU mode, device_map='auto'")
    else:
        device_map = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        print(f"[Rank {rank}] multi-process mode, device_map='{device_map}'")
    
    bundle = load_model_bundle(
        model_type=args.model_type,
        model_path=args.model_path,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )

    wandb_run = None
    if args.wandb_log and rank == 0 and wandb is not None:
        wandb_config = {
            "eeg_dir": str(eeg_dir),
            "model_path": str(args.model_path),
            "model_type": args.model_type,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k_channels": args.top_k_channels,
            "world_size": world_size,
        }
        wandb_run = wandb.init(
            project=args.wandb_project or "eeg_directory_analysis",
            name=args.wandb_runname or f"ddp_rank0_{time.strftime('%Y%m%d_%H%M%S')}",
            config=wandb_config,
        )

    if not pkl_files:
        if rank == 0:
            print("⚠️ No .pkl files found in the directory.")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    ignore_dirs = getattr(args, 'ignore_group_dirs', None)
    if ignore_dirs:
        ignore_dirs = [d.strip() for d in ignore_dirs.split(',') if d.strip()]
    else:
        ignore_dirs = None

    grouped_files: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for pkl_path in pkl_files:
        relative_dir = pkl_path.relative_to(eeg_dir).parent
        group_key = _extract_group_key(relative_dir, ignore_dirs)
        sample_stem = pkl_path.stem
        sample_base = re.sub(r"_(\d+)$", "", sample_stem)
        grouped_files[(group_key, sample_base)].append(pkl_path)

    grouped_items = sorted(grouped_files.items(), key=lambda item: (item[0][0], item[0][1]))
    total_groups = len(grouped_items)

    output_dir = args.output_dir or eeg_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"{args.model_path.name}"/f"{args.model_path.name}_{args.eeg_dir.name}_analysis_reports_ddp.jsonl"
    
    if world_size == 1:
        output_file = final_output_path
    else:
        output_file = output_dir / f"{args.model_path.name}"/f"{args.model_path.name}_{args.eeg_dir.name}_analysis_reports_ddp_rank{rank}.jsonl"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed_groups: set[Tuple[str, str]] = set()
    if final_output_path.exists():
        extra_processed = _load_processed_groups(final_output_path)
        processed_groups.update(extra_processed)
        if extra_processed and rank == 0:
            print(f"[Rank 0] loaded {len(extra_processed)} processed groups from existing results; these will be skipped.")

    if world_size > 1:
        if final_output_path.exists():
            summary_processed = _load_processed_groups(final_output_path)
            processed_groups.update(summary_processed)
            if summary_processed and rank == 0:
                print(f"[Rank 0] loaded {len(summary_processed)} processed groups from final file")
        
        for r in range(world_size):
            rank_file = output_dir / f"{args.model_path.name}"/f"{args.model_path.name}_{args.eeg_dir.name}_analysis_reports_ddp_rank{r}.jsonl"
            if rank_file.exists():
                rank_processed = _load_processed_groups(rank_file)
                processed_groups.update(rank_processed)
                if rank_processed and rank == 0:
                    print(f"[Rank 0] loaded {len(rank_processed)} processed groups from rank {r} file")
    else:
        processed_groups = _load_processed_groups(output_file)
    
    grouped_keys = {item[0] for item in grouped_items}
    if processed_groups:
        original_processed_len = len(processed_groups)
        processed_groups = {g for g in processed_groups if g in grouped_keys}
        filtered_out = original_processed_len - len(processed_groups)
        if filtered_out and rank == 0:
            print(f"[Rank 0] filtered out {filtered_out} processed groups not in current task.")
        print(f"[Rank {rank}] detected {len(processed_groups)} processed groups; these will be skipped.")
    
    assigned_groups = _distribute_groups(grouped_items, rank, world_size, processed_groups)
    
    remaining_groups = total_groups - len(processed_groups) if processed_groups else total_groups

    if rank == 0:
        print(f"Total {total_groups} groups, distributed over {world_size} processes.")
        if processed_groups:
            print(f"{len(processed_groups)} groups already processed, {remaining_groups} remaining.")
            print(f"⚠️ Resume mode: remaining tasks will be redistributed across {world_size} processes.")
    
    if remaining_groups > 0:
        print(f"[Rank {rank}] assigned {len(assigned_groups)} groups (remaining {remaining_groups} unprocessed groups will be redistributed).")
    else:
        print(f"[Rank {rank}] all groups already processed; nothing to do.")

    success_count = 0
    failed_count = 0
    
    file_mode = "a" if output_file.exists() else "w"
    with output_file.open(file_mode, encoding="utf-8") as temp_file:
        for local_idx, (group_idx, ((group_path, sample_base), slice_paths)) in enumerate(assigned_groups, start=1):
            header = f"[Rank {rank}] ({local_idx}/{len(assigned_groups)}) Global group {group_idx}/{total_groups}"
            print("\n" + "=" * 80)
            print(f"{header}\nProcessing group: {group_path} -> {sample_base}")

            result_item, metrics_text, elapsed = _process_group(
                slice_paths=slice_paths,
                sample_base=sample_base,
                group_path=group_path,
                top_k_channels=args.top_k_channels,
                bundle=bundle,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )


            temp_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            
            print(f"Inference time: {elapsed:.2f}s")
            if metrics_text:
                print(metrics_text)

            if wandb_run and rank == 0:
                progress_percent = (group_idx / total_groups) * 100
                slice_count = len(result_item.get("slice_files", []))
                wandb_run.log(
                    {
                        "progress": group_idx,
                        "progress_percent": progress_percent,
                        "inference_time_sec": elapsed,
                        "last_group": group_path,
                        "slice_count": slice_count,
                    }
                )
            if result_item.get("report") is None:
                print(f"{header} failed: {metrics_text}")
                failed_count += 1
                continue
            success_count += 1

    print(f"\n[Rank {rank}] finished: {success_count} succeeded, {failed_count} failed")
    print(f"[Rank {rank}] result file saved to: {output_file}")

    if distributed and world_size > 1:
        dist.barrier()
        
        if rank == 0:
            summary_path = final_output_path
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 80)
            print(f"[Rank 0] start merging result files from all ranks...")
            
            existing_records: set[Tuple[str, str]] = set()
            existing_lines: list[str] = []
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as ef:
                    for line in ef:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            group_path = item.get("group_path", "")
                            sample_name = item.get("sample_name", "")
                            if group_path and sample_name:
                                existing_records.add((group_path, sample_name))
                                existing_lines.append(line)
                        except (json.JSONDecodeError, KeyError):
                            continue
                if existing_lines:
                    print(f"[Rank 0] final file exists with {len(existing_lines)} records; keeping them.")
            
            total_written = len(existing_lines)
            with summary_path.open("w", encoding="utf-8") as final_file:
                for line in existing_lines:
                    final_file.write(line + "\n")
                
                for r in range(world_size):
                    rank_temp_file = output_dir / f"{args.model_path.name}"/f"{args.model_path.name}_{args.eeg_dir.name}_analysis_reports_ddp_rank{r}.jsonl"
                    if rank_temp_file.exists():
                        new_count = 0
                        with rank_temp_file.open("r", encoding="utf-8") as rf:
                            for line in rf:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    item = json.loads(line)
                                    group_path = item.get("group_path", "")
                                    sample_name = item.get("sample_name", "")
                                    if group_path and sample_name:
                                        key = (group_path, sample_name)
                                        if key not in existing_records:
                                            final_file.write(line + "\n")
                                            existing_records.add(key)
                                            new_count += 1
                                            total_written += 1
                                except (json.JSONDecodeError, KeyError):
                                    continue
                        rank_temp_file.unlink()
                        print(f"[Rank 0] merged results from rank {r} (added {new_count} new records) and removed temp file")
            
            print("\n" + "=" * 80)
            print(f"✅ Analysis reports written to: {summary_path}")
            print(f"✅ Total {total_written} records written")
            dist.barrier()
        else:
            dist.barrier()
            if output_file.exists():
                output_file.unlink()
                print(f"[Rank {rank}] removed temp file")
    elif rank == 0:
        print("\n" + "=" * 80)
        print(f"✅ Analysis reports written to: {output_file}")
        print(f"✅ Total {success_count} records written")

    if wandb_run:
        wandb_run.finish()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

