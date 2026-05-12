import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import signal
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
import wandb
from qwen_vl_utils import process_vision_info
DEFAULT_EEG_DIR = Path("/data/home/whn/dataset/eeg/train_regroup/")
DEFAULT_MODEL_PATH = Path("/data/home/whn/eeg-lm/BAR_whn/quality-EEG-Text-Pairs/model/qwen/qwen2.5_3b")
DEFAULT_MODEL_TYPE = "qwen"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K_CHANNELS = 6


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    processor: Any | None
    model_type: str
    supports_multimodal: bool


def load_eeg_sample(pkl_path: Path) -> Dict[str, Any]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data


def channel_statistics(
    x: np.ndarray,
    channel_names: List[str],
    top_k_channels: int
) -> Tuple[List[str], Dict[str, float]]:
    summaries = []
    energy = float(np.mean(x ** 2))
    kurtosis = float(np.mean((x - np.mean(x)) ** 4) / (np.var(x) ** 2 + 1e-8))
    global_stats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "energy": energy,
        "kurtosis": kurtosis,
    }

    channel_order = np.argsort(np.std(x, axis=1))[::-1][:top_k_channels]
    for idx in channel_order:
        trace = x[idx]
        mean = float(np.mean(trace))
        std = float(np.std(trace))
        peak = float(np.max(np.abs(trace)))
        zero_crossings = int(np.sum(trace[:-1] * trace[1:] < 0))
        summaries.append(
            f"{channel_names[idx]}: mean {mean:.4f}, std {std:.4f}, peak amplitude {peak:.4f}, zero-crossing count {zero_crossings}"
        )
    return summaries, global_stats


def calculate_frequency_features(
    x: np.ndarray,
    channel_names: List[str],
    sampling_rate: float = 256.0,
    top_k_channels: int = 6
) -> Tuple[List[str], Dict[str, float]]:
    channel_freq_summaries = []
    
    nyquist_freq = sampling_rate / 2
    gamma_upper = min(100.0, nyquist_freq - 0.1)
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, gamma_upper)
    }
    
    global_band_powers = {band: [] for band in freq_bands.keys()}
    global_peak_frequencies = []
    global_peak_powers = []
    
    channel_order = np.argsort(np.std(x, axis=1))[::-1][:top_k_channels]
    
    for idx in channel_order:
        trace = x[idx]
        
        frequencies, psd = signal.welch(
            trace,
            fs=sampling_rate,
            nperseg=min(256, len(trace) // 4),
            noverlap=None,
            nfft=None,
            detrend='constant'
        )
        
        peak_freq_idx = np.argmax(psd)
        peak_frequency = float(frequencies[peak_freq_idx])
        peak_power = float(psd[peak_freq_idx])
        
        global_peak_frequencies.append(peak_frequency)
        global_peak_powers.append(peak_power)
        
        total_power = float(np.trapz(psd, frequencies))
        band_info = []
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            if np.any(band_mask):
                band_power = float(np.trapz(psd[band_mask], frequencies[band_mask]))
                relative_power = (band_power / total_power * 100) if total_power > 0 else 0.0
                band_info.append(f"{band_name} power: {band_power:.4f} ({relative_power:.2f}%)")
                global_band_powers[band_name].append(band_power)
        
        channel_summary = (
            f"{channel_names[idx]}: peak frequency {peak_frequency:.2f}Hz, peak power {peak_power:.4f}, "
            f"{', '.join(band_info)}"
        )
        channel_freq_summaries.append(channel_summary)
    
    global_freq_stats = {
        "mean_peak_frequency": float(np.mean(global_peak_frequencies)) if global_peak_frequencies else 0.0,
        "std_peak_frequency": float(np.std(global_peak_frequencies)) if global_peak_frequencies else 0.0,
        "mean_peak_power": float(np.mean(global_peak_powers)) if global_peak_powers else 0.0,
    }
    
    for band_name in freq_bands.keys():
        if global_band_powers[band_name]:
            global_freq_stats[f"mean_{band_name}_power"] = float(np.mean(global_band_powers[band_name]))
        else:
            global_freq_stats[f"mean_{band_name}_power"] = 0.0
    
    return channel_freq_summaries, global_freq_stats


def build_system_prompt(
    sample: Dict[str, Any],
    channel_summaries: List[str],
    global_stats: Dict[str, float],
    channel_freq_summaries: List[str] | None = None,
    global_freq_stats: Dict[str, float] | None = None
) -> str:
    sample_name = sample.get("sample_name", "Unknown_Sample")
    dataset_name = sample.get("dataset_name", "Unknown_Dataset")
    ch_names = sample.get("ch_names", [])
    num_channels = len(ch_names)
    num_points = sample.get("X", np.zeros((0, 0))).shape[1] if isinstance(sample.get("X"), np.ndarray) else 0

    channel_info = "\n".join(f"- {desc}" for desc in channel_summaries)
    global_info = (
        f"Global statistics across all channels: mean {global_stats['mean']:.4f}, std {global_stats['std']:.4f}, "
        f"energy {global_stats['energy']:.4f}, peak-to-peak {global_stats['peak_to_peak']:.4f}, kurtosis {global_stats['kurtosis']:.4f}"
    )

    freq_features_info = ""
    if channel_freq_summaries and global_freq_stats:
        channel_freq_info = "\n".join(f"- {desc}" for desc in channel_freq_summaries)
        global_freq_info = (
            f"Global time-frequency statistics: mean peak frequency {global_freq_stats.get('mean_peak_frequency', 0.0):.2f}Hz, "
            f"std of peak frequency {global_freq_stats.get('std_peak_frequency', 0.0):.2f}Hz, "
            f"mean peak power {global_freq_stats.get('mean_peak_power', 0.0):.4f}, "
            f"mean delta power {global_freq_stats.get('mean_delta_power', 0.0):.4f}, "
            f"mean theta power {global_freq_stats.get('mean_theta_power', 0.0):.4f}, "
            f"mean alpha power {global_freq_stats.get('mean_alpha_power', 0.0):.4f}, "
            f"mean beta power {global_freq_stats.get('mean_beta_power', 0.0):.4f}, "
            f"mean gamma power {global_freq_stats.get('mean_gamma_power', 0.0):.4f}"
        )
        freq_features_info = (
            "Time-frequency features (computed):\n"
            f"{global_freq_info}\n"
            "Representative channel time-frequency features:\n"
            f"{channel_freq_info}\n"
            "Note: peak frequency refers to the frequency with maximum power in the spectrum; band power includes absolute and relative (percentage) power.\n"
            "Band definitions: delta 0.5-4Hz, theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz, gamma 30-100Hz.\n\n"
        )
    else:
        freq_features_info = (
            "Time-frequency features: not computed (please reason based on existing statistical features).\n\n"
        )


    prompt = (
        "You are an EEG signal analysis expert.\n"
        "Based on the provided EEG signal statistics, generate a purely objective and technical data report."
        "Core principle: the report must have two parts. The first part is generic textbook-style background introduction (independent of the current sample)."
        "The second part is an objective description of the physical properties of the current sample."
        "It is strictly forbidden to perform clinical diagnosis, infer diseases, or predict the specific class label of the current sample.\n\n"
        "Data summary:\n"
        f"- Sample name: {sample_name}\n"
        f"- Dataset name: {dataset_name}\n"
        f"- Number of channels: {num_channels}\n"
        f"- Time series length: {num_points}\n"
        f"{global_info}\n"
        "Channel feature statistics:\n"
        f"{channel_info if channel_info else '- (no available channel information)'}\n\n"
        f"{freq_features_info}"
        "\n\n"
        "Analysis requirements:\n"
        "1. Dataset task description:\n"
        "   - Describe the general experimental paradigm of this dataset (e.g., a sleep monitoring experiment or a video-evoked emotion experiment).\n"
        "   - The description must be generic and applicable to all samples in the dataset. Do not describe the specific state of this sample.\n"
        "2. Task-related prior knowledge:\n"
        "   - List relevant neuroscience or physiological background knowledge (e.g., frontal lobe is related to executive function; delta band is dominant in deep sleep).\n"
        "   - These must be abstract theoretical knowledge and must not imply the state of the current sample.\n"
        "3. Physical properties of the signal:\n"
        "   - Objectively describe time-domain (amplitude, variance), frequency-domain (PSD distribution, main bands), and spatial (active channels) characteristics.\n"
        "   - Only state data facts, do not infer intent or diagnosis.\n"
        "\n"
        "Output format (JSON):\n"
        "{\n"
        '  \"dataset_name\": \"current dataset name\",\n'
        '  \"sample_name\": \"current sample name\",\n'
        '  \"dataset_task_description\": \"generic experimental paradigm description (background only, no sample-level conclusion).\",\n'
        '  \"task_prior_knowledge\": \"generic medical/neuroscience background (theoretical only, no sample-specific analysis).\",\n'
        '  \"signal_physical_features\": \"purely objective description, such as global mean, band power distribution, and peak frequency. Do not use words like abnormal, pathological, seizure, etc.\",\n'
        '  \"spatial_distribution_features\": \"objective description of which brain regions or channels show prominent signal or frequency characteristics.\",\n'
        '  \"data_quality_notes\": \"indicate whether there are outlier channels or possible noise based on statistics.\",\n'
        '  \"feature_summary\": \"one-sentence summary of the physical shape of the signal without any diagnostic conclusion.\"\n'
        "}\n\n"
        "Strictly follow JSON format. If unsure how to describe without revealing labels, only list numerical statistics."
    )

    return prompt


def run_inference_qwen(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float
) -> str:
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils is missing; cannot process Qwen multimodal inputs.")

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text


def run_inference_causal(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float
) -> str:
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output_text = tokenizer.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    return output_text


def replace_chinese_to_english(text: str) -> str:
    chinese_to_english = {
        '，': ',',
        '。': '.',
        '：': ':',
        '；': ';',
        '？': '?',
        '！': '!',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
        '、': ',',
    }
    result = text
    for chinese_char, english_char in chinese_to_english.items():
        result = result.replace(chinese_char, english_char)
    return result


def parse_json_response(output_text: str) -> Dict[str, Any]:
    start = output_text.find("{")
    end = output_text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("Valid JSON structure not found in model output.")
    json_str = output_text[start:end]
    json_str = replace_chinese_to_english(json_str)
    return json.loads(json_str)


def calculate_json_token_count(tokenizer, json_dict: Dict[str, Any]) -> int:
    json_str = json.dumps(json_dict, ensure_ascii=False)
    tokens = tokenizer.encode(json_str, add_special_tokens=False)
    return len(tokens)


def calculate_json_char_length(json_dict: Dict[str, Any]) -> int:
    json_str = json.dumps(json_dict, ensure_ascii=False)
    return len(json_str)


def load_model_bundle(
    model_type: str,
    model_path: Path,
    device_map: str,
    attn_implementation: str
) -> ModelBundle:
    model_type = model_type.lower()
    if model_type == "qwen_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=attn_implementation,
            device_map=device_map
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        tokenizer = processor.tokenizer
        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            model_type=model_type,
            supports_multimodal=True,
        )

    if model_type in {"llama", "llama3", "deepseek", "qwen"}:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=attn_implementation,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            processor=None,
            model_type=model_type,
            supports_multimodal=False,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EEG PKL files with configurable LLM backends.")
    parser.add_argument("--eeg_dir", type=Path, default=DEFAULT_EEG_DIR, help="Directory containing EEG PKL files.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model or weights.")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE,
                        choices=["qwen_vl", "qwen", "llama", "llama3", "deepseek"],
                        help="Model type to use.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for reports (default: eeg_dir).")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top_k_channels", type=int, default=DEFAULT_TOP_K_CHANNELS, help="Number of representative channels.")
    parser.add_argument("--device_map", type=str, default="auto", help="Model device map (default: auto).")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation.")
    parser.add_argument("--wandb_log", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name.")
    parser.add_argument("--wandb_runname", type=str, default=None, help="wandb run name.")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb API key.")
    parser.add_argument(
        "--ignore_group_dirs",
        type=str,
        default=None,
        help="Comma-separated directory names to ignore when grouping (e.g. 'eval,train,val'). Useful when fine-tuning datasets have extra directory levels."
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    os.environ["WANDB_API_KEY"] = args.wandb_api_key or ""
    eeg_dir = Path(args.eeg_dir)
    pkl_files = list(eeg_dir.rglob("*.pkl"))

    print("Loading model components...")
    bundle = load_model_bundle(
        model_type=args.model_type,
        model_path=args.model_path,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation
    )

    wandb_run = None
    if args.wandb_log:
        wandb_config = {
            "eeg_dir": str(eeg_dir),
            "model_path": str(args.model_path),
            "model_type": args.model_type,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k_channels": args.top_k_channels,
        }
        wandb_run = wandb.init(
            project=args.wandb_project or "eeg_directory_analysis",
            name=args.wandb_runname,
            config=wandb_config
        )

    if not pkl_files:
        print("No .pkl files found in the provided directory.")
        return

    output_dir = args.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.model_path.name}"/ f"{args.model_path.name}_{args.eeg_dir.name}_analysis_reports.jsonl"
    summary_file = summary_path.open("w", encoding="utf-8")

    ignore_dirs = getattr(args, 'ignore_group_dirs', None)
    if ignore_dirs:
        ignore_dirs = [d.strip() for d in ignore_dirs.split(',') if d.strip()]
    else:
        ignore_dirs = None
    
    def _extract_group_key(relative_dir: Path, ignore_dirs: List[str] | None = None) -> str:
        if ignore_dirs is None:
            return str(relative_dir)
        parts = [p for p in relative_dir.parts if p not in ignore_dirs]
        return str(Path(*parts)) if parts else str(relative_dir)

    grouped_files: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for pkl_path in pkl_files:
        relative_dir = pkl_path.relative_to(eeg_dir).parent
        group_key = _extract_group_key(relative_dir, ignore_dirs)
        sample_stem = pkl_path.stem
        sample_base = re.sub(r"_(\d+)$", "", sample_stem)
        grouped_files[(group_key, sample_base)].append(pkl_path)

    grouped_items = sorted(grouped_files.items(), key=lambda item: (item[0][0], item[0][1]))
    total_groups = len(grouped_items)

    print(f"Found {total_groups} groups aggregated by directory and sample. Starting processing...")

    for group_idx, ((group_path, sample_base), slice_paths) in enumerate(grouped_items, start=1):
        print("\n" + "=" * 80)
        print(f"Processing group: {group_path} -> {sample_base}")
        slice_paths_sorted = sorted(slice_paths)

        slice_samples: List[Dict[str, Any]] = []
        slice_arrays: List[np.ndarray] = []
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


        try:
            combined_x = np.concatenate(slice_arrays, axis=1)
        except ValueError as exc:
            print(f"⚠️ Failed to concatenate slices: {exc}")
            continue

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
            args.top_k_channels
        )
        
        sampling_rate = aggregated_sample.get("sfreq", aggregated_sample.get("sampling_rate", 256.0))
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            sampling_rate = 256.0
        
        channel_freq_summaries, global_freq_stats = calculate_frequency_features(
            x,
            ch_names,
            sampling_rate=float(sampling_rate),
            top_k_channels=args.top_k_channels
        )
        
        prompt = build_system_prompt(
            aggregated_sample,
            channel_summaries,
            global_stats,
            channel_freq_summaries,
            global_freq_stats
        )
        start_time = time.perf_counter()

        if bundle.supports_multimodal:
            output_text = run_inference_qwen(
                model=bundle.model,
                processor=bundle.processor,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
        else:
            output_text = run_inference_causal(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )

        elapsed = time.perf_counter() - start_time
        print(f"Inference time: {elapsed:.2f}s")

        print("Model Output:")
        print(output_text)

        try:
            report = parse_json_response(output_text)
            dataset_name_from_path = group_path.split('/')[0] if group_path else None
            dataset_name = (
                dataset_name_from_path 
                if dataset_name_from_path and dataset_name_from_path not in ['train', 'eval', 'val', 'test']
                else aggregated_sample.get("dataset_name", "Unknown_Dataset")
            )
            if dataset_name in ['train', 'eval', 'val', 'test']:
                dataset_name = aggregated_sample.get("dataset_name", dataset_name_from_path or "Unknown_Dataset")
            
            result_item = {
                "dataset_name": dataset_name,
                "sample_name": aggregated_sample.get("sample_name", sample_base),
                "report": report,
                "group_path": group_path,
                "slice_files": aggregated_sample.get("slice_files", []),
            }
            summary_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            summary_file.flush()
            print("✅ Parsed Report:")
            print(json.dumps(report, ensure_ascii=False, indent=2))

            tokenizer = bundle.tokenizer
            json_token_count = calculate_json_token_count(tokenizer, report)
            json_char_length = calculate_json_char_length(report)
            progress_percent = (group_idx / total_groups) * 100
            
            print(f"JSON token count: {json_token_count}, character length: {json_char_length}")
            print(f"Progress: {group_idx}/{total_groups} ({progress_percent:.1f}%)")


            if wandb_run:
                wandb_run.log(
                    {
                        "progress": group_idx,
                        "progress_percent": progress_percent,
                        "json_token_count": json_token_count,
                        "json_char_length": json_char_length,
                        "inference_time_sec": elapsed,
                        "last_group": group_path,
                        "slice_count": aggregated_sample["slice_count"],
                    }
                )
        except Exception as exc:
            print(f"JSON parsing failed: {exc}")
            if wandb_run:
                progress_percent = (group_idx / total_groups) * 100
                wandb_run.log(
                    {
                        "progress": group_idx,
                        "progress_percent": progress_percent,
                        "json_parse_error": True,
                        "last_group": group_path,
                    }
                )

    summary_file.close()
    print("\n" + "=" * 80)
    print(f"Analysis reports written to: {summary_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

