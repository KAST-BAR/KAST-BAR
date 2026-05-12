
import os
import time
import math
import argparse
from contextlib import nullcontext
from accelerate.utils import set_seed
import numpy as np
import torch
from einops import rearrange, repeat

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from model.model_ITBAR import EEGBARLLM

from model.model_bth_dual_stream import  BTH_DualStream
from model.model_STARouter import STARouter
from model.model_neural_transformer import NTConfig

from pathlib import Path
from utils import cosine_scheduler
from model.standard_1020_chorder import remove_unused_ch
from collections import OrderedDict
from dataset4STARLM import create_data
from accelerate.state import AcceleratorState
from accelerate import Accelerator, DeepSpeedPlugin


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None
text_reports = {}
text_tokenizer = None
accelerator = None


class UnifiedEEGBAR(torch.nn.Module):
    def __init__(self, starouter, llm_model):
        super().__init__()
        self.starouter = starouter
        self.llm_model = llm_model

    def forward(self, rest_BND, X_text, text_mask, eeg_codes, text_label_mask=None):
        STAR_out = self.starouter(rest_BND, X_text, text_mask)
        outputs = self.llm_model(X_text, STAR_out, eeg_codes, text_label_mask=text_label_mask)
        return outputs

    def get_orthogonality_loss(self):
        return self.starouter.get_orthogonality_loss()

def init(args, accelerator_obj: Accelerator):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, text_reports, text_tokenizer, accelerator
    accelerator = accelerator_obj

    ddp_world_size = accelerator.num_processes
    ddp_rank = accelerator.process_index
    ddp_local_rank = accelerator.local_process_index
    ddp = ddp_world_size > 1
    master_process = accelerator.is_main_process
    device = accelerator.device
    device_type = device.type

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = 'bfloat16'
        mixed_precision = 'bf16'
    else:
        dtype = 'float16'
        mixed_precision = 'fp16'

    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if device_type == 'cpu':
        ctx = nullcontext()
    elif dtype == 'bfloat16':
        ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        ctx = torch.autocast(device_type=device_type, dtype=torch.float16)

    text_data_dir = Path(args.text_dataset_dir)
    text_model_path = Path(args.text_model_path)
    if master_process:
        print("Loading text dataset and tokenizer to match EEG...")
    text_dataset = load_from_disk(str(text_data_dir))
    if text_tokenizer is None:
        text_tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), trust_remote_code=True)
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id
    pad_token_id = text_tokenizer.pad_token_id
    
    for sample in text_dataset:
        key = (sample["dataset_name"], sample["sample_name"])
        if key not in text_reports:
            ids = sample["text_input_ids"]
            original_length = len(ids)
            if len(ids) > 1024:
                ids = ids[:1024]
                text_mask = [1] * len(ids)
            else:
                text_mask = [1] * len(ids) + [0] * (1024 - len(ids))
                ids = ids + [pad_token_id] * (1024 - len(ids))
            text_reports[key] = {
                "length": original_length,
                "ids": ids,
                "text_mask": text_mask,
            }
    print(f"Text lookup table built. Total entries: {len(text_reports)}.")


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--text_dataset_dir', default='./', help='path to text dataset directory')
    parser.add_argument('--text_model_path', default='./', help='path to text model')
    parser.add_argument('--EEG_tokenizer_path', default='./', help='path where EEG tokenizer is')
    parser.add_argument('--eeg_vocab_size', default=8192, type=int, help='EEG vocabulary size')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BAR')
    parser.add_argument('--entity', default=None, help='name of team')
    parser.add_argument('--wandb_runname', default='pretrain')
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=15, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay (default: 1e-1)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--orth_loss_weight', type=float, default=0.1, 
                        help='Weight for the expert orthogonality regularization loss')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()



def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, accelerator

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision = "bf16"
    else:
        mixed_precision = "fp16"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    
    state = AcceleratorState()
    if state.deepspeed_plugin is not None and state.deepspeed_plugin.deepspeed_config is not None:
        state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.eeg_batch_size

    init(args, accelerator)
    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/{}'.format(args.wandb_runname))
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    text_embedding_config = {
        'model_path': args.text_model_path
    }
    def get_batch(dataset_name, sample_name):
        text_input_ids = torch.zeros((len(dataset_name), 1024), dtype=torch.long)
        text_mask = torch.zeros((len(dataset_name), 1024), dtype=torch.long)
        for idx, (ds, sp) in enumerate(zip(dataset_name, sample_name)):
            report = text_reports.get((ds, sp))
            text_mask[idx] = torch.tensor(report["text_mask"], dtype=torch.long)
            text_input_ids[idx] = torch.tensor(report["ids"], dtype=torch.long)
        text_input_ids = text_input_ids.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        return text_input_ids, text_mask

    dataset_dir = args.dataset_dir
    data_loader_train = create_data(
        batch_size=args.eeg_batch_size,
        dataset_dir=dataset_dir,
        ddp=ddp,
        ddp_rank=ddp_rank if ddp else 0,
        ddp_world_size=ddp_world_size if ddp else 1,
        group_by_hash=True,
        num_workers=4
    )
    dataset_train = data_loader_train.dataset

    encoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=128)

    eeg_tokenizer_ckpt_path = os.path.join(args.out_dir, args.EEG_tokenizer_path)
    eeg_tokenizer_checkpoint = torch.load(eeg_tokenizer_ckpt_path, map_location=device, weights_only=False)
    eeg_tokenizer_checkpoint_model_args = eeg_tokenizer_checkpoint['encoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        encoder_args[k] = eeg_tokenizer_checkpoint_model_args[k]
    eeg_tokenizer_checkpoint_model_args = eeg_tokenizer_checkpoint['decoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        decoder_args[k] = eeg_tokenizer_checkpoint_model_args[k]
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    eeg_tokenizer = BTH_DualStream(encoder_conf, decoder_conf)
    eeg_tokenizer_state_dict = eeg_tokenizer_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(eeg_tokenizer_state_dict.items()):
        if k.startswith(unwanted_prefix):
            eeg_tokenizer_state_dict[k[len(unwanted_prefix):]] = eeg_tokenizer_state_dict.pop(k)
    all_keys = list(eeg_tokenizer_state_dict.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('VQ.'):
            new_dict[key[3:]] = eeg_tokenizer_state_dict[key]
    eeg_tokenizer.load_state_dict(new_dict, strict=False)
    eeg_tokenizer.eval()
    for param in eeg_tokenizer.parameters():
        param.requires_grad = False
    eeg_tokenizer.to(device)
    eeg_tokenizer_checkpoint = None

    text_model_config = AutoConfig.from_pretrained(args.text_model_path, trust_remote_code=True)
    text_hidden_size = getattr(text_model_config, 'hidden_size', None) \
        or getattr(text_model_config, 'n_embd', None) \
        or getattr(text_model_config, 'd_model', None)
    if text_hidden_size is None:
        raise ValueError("Cannot infer hidden size from text model config (hidden_size/n_embd/d_model).")

    tokenizer = text_tokenizer
    
    
    new_tokens = [f"<eeg_{i}>" for i in range(args.eeg_vocab_size)]
    
    if "<eeg_0>" not in tokenizer.get_vocab():
        print(f"[Info] Expanding vocabulary: original size {len(tokenizer)}")
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"[Info] Added {num_added} EEG tokens")
    else:
        print(f"[Info] EEG tokens already exist; skipping. Current size {len(tokenizer)}")

    eeg_token_start_id = tokenizer.convert_tokens_to_ids("<eeg_0>")
    print(f"EEG Token Start ID: {eeg_token_start_id}")
    
    init_from = 'pretrained'
    latest_ckpt_dir = None
    if os.path.exists(checkpoint_out_dir):
        chk_dirs = [d for d in os.listdir(checkpoint_out_dir) if d.startswith('checkpoint-')]
        if len(chk_dirs) > 0:
            chk_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            latest_ckpt_dir = os.path.join(checkpoint_out_dir, chk_dirs[-1])
            init_from = 'resume'
            if master_process:
                print(f"Detected ZeRO-3 checkpoint folder, will resume from: {latest_ckpt_dir}")
    
    iter_num = 0
    start_epoch = 0
    
    print(f"Loading BAR model from: {args.text_model_path}")
    model_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.text_model_path,
        trust_remote_code=True,
        dtype=model_dtype,
        attn_implementation="flash_attention_2",
    )
    llm_model.resize_token_embeddings(len(tokenizer))
    llm_model.gradient_checkpointing_enable()
    llm_model.config.use_cache = False
    llm_model.enable_input_require_grads()
    llm_model = llm_model.to(device)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["lm_head"]
    )
    llm_model = get_peft_model(llm_model, peft_config)
    
    if master_process:
        llm_model.print_trainable_parameters()
    
    model = EEGBARLLM(llm_model, tokenizer, eeg_token_start_id, args.eeg_vocab_size).to(device)

    starouter = STARouter(
        d_model=text_hidden_size,
        num_queries=16,
        num_eeg_channels=91,
        nhead=8,
        dropout=0.1,
        text_embedding_config=text_embedding_config,
        text_embedding_module=llm_model.get_input_embeddings(),
        freeze_text=True
    )
    model_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    starouter = starouter.to(device).to(model_dtype)
    if master_process:
        print(f"Detected text hidden size: {text_hidden_size}")
        print(f"Model parameters of starouter: {sum(p.numel() for p in starouter.parameters()):,}")
        print(f"Trainable parameters of starouter: {sum(p.numel() for p in starouter.parameters() if p.requires_grad):,}")
    
    if init_from == 'resume' and latest_ckpt_dir is not None:
        meta_path = os.path.join(latest_ckpt_dir, 'training_metadata.pt')
        if os.path.exists(meta_path):
            metadata = torch.load(meta_path, map_location='cpu')
            iter_num = metadata.get('iter_num', 0)
            start_epoch = metadata.get('epoch', 0) + 1
        if master_process:
            print(f"Will resume iterator/epoch from metadata in: {latest_ckpt_dir}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f'Total BARLLM parameters: {total_params:,}')
        print(f'Trainable BARLLM parameters: {trainable_params:,}')

    lora_params = [p for p in model.llm.parameters() if p.requires_grad]
    eeg_embed_params = [model.trainable_eeg_embed.weight]
    router_params = [p for p in starouter.parameters() if p.requires_grad]
    all_trainable_params = lora_params + eeg_embed_params + router_params
    optimizer = torch.optim.AdamW(
        all_trainable_params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    unified_model = UnifiedEEGBAR(starouter, model)

    unified_model, optimizer = accelerator.prepare(
        unified_model, optimizer
    )

    if init_from == 'resume' and latest_ckpt_dir is not None:
        if master_process:
            print(f"Loading ZeRO-3 sharded state from: {latest_ckpt_dir}")
        accelerator.load_state(latest_ckpt_dir)

    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key  
        wandb_settings = wandb.Settings(init_timeout=300)
        if init_from == 'resume':
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=True, settings=wandb_settings)
        else:
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), settings=wandb_settings)

    num_training_steps_per_epoch = len(dataset_train) // args.eeg_batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )

    t0 = time.time()
    local_iter_num = 0
    skip_count = 0

    for epoch in range(start_epoch, args.epochs):
        if ddp and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        for step, (batch) in enumerate(data_loader_train):
            grad_norm = None

            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            X_eeg,_, _, input_chans, input_time, input_mask , dataset_names, sample_names= batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            X_text, text_mask = get_batch(dataset_names, sample_names)
            
            with torch.no_grad():
                with ctx:
                    inds_BN, rest_BND = eeg_tokenizer.get_codebook_msinds_and_msfeats(X_eeg, input_chans, input_time, input_mask)
                    inds_BN = inds_BN.detach()
                    rest_BND = rest_BND.detach()

                    original_vocab_size = len(tokenizer) - args.eeg_vocab_size
            
            with accelerator.accumulate(unified_model):
                model_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32
                rest_BND = rest_BND.to(dtype=model_dtype)

                with ctx:
                    eeg_codes = torch.clamp(inds_BN.clone(), min=0, max=args.eeg_vocab_size - 1)
                    outputs = unified_model(rest_BND, X_text, text_mask, eeg_codes)
                    main_loss = outputs.loss / args.gradient_accumulation_steps

                    text_loss_value = float(outputs.text_loss.detach())
                    eeg_loss_value = float(outputs.eeg_loss.detach())
                    eeg_acc_value = outputs.eeg_acc
                    orth_loss = unified_model.get_orthogonality_loss()
                    
                    weighted_orth_loss = (args.orth_loss_weight * orth_loss) / args.gradient_accumulation_steps
                    loss = main_loss + weighted_orth_loss

                    log1 = {
                        'train/loss': eeg_loss_value,
                        'train/accuracy': eeg_acc_value,
                    }
                    log2 = {
                        'train/loss': text_loss_value,
                        'train/accuracy': 0.0,
                    }

                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 150:
                    skip_count += 1
                    loss_value = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else str(loss.item())
                    if master_process:
                        print(f"Warning: Skipping batch {skip_count} due to abnormal loss: {loss_value}")
                        print(f"  - EEG loss: {log1['train/loss']:.4f}, Text loss: {log2['train/loss']:.4f}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if args.grad_clip != 0.0:
                        grad_norm = accelerator.clip_grad_norm_(
                            unified_model.parameters(),
                            args.grad_clip
                        )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                skip_info = f"skipped: {skip_count}" if skip_count > 0 else ""
                
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: "
                      f"Total: {loss.item()*args.gradient_accumulation_steps:.4f}, "
                      f"Main: {outputs.loss.item():.4f}, "
                      f"Text: {text_loss_value:.4f}, "
                      f"EEG: {eeg_loss_value:.4f} (Acc: {eeg_acc_value:.2%}), "
                      f"Orth: {orth_loss.item():.4f}, "
                      f"{skip_info}")
                
                if args.wandb_log:
                    log_dict = {
                        "iter": iter_num,
                        "train/total_loss": loss.item() * args.gradient_accumulation_steps,
                        "train/main_loss": outputs.loss.item(),
                        "train/orth_loss": orth_loss.item(),
                        "train/eeg_loss": eeg_loss_value,
                        "train/text_loss": text_loss_value,
                        "train/eeg_accuracy": eeg_acc_value,
                        "lr": lr,
                        "epoch": epoch,
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    wandb.log(log_dict)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iter_num += 1
            local_iter_num += 1
        
        if args.evaluate:
            loss, accuracy = evaluate(
                unified_model=unified_model,
                eeg_tokenizer=eeg_tokenizer,
                dataloader=data_loader_train,
                get_batch=get_batch,
                eeg_vocab_size=args.eeg_vocab_size,
            )
            if master_process:
                print('='* 10)
                print(f"Evaluate : loss {loss:.4f}, accuracy {accuracy:.4f}, perplexity {math.exp(loss):.4f}")
                print('='* 10)
                if args.wandb_log:
                    wandb.log({
                                "val/eeg_loss": loss,
                                "val/eeg_accuracy": accuracy,
                                'val/perplexity': math.exp(loss),
                            })
        
        accelerator.wait_for_everyone()

        save_dir = os.path.join(checkpoint_out_dir, f'checkpoint-{epoch}')

        if (epoch + 1) % args.save_ckpt_freq == 0:
            if master_process:
                print(f"Saving ZeRO-3 sharded checkpoint to {save_dir}...")

            accelerator.save_state(save_dir)

            if master_process:
                metadata = {
                    'model_args': {'text_model_path': args.text_model_path, 'eeg_vocab_size': args.eeg_vocab_size},
                    'iter_num': iter_num,
                    'epoch': epoch,
                }
                torch.save(metadata, os.path.join(save_dir, 'training_metadata.pt'))

            if master_process:
                print(f"Exporting LoRA weights and tokenizer to {save_dir}...")

                unwrapped_model = accelerator.unwrap_model(unified_model)

                save_path_lora = os.path.join(checkpoint_out_dir, f'lora_weights-{epoch}')

                try:
                    unwrapped_model.llm_model.llm.save_pretrained(save_path_lora)
                    tokenizer.save_pretrained(save_path_lora)
                    print("LoRA export success.")
                except RuntimeError as e:
                    print(f"Warning: LoRA export failed due to memory/config issues: {e}")
                    print("Don't worry, the training state is safely saved in checkpoint folders.")


@torch.no_grad()
def evaluate(unified_model, eeg_tokenizer, dataloader, get_batch, eeg_vocab_size=8192):
    global ctx, device, dtype

    model_was_training = unified_model.training
    unified_model.eval()

    losses = []
    acc = []

    for _, batch in enumerate(dataloader):
        X_eeg, _, _, input_chans, input_time, input_mask, dataset_names, sample_names = batch

        X_eeg = X_eeg.float().to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)

        X_text, text_mask = get_batch(dataset_names, sample_names)

        with ctx:
            inds_BN, rest_BND = eeg_tokenizer.get_codebook_msinds_and_msfeats(
                X_eeg, input_chans, input_time, input_mask
            )
            inds_BN = inds_BN.detach()
            rest_BND = rest_BND.detach()

        model_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16 if dtype == 'bfloat16' else torch.float32
        rest_BND = rest_BND.to(dtype=model_dtype)

        with ctx:
            eeg_codes = torch.clamp(inds_BN.clone(), min=0, max=eeg_vocab_size - 1)
            outputs = unified_model(rest_BND, X_text, text_mask, eeg_codes)

        losses.append(outputs.loss.item())
        acc.append(0.0)

    if model_was_training:
        unified_model.train()

    return np.mean(losses), np.mean(acc)





if __name__ == '__main__':
    args = get_args()
    main(args)
