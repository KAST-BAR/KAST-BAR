import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True
import torch._dynamo
torch._dynamo.config.optimize_ddp = False
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer
from model.model_bth_dual_stream import  BTH_DualStream
from model.model_neural_transformer import NTConfig
from dataset import create_test_data_pooled
from dataset import load_from_disk
from utils import cosine_scheduler
import math
import wandb

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    backend = 'nccl'
    device = 'cuda' 
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def get_args():
    parser = argparse.ArgumentParser('MSVQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path to dataset directory')
    parser.add_argument('--text_dataset_dir', default='./', help='path to text dataset directory')
    parser.add_argument('--text_model_path', default='./', help='path to text model')
    parser.add_argument('--use_preprocessed', default=False, action='store_true', help='use preprocessed pooled data')
    parser.add_argument('--preprocessed_dir', default='./', type=str, help='path to preprocessed data directory')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BAR_msvq')
    parser.add_argument('--entity', default=None, help='name of team')
    parser.add_argument('--wandb_runname', default='MSVQ')
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--text_batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')
    parser.add_argument('--dual_shared', default=False, action='store_true')

    return parser.parse_args()

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/{}'.format(args.wandb_runname))
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    text_embedding_config = {
        'model_path': args.text_model_path
    }
    text_data_dir = Path(args.text_dataset_dir)
    text_model_path = Path(args.text_model_path)
    text_reports = {}
    _text_dataset = load_from_disk(str(text_data_dir))
    _tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), trust_remote_code=True)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    pad_token_id = _tokenizer.pad_token_id
    
    for sample in _text_dataset:
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


    num_workers = args.num_workers
    dataset_dir = args.dataset_dir
    data_loader_train = create_test_data_pooled(
        use_preprocessed=False,
        batch_size=args.batch_size,
        preprocessed_dir=None,
        dataset_dir=dataset_dir,
        ddp=ddp,
        ddp_rank=ddp_rank if ddp else 0,
        ddp_world_size=ddp_world_size if ddp else 1,
        group_by_hash=True,
        num_workers=num_workers
    )
    
    dataset_train = data_loader_train.dataset
    
    iter_num = 0

    encoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=128)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'scratch'

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        encoder_conf = NTConfig(**encoder_args)
        decoder_conf = NTConfig(**decoder_args)
        model = BTH_DualStream(
            encoder_conf, 
            decoder_conf,
            share_cross_scale=args.dual_shared
        )
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['encoder_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            encoder_args[k] = checkpoint_model_args[k]
        checkpoint_model_args = checkpoint['decoder_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            decoder_args[k] = checkpoint_model_args[k]
        encoder_conf = NTConfig(**encoder_args)
        decoder_conf = NTConfig(**decoder_args)
        model = BTH_DualStream(
            encoder_conf, 
            decoder_conf,
            share_cross_scale=args.dual_shared
        )
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        keys_to_remove = []
        for k in state_dict.keys():
            if 'patch_embed.l.0' in k:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            state_dict.pop(k)
            if master_process:
                print(f"Removed unexpected key from checkpoint: {k}")
        
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1

    model.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    if args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    if args.wandb_log and master_process:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if init_from == 'resume':
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=True)
        else:
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'))

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )


    t0 = time.time()
    raw_model = model.module if ddp else model

    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if ddp:
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            
            X, Y_freq, Y_raw, input_chans, input_time, input_mask, dataset_name, sample_name = batch
            scale_data = None  
            input_mask = input_mask.to(device, non_blocking=True)
            Y_freq = Y_freq.float().to(device, non_blocking=True)
            Y_raw = Y_raw.float().to(device, non_blocking=True)
            
            X = X.float().to(device, non_blocking=True)
            
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            

            with ctx:
                loss, _, log = model(X, Y_freq, Y_raw, scale_data, input_chans, input_time, input_mask)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train loss {log['train/total_loss']:.4f}, freq loss {log['train/rec_freq_loss']:.4f}, raw loss {log['train/rec_raw_loss']:.4f}, quant loss {log['train/quant_loss']:.4f}, pcc {log['train/pcc']}")                
                if args.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/total_loss": log['train/total_loss'],
                        "train/freq_loss": log['train/rec_freq_loss'],
                        "train/raw_loss": log['train/rec_raw_loss'],
                        "train/quant_loss": log['train/quant_loss'],
                        "train/pcc": log['train/pcc'],
                        "lr": lr,
                        "epoch": epoch
                    })
            

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
        if ddp:
            torch.distributed.barrier()
        if master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
        if ddp:
            torch.distributed.barrier()
    if ddp:
        destroy_process_group()





if __name__ == '__main__':
    args = get_args()
    main(args)
