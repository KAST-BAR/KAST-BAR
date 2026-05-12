import os
import time
import math
import random
import argparse
from contextlib import nullcontext
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from einops import rearrange
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from pathlib import Path
from collections import OrderedDict, defaultdict
from dataset import load_from_disk
from model.model_bth_dual_stream import BTH_DualStream
from model.model_neural_transformer import NTConfig
from model.model_STARouter import STARouter
from model.model_ITBAR import EEGBARLLM, EEGBARLLM_instruction
from dataset4STARLM_instruction import create_data as create_eeg_loader
from utils import cosine_scheduler


master_process = None
device = None
dtype = None
ctx = None
ddp_rank = None
device_type = None
ddp = None
ddp_world_size = None
ddp_local_rank = None
accelerator = None
text_reports = {}
text_tokenizer = None

class UnifiedEEGBAR(torch.nn.Module):
    def __init__(self, starouter, llm_model):
        super().__init__()
        self.starouter = starouter
        self.llm_model = llm_model

    def forward(
        self,
        rest_BND,
        prefix_text_input_ids,
        prefix_text_mask,
        eeg_codes,
        suffix_input_ids=None,
        suffix_label_mask=None,
        sample_weights=None,
        use_eeg_code_segment=True,
    ):
        text_for_router = prefix_text_input_ids
        router_mask = prefix_text_mask
        STAR_out = self.starouter(rest_BND, text_for_router, router_mask)
        outputs = self.llm_model(
            text_input_ids=prefix_text_input_ids,
            eeg_feats=STAR_out,
            eeg_codes=eeg_codes,
            suffix_input_ids=suffix_input_ids,
            suffix_label_mask=suffix_label_mask,
            sample_weights=sample_weights,
            use_eeg_code_segment=use_eeg_code_segment,
        )
        return outputs

    def get_orthogonality_loss(self):
        return self.starouter.get_orthogonality_loss()


def init(args, accelerator_obj: Accelerator):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, accelerator, text_tokenizer, text_reports

    accelerator = accelerator_obj
    ddp_world_size = accelerator.num_processes
    ddp_rank = accelerator.process_index
    ddp_local_rank = accelerator.local_process_index
    ddp = ddp_world_size > 1
    master_process = accelerator.is_main_process
    device = accelerator.device
    device_type = device.type

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype_str = "bfloat16"
        mixed_precision = "bf16"
    else:
        dtype_str = "float16"
        mixed_precision = "fp16"
    globals()["dtype"] = dtype_str

    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]
    if device_type == "cpu":
        ctx = nullcontext()
    elif dtype_str == "bfloat16":
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
    print(f"Text lookup table built with {len(text_reports)} entries.")

def get_args():
    parser = argparse.ArgumentParser("EEGBAR instruction fine-tuning script", add_help=False)
    parser.add_argument("--out_dir", default="./output", help="output directory")
    parser.add_argument("--dataset_dir", default="./", help="root directory of downstream EEG datasets")
    parser.add_argument("--EEG_tokenizer_path", default="./checkpoints/VQ.pt", help="EEG tokenizer checkpoint path relative to out_dir")
    parser.add_argument("--text_dataset_dir", default="./text_dataset", help="text dataset directory")
    parser.add_argument("--text_model_path", default="./qwen", help="base Qwen text model path")
    parser.add_argument(
        "--pretrained_lora_path",
        default=None,
        help="LoRA weights directory exported from pretraining (e.g., lora_weights-XX from train_pretrain_dual_STAR_BAR.py)",
    )
    parser.add_argument(
        "--pretrained_ckpt_dir",
        default=None,
        help="DeepSpeed checkpoint directory from pretraining (for loading STARouter weights, e.g. checkpoint-0/)",
    )
    parser.add_argument(
        "--resume_path", 
        default=None, 
        help="DeepSpeed checkpoint path for resuming training or testing (e.g. ./output/checkpoints/run_name/checkpoint-5)"
    )
    parser.add_argument(
        "--test_only", 
        action="store_true", 
        help="only run evaluation (load weights and skip training)"
    )
    parser.add_argument(
        "--load_model_only",
        action="store_true",
        help="load model weights only without optimizer state (useful when GPU count changes)",
    )
    parser.add_argument("--eeg_vocab_size", default=8192, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eeg_batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--block_size", default=1024, type=int)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="whether to use cosine LR scheduler",
    )
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument(
        "--train_starouter",
        default=False,
        action="store_true",
        help="whether to train STARouter (default: frozen, used only as feature extractor)",
    )
    parser.add_argument(
        "--starouter_lr",
        type=float,
        default=None,
        help="learning rate for STARouter (if None, use same LR as main model)",
    )

    parser.add_argument("--wandb_log", default=False, action="store_true")
    parser.add_argument("--wandb_project", default="BAR")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb_runname", default="instruction-STAR-BAR")
    parser.add_argument("--wandb_api_key", type=str)

    return parser.parse_args()


def load_eeg_tokenizer(args):
    encoder_args = dict(
        n_layer=4,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=False,
        dropout=0.1,
        num_classes=0,
        in_chans=1,
        out_chans=16,
    )
    decoder_args = dict(
        n_layer=4,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=False,
        dropout=0.1,
        num_classes=0,
        in_chans=128,
    )
    eeg_tokenizer_ckpt_path = os.path.join(args.out_dir, args.EEG_tokenizer_path)
    eeg_tokenizer_checkpoint = torch.load(
        eeg_tokenizer_ckpt_path, map_location=device, weights_only=False
    )
    enc_args_ckpt = eeg_tokenizer_checkpoint["encoder_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias"]:
        encoder_args[k] = enc_args_ckpt[k]
    dec_args_ckpt = eeg_tokenizer_checkpoint["decoder_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias"]:
        decoder_args[k] = dec_args_ckpt[k]
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    eeg_tokenizer = BTH_DualStream(encoder_conf, decoder_conf)
    state_dict = eeg_tokenizer_checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    cleaned_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            k = k[len(unwanted_prefix) :]
        if k.startswith("VQ."):
            k = k[3:]
        cleaned_state[k] = v
    load_res = eeg_tokenizer.load_state_dict(cleaned_state, strict=False)
    if 'master_process' in globals() and master_process:
        miss = getattr(load_res, "missing_keys", [])
        unexp = getattr(load_res, "unexpected_keys", [])
        if miss:
            print(f"[Warn] EEG tokenizer missing keys: {miss}")
        if unexp:
            print(f"[Warn] EEG tokenizer unexpected keys: {unexp}")
    eeg_tokenizer.eval()
    for p in eeg_tokenizer.parameters():
        p.requires_grad = False
    eeg_tokenizer.to(device)
    return eeg_tokenizer


def build_instruction_text(tokenizer, dataset_name, label_id):
    if dataset_name == "SEED":
        prompt = (
            "Question: Which emotion type does this EEG segment belong to? "
            "Options: Positive, Neutral, Negative. Answer:"
        )
        answers = [" Positive", " Neutral", " Negative"]
    elif dataset_name == "TUAB":
        prompt = "Question: Is this EEG segment abnormal? Answer:"
        answers = [" No", " Yes"]
    elif dataset_name == "TUEV":
        prompt = (
            "Question: Which event type does this EEG segment belong to? Options: "
            "(A) spike and slow wave. (B) generalized periodic epileptiform discharge. "
            "(C) periodic lateralized epileptiform discharge. (D) eye movement. "
            "(E) artifact. (F) background. Answer:"
        )
        answers = [" (A)", " (B)", " (C)", " (D)", " (E)", " (F)"]
    elif dataset_name == "TUSL":
        prompt = (
            "Question: Which type does this EEG segment belong to? Options: "
            "(G) background. (H) seizure. (I) slowing. Answer:"
        )
        answers = [" (G)", " (H)", " (I)"]
    elif dataset_name == "HMC":
        prompt = (
            "Question: Which sleep type does this EEG segment belong to? Options: "
            "(J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer:"
        )
        answers = [" (J)", " (K)", " (L)", " (M)", " (N)"]
    elif dataset_name == "Workload":
        prompt = "Question: Is this EEG segment of high workload? Answer:"
        answers = [" No", " Yes"]
    elif dataset_name == "EDF":
        prompt = (
            "Question: What is the sleep stage of this EEG segment? Options: "
            "(O) wake. (P) N1. (Q) N2. (R) N3. (S) Movement. Answer:"
        ) 
        answers = [" (O)", " (P)", " (Q)", " (R)", " (S)"] 
    elif dataset_name == "BCICIV1":
        prompt = "Question: Is this EEG segments for right hand motor imagery? Answer:"
        answers = [" No", " Yes"] 
    elif dataset_name == "BCICIV2":
        prompt = (
            "Question: Which motor imagery type does this EEG segment belong to? Options: "
            "(T) left hand. (U) right hand. (V) foot. (W) tongue. Answer:"
        )
        answers = [" (T)", " (U)", " (V)", " (W)"]
    elif dataset_name == "STEW":
        prompt = "Question: Is this EEG segment of high mental workload? Answer:"
        answers = [" No", " Yes"]
    else:
        prompt = "Question: Is this EEG segment abnormal? Answer:"
        answers = [" No", " Yes"]

    label_id = int(label_id)
    label_id = max(0, min(label_id, len(answers) - 1))
    target_answer = answers[label_id]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    answer_text = target_answer + tokenizer.eos_token
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    
    input_ids = prompt_ids + answer_ids
    
    mask_prompt = [0] * len(prompt_ids)
    
    mask_answer = [1] * len(answer_ids)
    
    text_label_mask = mask_prompt + mask_answer
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    text_label_mask = torch.tensor(text_label_mask, dtype=torch.long)

    return input_ids, text_label_mask


def build_instruction_prompt_only(tokenizer, dataset_name):
    if "SEED" in dataset_name:
        dataset_name_normalized = "SEED"
    elif "TUAB" in dataset_name:
        dataset_name_normalized = "TUAB"
    elif "TUEV" in dataset_name:
        dataset_name_normalized = "TUEV"
    elif "TUSL" in dataset_name:
        dataset_name_normalized = "TUSL"
    elif "HMC" in dataset_name:
        dataset_name_normalized = "HMC"
    elif "Workload" in dataset_name or "workload" in dataset_name:
        dataset_name_normalized = "Workload"
    elif "EDF" in dataset_name:
        dataset_name_normalized = "EDF"
    elif "BCICIV1" in dataset_name or "BCI_IV1" in dataset_name:
        dataset_name_normalized = "BCICIV1"
    elif "BCICIV2" in dataset_name or "BCI_IV2" in dataset_name:
        dataset_name_normalized = "BCICIV2"
    elif "STEW" in dataset_name:
        dataset_name_normalized = "STEW"
    else:
        dataset_name_normalized = dataset_name
    
    if dataset_name_normalized == "SEED":
        prompt = (
            "Question: Which emotion type does this EEG segment belong to? "
            "Options: Positive, Neutral, Negative. Answer:"
        )
    elif dataset_name_normalized == "TUAB":
        prompt = "Question: Is this EEG segment abnormal? Answer:"
    elif dataset_name_normalized == "TUEV":
        prompt = (
            "Question: Which event type does this EEG segment belong to? Options: "
            "(A) spike and slow wave. (B) generalized periodic epileptiform discharge. "
            "(C) periodic lateralized epileptiform discharge. (D) eye movement. "
            "(E) artifact. (F) background. Answer:"
        )
    elif dataset_name_normalized == "TUSL":
        prompt = (
            "Question: Which type does this EEG segment belong to? Options: "
            "(G) background. (H) seizure. (I) slowing. Answer:"
        )
    elif dataset_name_normalized == "HMC":
        prompt = (
            "Question: Which sleep type does this EEG segment belong to? Options: "
            "(J) Wake. (K) NREM-1. (L) NREM-2. (M) NREM-3. (N) REM. Answer:"
        )
    elif dataset_name_normalized == "Workload":
        prompt = "Question: Is this EEG segment of high workload? Answer:"
    elif dataset_name_normalized == "EDF":
        prompt = (
            "Question: What is the sleep stage of this EEG segment? Options: "
            "(O) wake. (P) N1. (Q) N2. (R) N3. (S) Movement. Answer:"
        ) 
    elif dataset_name_normalized == "BCICIV1":
        prompt = "Question: Is this EEG segments for right hand motor imagery? Answer:"
    elif dataset_name_normalized == "BCICIV2":
        prompt = (
            "Question: Which motor imagery type does this EEG segment belong to? Options: "
            "(T) left hand. (U) right hand. (V) foot. (W) tongue. Answer:"
        )
    elif dataset_name_normalized == "STEW":
        prompt = "Question: Is this EEG segment of high mental workload? Answer:"
    else:
        prompt = "Question: Is this EEG segment abnormal? Answer:"
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    
    return prompt, prompt_ids

def get_task_candidate_ids(tokenizer, dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "SEED":
        candidates = [" Positive", " Neutral", " Negative"]
    elif dataset_name in ["TUAB", "Workload"]:
        candidates = [" No", " Yes"]
    elif dataset_name == "TUEV":
        candidates = [" (A)", " (B)", " (C)", " (D)", " (E)", " (F)"]
    elif dataset_name == "TUSL":
        candidates = [" (G)", " (H)", " (I)"]
    elif dataset_name == "HMC":
        candidates = [" (J)", " (K)", " (L)", " (M)", " (N)"]
    elif dataset_name == "EDF":
        candidates = [" (O)", " (P)", " (Q)", " (R)", " (S)"]   
    elif dataset_name == "BCICIV1":
        candidates = [" No", " Yes"]  
    elif dataset_name == "BCICIV2":
        candidates = [" (T)", " (U)", " (V)", " (W)"] 
    elif dataset_name == "STEW":
        candidates = [" No", " Yes"]  
    else:
        candidates = [" No", " Yes"]

    candidate_ids = []
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        
        if len(ids) == 0:
            print(f"[Warning] Candidate '{c}' encoded to empty ids!")
            candidate_ids.append(tokenizer.pad_token_id)
            continue

        if ")" in c and len(ids) >= 2:
            target_id = ids[-2]
            decoded = tokenizer.decode([target_id])
            if decoded.strip() == ")" or decoded.strip() == "(":
                 for tid in reversed(ids):
                     t_str = tokenizer.decode([tid])
                     if any(ch.isalnum() for ch in t_str):
                         target_id = tid
                         break
            candidate_ids.append(target_id)
            
            
        else:
            candidate_ids.append(ids[-1])
            
    return candidate_ids, candidates

def normalize_dataset_name(dataset_name: str) -> str:
    if "SEED" in dataset_name:
        return "SEED"
    if "TUAB" in dataset_name:
        return "TUAB"
    if "TUEV" in dataset_name:
        return "TUEV"
    if "TUSL" in dataset_name:
        return "TUSL"
    if "HMC" in dataset_name:
        return "HMC"
    if "Workload" in dataset_name or "workload" in dataset_name:
        return "Workload"
    if "EDF" in dataset_name:
        return "EDF"
    if "BCICIV1" in dataset_name or "BCI_IV1" in dataset_name:
        return "BCICIV1"
    if "BCICIV2" in dataset_name or "BCI_IV2" in dataset_name:
        return "BCICIV2"
    if "STEW" in dataset_name:
        return "STEW"
    return dataset_name

def is_bracket_answer_dataset(dataset_name: str) -> bool:
    dataset_name = normalize_dataset_name(dataset_name)
    bracket_datasets = {"TUEV", "TUSL", "HMC", "EDF", "BCICIV2"}
    return dataset_name in bracket_datasets


def get_num_classes_by_dataset(dataset_name: str) -> int:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "SEED":
        return 3
    if dataset_name in ["TUAB", "Workload"]:
        return 2
    if dataset_name == "TUEV":
        return 6
    if dataset_name == "TUSL":
        return 3
    if dataset_name == "HMC":
        return 5
    if dataset_name == "EDF":
        return 5 
    if dataset_name == "BCICIV1":
        return 2  
    if dataset_name == "BCICIV2":
        return 4  
    if dataset_name == "STEW":
        return 2  
    return 2


def is_binary_classification(dataset_name: str) -> bool:
    return get_num_classes_by_dataset(dataset_name) == 2


def get_binary_token_ids(tokenizer, dataset_name: str):
    dataset_name = normalize_dataset_name(dataset_name)
    if not is_binary_classification(dataset_name):
        return None, None
    
    pos_candidates = [" Yes", " yes", "Yes", "yes"]
    neg_candidates = [" No", " no", "No", "no"]
    
    def first_token(cands):
        for text in cands:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > 0:
                return ids[0]
        return None
    
    pos_id = first_token(pos_candidates)
    neg_id = first_token(neg_candidates)
    return pos_id, neg_id


def parse_answer_from_text(text, dataset_name):
    text = text.lower().strip()
    dataset_name_normalized = dataset_name
    if "SEED" in dataset_name:
        dataset_name_normalized = "SEED"
    elif "TUAB" in dataset_name:
        dataset_name_normalized = "TUAB"
    elif "TUEV" in dataset_name:
        dataset_name_normalized = "TUEV"
    elif "TUSL" in dataset_name:
        dataset_name_normalized = "TUSL"
    elif "HMC" in dataset_name:
        dataset_name_normalized = "HMC"
    elif "Workload" in dataset_name or "workload" in dataset_name:
        dataset_name_normalized = "Workload"
    elif "EDF" in dataset_name:
        dataset_name_normalized = "EDF"
    elif "BCICIV1" in dataset_name or "BCI_IV1" in dataset_name:
        dataset_name_normalized = "BCICIV1"
    elif "BCICIV2" in dataset_name or "BCI_IV2" in dataset_name:
        dataset_name_normalized = "BCICIV2"
    elif "STEW" in dataset_name:
        dataset_name_normalized = "STEW"
    
    if dataset_name_normalized == "SEED":
        if "positive" in text:
            return 0
        elif "neutral" in text:
            return 1
        elif "negative" in text:
            return 2
    elif dataset_name_normalized in ["TUAB", "Workload", "BCICIV1", "STEW"]:
        if "yes" in text:
            return 1
        elif "no" in text:
            return 0
    elif dataset_name_normalized == "TUEV":
        for i, option in enumerate(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
            if option in text:
                return i
    elif dataset_name_normalized == "TUSL":
        for i, option in enumerate(["(g)", "(h)", "(i)"]):
            if option in text:
                return i
    elif dataset_name_normalized == "HMC":
        for i, option in enumerate(["(j)", "(k)", "(l)", "(m)", "(n)"]):
            if option in text:
                return i
    elif dataset_name_normalized == "EDF":
        for i, option in enumerate(["(o)", "(p)", "(q)", "(r)", "(s)"]):
            if option in text:
                return i
    elif dataset_name_normalized == "BCICIV2":
        for i, option in enumerate(["(t)", "(u)", "(v)", "(w)"]):
            if option in text:
                return i
    return None


def compute_class_weights(data_loader, normalize_dataset_name_func, get_num_classes_func):
    class_counts = defaultdict(lambda: defaultdict(int))
    
    if master_process:
        print("Counting class distribution in the training set...")
    
    total_samples = 0
    for batch in data_loader:
        (
            X_eeg,
            label_ids,
            input_chans,
            input_time,
            input_mask,
            dataset_names,
            sample_names,
        ) = batch
        
        label_ids_list = label_ids.tolist() if isinstance(label_ids, torch.Tensor) else label_ids
        
        for ds_name, label_id in zip(dataset_names, label_ids_list):
            ds_name_norm = normalize_dataset_name_func(ds_name)
            class_id = int(label_id)
            class_counts[ds_name_norm][class_id] += 1
            total_samples += 1
    
    if master_process:
        print(f"Counted {total_samples} training samples in total.")
    
    class_weights = {}
    
    for ds_name, class_count_dict in class_counts.items():
        num_classes = get_num_classes_func(ds_name)
        total_samples_ds = sum(class_count_dict.values())
        
        weights = {}
        for class_id in range(num_classes):
            count = class_count_dict.get(class_id, 1)
            weight = total_samples_ds / (num_classes * count) if count > 0 else 1.0
            weights[class_id] = weight
        
        weight_sum = sum(weights.values())
        weight_mean = weight_sum / len(weights)
        if weight_mean > 0:
            weights = {k: v / weight_mean for k, v in weights.items()}
        
        class_weights[ds_name] = weights
        
        if master_process:
            print(f"\nDataset {ds_name}:")
            print(f"  Class distribution: {dict(class_count_dict)}")
            print(f"  Class weights: {weights}")
    
    return class_weights


def calculate_balanced_accuracy(pred_labels, true_labels, dataset_name):
    if len(pred_labels) != len(true_labels):
        return 0.0
    
    dataset_name = normalize_dataset_name(dataset_name)
    num_classes = get_num_classes_by_dataset(dataset_name)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for pred, true in zip(pred_labels, true_labels):
        if true < num_classes:
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1
    
    class_accs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
    
    balanced_acc = sum(class_accs) / len(class_accs) if class_accs else 0.0
    
    return balanced_acc


def compute_classification_metrics(pred_labels, true_labels, dataset_name, pred_scores=None):
    if len(pred_labels) == 0 or len(true_labels) == 0:
        return {}
    
    dataset_name_norm = normalize_dataset_name(dataset_name)
    metrics = {}
    
    metrics["balanced_acc"] = calculate_balanced_accuracy(
        pred_labels, true_labels, dataset_name_norm
    )
    
    if is_binary_classification(dataset_name_norm):
        y_true = np.array(true_labels)
        if pred_scores is not None and len(pred_scores) == len(true_labels):
            y_pred_scores = np.array(pred_scores)
        else:
            y_pred_scores = np.array(pred_labels)
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_pred_scores)
        except Exception:
            metrics["auroc"] = 0.0
        try:
            metrics["auc_pr"] = average_precision_score(y_true, y_pred_scores)
        except Exception:
            metrics["auc_pr"] = 0.0
    else:
        try:
            metrics["cohen_kappa"] = cohen_kappa_score(true_labels, pred_labels)
        except Exception:
            metrics["cohen_kappa"] = 0.0
        try:
            metrics["f1_macro"] = f1_score(true_labels, pred_labels, average="macro")
        except Exception:
            metrics["f1_macro"] = 0.0
        try:
            metrics["f1_weighted"] = f1_score(true_labels, pred_labels, average="weighted")
        except Exception:
            metrics["f1_weighted"] = 0.0
    
    return metrics


def evaluate_model_normal(unified_model, data_loader, eeg_tokenizer, tokenizer, device, ctx, args, get_batch_func, split_name='eval'):
    unified_model.eval()
    
    total_loss = 0.0
    total_instruction_loss = 0.0
    total_instruction_acc = 0.0
    
    results_buffer = defaultdict(lambda: {'true': [], 'pred': [], 'prob': []})
    
    num_samples = 0
    num_batches_to_print = 2
    
    known_datasets = ["SEED", "TUAB", "TUEV", "TUSL", "HMC", "Workload", "EDF", "BCICIV1", "BCICIV2", "STEW"]
    candidate_maps = {}
    candidate_texts = {}
    for ds in known_datasets:
        ids, texts = get_task_candidate_ids(tokenizer, ds)
        candidate_maps[ds] = torch.tensor(ids, device=device, dtype=torch.long)
        candidate_texts[ds] = texts

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            (
                X_eeg,
                label_ids,
                input_chans,
                input_time,
                input_mask,
                dataset_names,
                sample_names,
            ) = batch
            
            batch_size = len(dataset_names)
            
            prefix_text_input_ids, prefix_text_mask = get_batch_func(dataset_names, sample_names)
            
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            
            with ctx:
                inds_BN, rest_BND = eeg_tokenizer.get_codebook_msinds_and_msfeats(
                    X_eeg, input_chans, input_time, input_mask
                )
                inds_BN = inds_BN.detach()
                rest_BND = rest_BND.detach()

            dataset_names_norm = [normalize_dataset_name(ds) for ds in dataset_names]
            label_ids_list = label_ids.tolist() if isinstance(label_ids, torch.Tensor) else label_ids
            
            suffix_input_list = []
            suffix_label_mask_list = []
            for ds_name, y in zip(dataset_names, label_ids_list):
                text_ids, label_mask = build_instruction_text(tokenizer, ds_name, int(y))
                suffix_input_list.append(text_ids)
                suffix_label_mask_list.append(label_mask)
            
            max_suffix_len = max(len(t) for t in suffix_input_list)
            suffix_input_ids = torch.full((batch_size, max_suffix_len), tokenizer.pad_token_id, device=device, dtype=torch.long)
            suffix_label_mask = torch.zeros((batch_size, max_suffix_len), device=device, dtype=torch.long)
            
            for i, (t_ids, m) in enumerate(zip(suffix_input_list, suffix_label_mask_list)):
                l = len(t_ids)
                suffix_input_ids[i, :l] = t_ids.to(device)
                suffix_label_mask[i, :l] = m.to(device)

            model_dtype = (
                torch.float16
                if dtype == "float16"
                else torch.bfloat16
                if dtype == "bfloat16"
                else torch.float32
            )
            rest_BND = rest_BND.to(dtype=model_dtype)
            eeg_codes = torch.clamp(inds_BN.clone(), min=0, max=args.eeg_vocab_size - 1)

            with ctx:
                outputs = unified_model(
                    rest_BND, prefix_text_input_ids, prefix_text_mask, eeg_codes,
                    suffix_input_ids=suffix_input_ids, suffix_label_mask=suffix_label_mask
                )
            
            total_loss += outputs.loss.item() * batch_size
            total_instruction_loss += float(outputs.instruction_loss.detach()) * batch_size
            total_instruction_acc += outputs.instruction_acc * batch_size

            prompt_input_list = []
            for ds_name in dataset_names:
                _, p_ids = build_instruction_prompt_only(tokenizer, ds_name)
                ds_norm = normalize_dataset_name(ds_name)
                if ds_norm in ["TUEV", "TUSL", "HMC"]:
                    prefix_part = " (" 
                else:
                    prefix_part = ""
                
                prefix_ids = tokenizer.encode(prefix_part, add_special_tokens=False)
                
                full_prompt_ids = torch.cat([
                    p_ids, 
                    torch.tensor(prefix_ids, dtype=torch.long, device=p_ids.device)
                ])
                prompt_input_list.append(full_prompt_ids)
            
            max_prompt_len = max(len(t) for t in prompt_input_list)
            prompt_input_ids = torch.full((batch_size, max_prompt_len), tokenizer.pad_token_id, device=device, dtype=torch.long)
            
            for i, p_ids in enumerate(prompt_input_list):
                l = len(p_ids)
                prompt_input_ids[i, :l] = p_ids.to(device)
            bar_model = unified_model.llm_model
            router_model = unified_model.starouter
            
            prefix_embeds = bar_model.frozen_text_embed(prefix_text_input_ids)
            prompt_embeds = bar_model.frozen_text_embed(prompt_input_ids)
            
            with ctx:
                eeg_feat_embeds = router_model(rest_BND, prefix_text_input_ids, prefix_text_mask)
            eeg_code_embeds = bar_model.trainable_eeg_embed(eeg_codes).to(prefix_embeds.dtype)
            
            current_embeds = torch.cat([prefix_embeds, eeg_feat_embeds, eeg_code_embeds, prompt_embeds], dim=1)
            
            prefix_mask = (prefix_text_input_ids != tokenizer.pad_token_id).long()
            feat_mask = torch.ones((batch_size, eeg_feat_embeds.shape[1]), device=device, dtype=torch.long)
            code_mask = torch.ones((batch_size, eeg_codes.shape[1]), device=device, dtype=torch.long)
            prompt_mask = (prompt_input_ids != tokenizer.pad_token_id).long()
            attention_mask = torch.cat([prefix_mask, feat_mask, code_mask, prompt_mask], dim=1)

            if hasattr(bar_model.llm, "base_model") and hasattr(bar_model.llm.base_model, "model"):
                transformer_body = bar_model.llm.base_model.model.model
                lm_head = bar_model.llm.base_model.model.lm_head
            else:
                transformer_body = bar_model.llm.model
                lm_head = bar_model.llm.lm_head
                
            with ctx:
                transformer_outputs = transformer_body(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=False
                )
                hidden_states = transformer_outputs[0]
                all_logits = lm_head(hidden_states)

            len_prefix = prefix_embeds.shape[1]
            len_feat = eeg_feat_embeds.shape[1]
            len_code = eeg_code_embeds.shape[1]
            base_len = len_prefix + len_feat + len_code

            valid_prompt_lens = prompt_mask.sum(dim=1)

            for i, ds_name in enumerate(dataset_names_norm):
                valid_idx = base_len + valid_prompt_lens[i].item() - 1
                
                sample_logits = all_logits[i, valid_idx, :] 

                if ds_name not in candidate_maps:
                    c_ids, c_texts = get_task_candidate_ids(tokenizer, ds_name)
                    candidate_maps[ds_name] = torch.tensor(c_ids, device=device, dtype=torch.long)
                    candidate_texts[ds_name] = c_texts
                
                c_ids_tensor = candidate_maps[ds_name]
                
                relevant_logits = sample_logits[c_ids_tensor]
                
                probs = torch.softmax(relevant_logits, dim=-1)
                pred_label = torch.argmax(probs).item()
                
                num_classes = len(c_ids_tensor)
                true_label = int(label_ids_list[i])
                
                if true_label >= num_classes:
                    true_label = 0
                
                probs_np = probs.float().cpu().numpy()
                
                results_buffer[ds_name]['true'].append(true_label)
                results_buffer[ds_name]['pred'].append(pred_label)
                results_buffer[ds_name]['prob'].append(probs_np)
                
                if step < num_batches_to_print:
                    if ds_name in candidate_texts:
                        answer_texts = candidate_texts[ds_name]
                        true_answer_text = answer_texts[true_label] if true_label < len(answer_texts) else f"Label_{true_label}"
                        pred_answer_text = answer_texts[pred_label] if pred_label < len(answer_texts) else f"Label_{pred_label}"
                    else:
                        true_answer_text = f"Label_{true_label}"
                        pred_answer_text = f"Label_{pred_label}"
                    
                    print(f"[Batch {step}, Sample {i}] Dataset: {ds_name}")
                    print(f"  True Label: {true_label} -> {true_answer_text}")
                    print(f"  Pred Label: {pred_label} -> {pred_answer_text}")
                    print(f"  Probabilities: {[f'{p:.4f}' for p in probs_np]}")
                    print(f"  Correct: {'✓' if true_label == pred_label else '✗'}")
                    print()
            
            num_samples += batch_size

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_instruction_loss = total_instruction_loss / num_samples if num_samples > 0 else 0.0
    avg_instruction_acc = total_instruction_acc / num_samples if num_samples > 0 else 0.0

    dataset_metrics = {}
    dataset_balanced_accs = {}
    
    binary_metric_sum = defaultdict(float)
    multi_metric_sum = defaultdict(float)
    binary_count = 0
    multi_count = 0

    for ds_name, data in results_buffer.items():
        y_true = np.array(data['true'])
        y_pred = np.array(data['pred'])
        y_prob = np.array(data['prob'])
        
        pred_scores = None
        if y_prob.shape[1] == 2:
            pred_scores = y_prob[:, 1]
        bal_acc = calculate_balanced_accuracy(y_pred, y_true, ds_name)
        dataset_balanced_accs[ds_name] = bal_acc
        
        metrics = {"balanced_acc": bal_acc}
        
        if is_binary_classification(ds_name):
            if pred_scores is not None and len(pred_scores) > 0:
                try:
                    unique_labels = np.unique(y_true)
                    if len(unique_labels) >= 2:
                        metrics["auroc"] = roc_auc_score(y_true, pred_scores)
                        metrics["auc_pr"] = average_precision_score(y_true, pred_scores)
                    else:
                        metrics["auroc"] = 0.0
                        metrics["auc_pr"] = 0.0
                except (ValueError, Exception) as e:
                    print(f"[Warning] Failed to compute AUROC/AUC-PR for {ds_name}: {e}")
                    metrics["auroc"] = 0.0
                    metrics["auc_pr"] = 0.0
            else:
                metrics["auroc"] = 0.0
                metrics["auc_pr"] = 0.0
                
            binary_metric_sum["balanced_acc"] += bal_acc
            binary_metric_sum["auroc"] += metrics["auroc"]
            binary_metric_sum["auc_pr"] += metrics["auc_pr"]
            binary_count += 1
        else:
            try:
                y_true_int = y_true.astype(np.int64)
                y_pred_int = y_pred.astype(np.int64)
                
                true_unique = np.unique(y_true_int)
                pred_unique = np.unique(y_pred_int)
                all_unique = np.unique(np.concatenate([y_true_int, y_pred_int]))
                
                if len(all_unique) > 0 and (all_unique.min() != 0 or not np.array_equal(all_unique, np.arange(len(all_unique)))):
                    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(all_unique))}
                    y_true_mapped = np.array([label_map[label] for label in y_true_int])
                    y_pred_mapped = np.array([label_map[label] for label in y_pred_int])
                else:
                    y_true_mapped = y_true_int
                    y_pred_mapped = y_pred_int
                
                if len(y_true_mapped) == 0:
                    metrics["cohen_kappa"] = 0.0
                    metrics["f1_macro"] = 0.0
                    metrics["f1_weighted"] = 0.0
                else:
                    kappa = cohen_kappa_score(y_true_mapped, y_pred_mapped)
                    metrics["cohen_kappa"] = float(kappa) if not np.isnan(kappa) else 0.0
                    metrics["f1_macro"] = f1_score(y_true_mapped, y_pred_mapped, average="macro", zero_division=0)
                    metrics["f1_weighted"] = f1_score(y_true_mapped, y_pred_mapped, average="weighted", zero_division=0)
                    
                    if metrics["cohen_kappa"] == 0.0 and len(y_true_mapped) > 0:
                        unique_true = np.unique(y_true_mapped)
                        unique_pred = np.unique(y_pred_mapped)
                        if len(unique_true) > 1 or len(unique_pred) > 1:
                            print(f"[Debug] {ds_name}: kappa=0.0, true_labels={unique_true.tolist()}, pred_labels={unique_pred.tolist()}, "
                                  f"true_dist={np.bincount(y_true_mapped).tolist()}, pred_dist={np.bincount(y_pred_mapped).tolist()}")
            except (ValueError, Exception) as e:
                print(f"[Warning] Failed to compute metrics for {ds_name}: {e}")
                print(f"[Debug] y_true: unique={np.unique(y_true) if len(y_true) > 0 else 'empty'}, "
                      f"dtype={y_true.dtype}, shape={y_true.shape}")
                print(f"[Debug] y_pred: unique={np.unique(y_pred) if len(y_pred) > 0 else 'empty'}, "
                      f"dtype={y_pred.dtype}, shape={y_pred.shape}")
                metrics["cohen_kappa"] = 0.0
                metrics["f1_macro"] = 0.0
                metrics["f1_weighted"] = 0.0
            
            multi_metric_sum["balanced_acc"] += bal_acc
            multi_metric_sum["cohen_kappa"] += metrics["cohen_kappa"]
            multi_metric_sum["f1_macro"] += metrics["f1_macro"]
            multi_metric_sum["f1_weighted"] += metrics["f1_weighted"]
            multi_count += 1
            
        dataset_metrics[ds_name] = metrics

    overall_balanced_acc = np.mean(list(dataset_balanced_accs.values())) if dataset_balanced_accs else 0.0
    
    overall_binary_metrics = {
        "balanced_acc": (binary_metric_sum["balanced_acc"] / binary_count) if binary_count else 0.0,
        "auroc": (binary_metric_sum["auroc"] / binary_count) if binary_count else 0.0,
        "auc_pr": (binary_metric_sum["auc_pr"] / binary_count) if binary_count else 0.0,
    }
    overall_multi_metrics = {
        "balanced_acc": (multi_metric_sum["balanced_acc"] / multi_count) if multi_count else 0.0,
        "cohen_kappa": (multi_metric_sum["cohen_kappa"] / multi_count) if multi_count else 0.0,
        "f1_macro": (multi_metric_sum["f1_macro"] / multi_count) if multi_count else 0.0,
        "f1_weighted": (multi_metric_sum["f1_weighted"] / multi_count) if multi_count else 0.0,
    }

    return {
        'loss': avg_loss,
        'instruction_loss': avg_instruction_loss,
        'instruction_acc': avg_instruction_acc,
        'balanced_acc': overall_balanced_acc,
        'dataset_balanced_accs': dataset_balanced_accs,
        'dataset_metrics': dataset_metrics,
        'overall_binary_metrics': overall_binary_metrics,
        'overall_multi_metrics': overall_multi_metrics,
        'num_samples': num_samples,
    }

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
    if (
        state.deepspeed_plugin is not None
        and state.deepspeed_plugin.deepspeed_config is not None
    ):
        state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = args.eeg_batch_size

    init(args, accelerator)

    checkpoint_out_dir = os.path.join(
        args.out_dir, "checkpoints/{}".format(args.wandb_runname)
    )
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
    text_embedding_config = {
        'model_path': args.text_model_path
    }
    def get_batch(dataset_name, sample_name):
        text_input_ids = torch.zeros((len(dataset_name), 1024), dtype=torch.long)
        text_mask = torch.zeros((len(dataset_name), 1024), dtype=torch.long)
        pad_id = text_tokenizer.pad_token_id if text_tokenizer is not None else 0
        for idx, (ds, sp) in enumerate(zip(dataset_name, sample_name)):
            report = text_reports.get((ds, sp))
            if report is None:
                if master_process:
                    print(f"[Warn] Text entry not found for {(ds, sp)}, using pad as placeholder.")
                text_input_ids[idx] = torch.full((1024,), pad_id, dtype=torch.long)
                text_mask[idx] = torch.zeros(1024, dtype=torch.long)
                continue
            text_mask[idx] = torch.tensor(report["text_mask"], dtype=torch.long)
            text_input_ids[idx] = torch.tensor(report["ids"], dtype=torch.long)
        text_input_ids = text_input_ids.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        return text_input_ids, text_mask


    eeg_tokenizer = load_eeg_tokenizer(args)

    text_model_config = AutoConfig.from_pretrained(
        args.text_model_path, trust_remote_code=True
    )
    text_hidden_size = (
        getattr(text_model_config, "hidden_size", None)
        or getattr(text_model_config, "n_embd", None)
        or getattr(text_model_config, "d_model", None)
    )
    if text_hidden_size is None:
        raise ValueError("Cannot infer hidden size from text model config.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    new_tokens = [f"<eeg_{i}>" for i in range(args.eeg_vocab_size)]
    if "<eeg_0>" not in tokenizer.get_vocab():
        if master_process:
            print(f"[Info] Extending vocabulary: original size {len(tokenizer)}")
        num_added = tokenizer.add_tokens(new_tokens)
        if master_process:
            print(f"[Info] Successfully added {num_added} EEG tokens")
    eeg_token_start_id = tokenizer.convert_tokens_to_ids("<eeg_0>")
    if master_process:
        print(f"EEG Token Start ID: {eeg_token_start_id}")

    model_dtype = (
        torch.float16
        if dtype == "float16"
        else torch.bfloat16
        if dtype == "bfloat16"
        else torch.float32
    )
    
    if args.pretrained_lora_path is not None and os.path.isdir(
        args.pretrained_lora_path
    ):
        if master_process:
            print(f"Loading and merging LoRA from pretrained directory: {args.pretrained_lora_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.text_model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation="flash_attention_2",
        )
        base_model.resize_token_embeddings(len(tokenizer))
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.pretrained_lora_path,
            is_trainable=False,
        )
        peft_model = peft_model.merge_and_unload()
        llm_model = peft_model
    else:
        if master_process:
            print(
                "No valid pretrained LoRA directory provided; loading base model from text_model_path for instruction fine-tuning."
            )
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.text_model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype,
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["lm_head"],
    )
    llm_model = get_peft_model(llm_model, peft_config)
    if master_process:
        llm_model.print_trainable_parameters()

    bar_model = EEGBARLLM_instruction(
        llm_model,
        tokenizer,
        eeg_token_start_id,
        args.eeg_vocab_size,
        use_eeg_code_loss=False,
    ).to(device)

    starouter = STARouter(
        d_model=text_hidden_size,
        num_queries=16,
        num_eeg_channels=91,
        nhead=8,
        dropout=0.1,
        text_embedding_config=text_embedding_config,
        text_embedding_module=llm_model.get_input_embeddings(),
        freeze_text=True,
    )
    starouter = starouter.to(device).to(model_dtype)
    if args.pretrained_ckpt_dir is not None and os.path.isdir(args.pretrained_ckpt_dir):
        if master_process:
            print(f"Loading pretrained STARouter weights from: {args.pretrained_ckpt_dir}")
        try:
            pytorch_model_dir = os.path.join(args.pretrained_ckpt_dir, "pytorch_model")
            if os.path.isdir(pytorch_model_dir):
                model_state_file = os.path.join(pytorch_model_dir, "mp_rank_00_model_states.pt")
                if os.path.isfile(model_state_file):
                    checkpoint = torch.load(model_state_file, map_location=device, weights_only=False)
                    
                    if isinstance(checkpoint, dict):
                        state_to_search = checkpoint.get("module", checkpoint)

                        starouter_state = {}
                        for key, value in state_to_search.items():
                            if key.startswith("starouter."):
                                new_key = key[len("starouter."):]
                                starouter_state[new_key] = value
                        
                        if starouter_state:
                            current_state = starouter.state_dict()
                            total_params_in_model = sum(p.numel() for p in starouter.parameters())
                            
                            incompatible_keys = starouter.load_state_dict(starouter_state, strict=False)
                            
                            loaded_param_count = 0
                            loaded_param_numel = 0
                            missing_keys = set(incompatible_keys.missing_keys) if hasattr(incompatible_keys, 'missing_keys') else set()
                            unexpected_keys = set(incompatible_keys.unexpected_keys) if hasattr(incompatible_keys, 'unexpected_keys') else set()
                            
                            for key, value in starouter_state.items():
                                if key not in missing_keys:
                                    loaded_param_count += 1
                                    loaded_param_numel += value.numel()
                            
                            unloaded_param_count = len(missing_keys)
                            unloaded_param_numel = 0
                            for key in missing_keys:
                                if key in current_state:
                                    unloaded_param_numel += current_state[key].numel()
                            
                            if master_process:
                                print(f"\n{'='*70}")
                                print(f"STARouter weight loading statistics:")
                                print(f"{'='*70}")
                                print(f"✓ Loaded: {loaded_param_count} parameter tensors, {loaded_param_numel:,} values")
                                print(f"✗ Not loaded: {unloaded_param_count} parameter tensors, {unloaded_param_numel:,} values")
                                print(f"Total model parameters: {total_params_in_model:,} values")
                                print(f"Coverage: {loaded_param_numel / total_params_in_model * 100:.2f}%")
                                
                                if missing_keys:
                                    print(f"\nMissing parameter tensors (first 10):")
                                    for i, key in enumerate(sorted(missing_keys)[:10]):
                                        if key in current_state:
                                            print(f"  - {key}: {current_state[key].shape}")
                                    if len(missing_keys) > 10:
                                        print(f"  ... {len(missing_keys) - 10} more not shown")
                                
                                if unexpected_keys:
                                    print(f"\nExtra parameter tensors in checkpoint (first 10):")
                                    for i, key in enumerate(sorted(unexpected_keys)[:10]):
                                        print(f"  - {key}")
                                    if len(unexpected_keys) > 10:
                                        print(f"  ... {len(unexpected_keys) - 10} more not shown")
                                
                                print(f"{'='*70}\n")
                        else:
                            if master_process:
                                print("Warning: STARouter weights not found in checkpoint, using random initialization.")
                    else:
                        if master_process:
                            print("Warning: checkpoint is not a dict, skip loading STARouter weights.")
                else:
                    if master_process:
                        print(f"Warning: {model_state_file} not found, using random initialization for STARouter.")
            else:
                if master_process:
                    print(f"Warning: pytorch_model directory not found, using random initialization for STARouter.")
        except Exception as e:
            if master_process:
                print(f"Warning: failed to load STARouter weights: {e}")
    elif master_process:
        print("No pretrained STARouter checkpoint provided, using random initialization.")
    
    if not args.train_starouter:
        for p in starouter.parameters():
            p.requires_grad = False
        if master_process:
            print("STARouter is frozen (used only for feature extraction).")
    else:
        for p in starouter.parameters():
            p.requires_grad = True
        if master_process:
            print("STARouter is set to trainable.")

    if master_process:
        print(f"Detected text hidden size: {text_hidden_size}")
        print(
            f"STARouter params: {sum(p.numel() for p in starouter.parameters()):,}, "
            f"trainable: {sum(p.numel() for p in starouter.parameters() if p.requires_grad):,}"
        )

    data_loader_train = create_eeg_loader(
        batch_size=args.eeg_batch_size,
        dataset_dir=args.dataset_dir,
        ddp=ddp,
        ddp_rank=ddp_rank if ddp else 0,
        ddp_world_size=ddp_world_size if ddp else 1,
        group_by_hash=True,
        num_workers=4,
        split='train',
        enable_downsample=True
    )
    dataset_train = data_loader_train.dataset
    
    if master_process:
        print("\n" + "="*70)
        print("Computing class weights...")
        print("="*70)
    class_weights = compute_class_weights(
        data_loader_train,
        normalize_dataset_name,
        get_num_classes_by_dataset
    )
    if master_process:
        print("="*70 + "\n")
    
    data_loader_eval = create_eeg_loader(
        batch_size=args.eeg_batch_size,
        dataset_dir=args.dataset_dir,
        ddp=ddp,
        ddp_rank=ddp_rank if ddp else 0,
        ddp_world_size=ddp_world_size if ddp else 1,
        group_by_hash=True,  
        num_workers=4,
        split='eval',
    )
    
    data_loader_test = create_eeg_loader(
        batch_size=args.eeg_batch_size,
        dataset_dir=args.dataset_dir,
        ddp=ddp,
        ddp_rank=ddp_rank if ddp else 0,
        ddp_world_size=ddp_world_size if ddp else 1,
        group_by_hash=True,  
        num_workers=4,
        split='test',
    )

    lora_params = [p for p in bar_model.llm.parameters() if p.requires_grad]
    eeg_embed_params = [bar_model.trainable_eeg_embed.weight]
    
    param_groups = [
        {
            "params": lora_params + eeg_embed_params,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    
    if args.train_starouter:
        shared_embedding_params = set()
        if hasattr(starouter, 'wte') and starouter.wte is not None:
            shared_embedding_params = set(starouter.wte.parameters())
        
        existing_params = set(lora_params + eeg_embed_params)
        
        starouter_params = [
            p for p in starouter.parameters() 
            if p.requires_grad 
            and p not in shared_embedding_params 
            and p not in existing_params
        ]
        starouter_lr = args.starouter_lr if args.starouter_lr is not None else args.learning_rate
        param_groups.append(
            {
                "params": starouter_params,
                "lr": starouter_lr,
                "weight_decay": args.weight_decay,
            }
        )
        if master_process:
            print(f"STARouter learning rate: {starouter_lr:.2e} (main model LR: {args.learning_rate:.2e})")
            starouter_trainable_numel = sum(p.numel() for p in starouter_params)
            starouter_total_numel = sum(p.numel() for p in starouter.parameters())
            print(f"STARouter trainable parameters: {starouter_trainable_numel:,} (tensors: {len(starouter_params)})")
            print(f"STARouter total parameters: {starouter_total_numel:,} (tensors: {sum(1 for _ in starouter.parameters())})")
            if shared_embedding_params:
                shared_embedding_numel = sum(p.numel() for p in shared_embedding_params)
                print(f"Excluded shared embedding parameters: {shared_embedding_numel:,} (tensors: {len(shared_embedding_params)})")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.beta1, args.beta2),
    )

    unified_model = UnifiedEEGBAR(starouter, bar_model)
    
    if master_process:
        def count_parameters_excluding_shared_embeddings(model, starouter_model, eeg_tokenizer_model):
            unified_params = set()
            for p in model.parameters():
                unified_params.add(p)
            
            eeg_tokenizer_params = set()
            for p in eeg_tokenizer_model.parameters():
                eeg_tokenizer_params.add(p)
            
            overlapping_params = unified_params & eeg_tokenizer_params
            overlapping_numel = sum(p.numel() for p in overlapping_params)
            
            shared_embedding_params = set()
            if hasattr(starouter_model, 'wte') and starouter_model.wte is not None:
                shared_embedding_params = set(starouter_model.wte.parameters())
            elif hasattr(starouter_model, 'text_embedding_module') and starouter_model.text_embedding_module is not None:
                shared_embedding_params = set(starouter_model.text_embedding_module.parameters())
            
            unified_params_numel = sum(p.numel() for p in unified_params)
            eeg_tokenizer_numel = sum(p.numel() for p in eeg_tokenizer_params)
            shared_embedding_numel = sum(p.numel() for p in shared_embedding_params)
            
            total_params = unified_params_numel + eeg_tokenizer_numel + shared_embedding_numel
            total_params_with_duplicates = unified_params_numel + eeg_tokenizer_numel
            
            llm_params = sum(p.numel() for p in model.llm_model.parameters())
            starouter_all_params = sum(p.numel() for p in starouter_model.parameters())
            starouter_unique_params = starouter_all_params - shared_embedding_numel
            
            unified_model_unique_params = unified_params_numel
            
            return {
                'total_params': total_params,
                'total_params_with_duplicates': total_params_with_duplicates,
                'unified_model_params': unified_params_numel,
                'unified_model_unique_params': unified_model_unique_params,
                'eeg_tokenizer_params': eeg_tokenizer_numel,
                'shared_embedding_params': shared_embedding_numel,
                'overlapping_params': overlapping_numel,
                'llm_params': llm_params,
                'starouter_all_params': starouter_all_params,
                'starouter_unique_params': starouter_unique_params,
            }
        
        param_stats = count_parameters_excluding_shared_embeddings(unified_model, starouter, eeg_tokenizer)
        print(f"\n{'='*70}")
        print(f"Model parameter statistics (excluding shared embeddings):")
        print(f"{'='*70}")
        print(f"Total params (deduplicated, including EEG tokenizer): {param_stats['total_params']:,}")
        print(f"Total params (deduplicated, excluding EEG tokenizer): {param_stats['unified_model_unique_params']:,}")
        print(f"Total params (with duplicates): {param_stats['total_params_with_duplicates']:,}")
        print(f"  - UnifiedEEGBAR params (deduplicated = LLM params + STARouter unique params): {param_stats['unified_model_params']:,}")
        print(f"  - EEG tokenizer params: {param_stats['eeg_tokenizer_params']:,}")
        print(f"  - Shared embedding params: {param_stats['shared_embedding_params']:,}")
        print(f"\nPer-module breakdown:")
        print(f"  - LLM params: {param_stats['llm_params']:,}")
        print(f"  - STARouter total params: {param_stats['starouter_all_params']:,}")
        print(f"  - STARouter unique params: {param_stats['starouter_unique_params']:,}")
        print(f"{'='*70}\n")
    
    unified_model, optimizer = accelerator.prepare(unified_model, optimizer)

    if args.resume_path is not None and args.load_model_only:
        if master_process:
            print(f"Loading model weights from {args.resume_path} (without optimizer state)...")
        try:
            if hasattr(unified_model, "module") and hasattr(unified_model.module, "load_checkpoint"):
                unified_model.module.load_checkpoint(
                    args.resume_path,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )
            elif hasattr(accelerator, "deepspeed_engine"):
                accelerator.deepspeed_engine.load_checkpoint(
                    args.resume_path,
                    load_optimizer_states=False,
                    load_lr_scheduler_states=False,
                )
            else:
                accelerator.load_state(args.resume_path)
            if master_process:
                print("✓ Model weights loaded successfully (optimizer state skipped).")
        except Exception as e:
            error_str = str(e)
            if master_process:
                print(f"[Error] Failed to load model weights: {error_str}")
                print(f"  Hint: If you encounter GPU world-size mismatch issues, please ensure the same number of GPUs is used.")
            raise

    if args.wandb_log and master_process:
        import wandb

        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb_settings = wandb.Settings(init_timeout=300)
        wandb.init(
            project=args.wandb_project,
            entity=args.entity,
            name=args.wandb_runname,
            dir=os.path.join(args.out_dir, "wandb"),
            settings=wandb_settings,
        )

    num_training_steps_per_epoch = (
        len(dataset_train) // args.eeg_batch_size // ddp_world_size
    )
    start_epoch = 0
    iter_num = 0
    
    if args.resume_path is not None and not args.load_model_only:
        if master_process:
            print(f"Restoring training state/loading weights from {args.resume_path}...")
        
        try:
            accelerator.load_state(args.resume_path)
            if master_process:
                print("✓ Training state loaded successfully.")
        except Exception as e:
            error_str = str(e)
            if "world size" in error_str or "DP world size" in error_str:
                if master_process:
                    print(f"\n[Error] GPU world size in DeepSpeed checkpoint does not match current setting.")
                    print(f"  Error message: {error_str}")
                    print(f"  Suggested solutions:")
                    print(f"    1. Use the same number of GPUs as when the checkpoint was created (recommended for resuming training).")
                    print(f"    2. Use --load_model_only to load only model weights (skip optimizer state).")
                    print(f"    3. For evaluation only, use --load_model_only --test_only.")
                raise
            else:
                if master_process:
                    print(f"[Error] Failed to load checkpoint: {error_str}")
                raise
        
        meta_path = os.path.join(args.resume_path, "training_metadata.pt")
        if os.path.exists(meta_path):
            metadata = torch.load(meta_path, map_location="cpu")
            saved_epoch = metadata.get("epoch", 0)
            iter_num = metadata.get("iter_num", 0)
            
            if not args.test_only:
                start_epoch = saved_epoch + 1
                if master_process:
                    print(f"Successfully restored metadata: last finished at epoch {saved_epoch}, iter {iter_num}.")
                    print(f"Training will resume from epoch {start_epoch}.")
        else:
            if master_process:
                print(f"[Warning] Metadata {meta_path} not found; starting from epoch 0 (weights are loaded).")
    
    total_epochs_for_schedule = start_epoch + args.epochs
    if master_process:
        print(f"LR scheduler: total epochs={total_epochs_for_schedule} (already done={start_epoch}, new={args.epochs}).")
    
    lr_schedule_values = cosine_scheduler(
        args.learning_rate,
        args.min_lr,
        total_epochs_for_schedule,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    
    if args.resume_path is not None and not args.test_only and iter_num > 0:
        if iter_num < len(lr_schedule_values):
            initial_lr = lr_schedule_values[iter_num]
        else:
            initial_lr = args.min_lr
            if master_process:
                print(f"[Warning] iter_num ({iter_num}) exceeds LR schedule length ({len(lr_schedule_values)}); using min LR.")
        
        optimizer.param_groups[0]["lr"] = initial_lr
        if args.train_starouter and len(optimizer.param_groups) > 1:
            if args.starouter_lr is not None:
                starouter_lr_ratio = args.starouter_lr / args.learning_rate
                starouter_initial_lr = initial_lr * starouter_lr_ratio
            else:
                starouter_initial_lr = initial_lr
            optimizer.param_groups[1]["lr"] = starouter_initial_lr
            if master_process:
                print(f"Restored initial LR: main model {initial_lr:.2e}, STARouter {starouter_initial_lr:.2e} (iter_num={iter_num}, schedule length={len(lr_schedule_values)})")
        else:
            if master_process:
                print(f"Restored initial LR: {initial_lr:.2e} (iter_num={iter_num}, schedule length={len(lr_schedule_values)})")

    t0 = time.time()
    skip_count = 0
    if args.test_only:
        accelerator.wait_for_everyone()
        if master_process:
            print(f"\n{'='*70}")
            print(f"Mode: Test Only (skip training).")
            print(f"Load path: {args.resume_path if args.resume_path else 'Not specified (using initialized weights)'}")
            print(f"{'='*70}")
        test_results = evaluate_model_normal(
            unified_model,
            data_loader_test,
            eeg_tokenizer,
            tokenizer,
            device,
            ctx,
            args,
            get_batch,
            split_name='test',
        )
        if master_process:
            print(f"\ntest set evaluation results:")
            print(f"  Loss: {test_results['loss']:.4f}")
            print(f"  Instruction Acc: {test_results['instruction_acc']:.2%}")
            print(f"  Balanced Acc: {test_results['balanced_acc']:.2%}")
            print(f"  Balanced accuracy per dataset:")
            for ds_name, acc in test_results.get('dataset_balanced_accs', {}).items():
                print(f"    {ds_name}: {acc:.2%}")
            if 'overall_binary_metrics' in test_results:
                print(f"  Overall binary: BalAcc {test_results['overall_binary_metrics'].get('balanced_acc', 0.0):.2%}, "
                    f"AUROC {test_results['overall_binary_metrics'].get('auroc', 0.0):.2%}, "
                    f"AUC-PR {test_results['overall_binary_metrics'].get('auc_pr', 0.0):.2%}")
            if 'overall_multi_metrics' in test_results:
                print(f"  Overall multi-class: BalAcc {test_results['overall_multi_metrics'].get('balanced_acc', 0.0):.2%}, "
                    f"Kappa {test_results['overall_multi_metrics'].get('cohen_kappa', 0.0):.2%}, "
                    f"F1-weighted {test_results['overall_multi_metrics'].get('f1_weighted', 0.0):.2%}")
            if 'dataset_metrics' in test_results:
                print(f"  Detailed metrics per dataset:")
                for ds_name, metrics in test_results['dataset_metrics'].items():
                    if is_binary_classification(ds_name):
                        print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                            f"AUROC {metrics.get('auroc', 0.0):.2%}, "
                            f"AUC-PR {metrics.get('auc_pr', 0.0):.2%}")
                    else:
                        print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                            f"Kappa {metrics.get('cohen_kappa', 0.0):.2%}, "
                            f"F1-weighted {metrics.get('f1_weighted', 0.0):.2%}")
            print(f"{'='*70}\n")

        return
    for epoch in range(start_epoch, args.epochs):
        if ddp and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        for step, batch in enumerate(data_loader_train):
            (
                X_eeg,
                label_ids,
                input_chans,
                input_time,
                input_mask,
                dataset_names,
                sample_names,
            ) = batch
            prefix_text_input_ids, prefix_text_mask = get_batch(dataset_names, sample_names)
            if args.decay_lr:
                if iter_num < len(lr_schedule_values):
                    lr = lr_schedule_values[iter_num]
                else:
                    lr = args.min_lr
            else:
                lr = args.learning_rate
            
            optimizer.param_groups[0]["lr"] = lr
            
            if args.train_starouter and len(optimizer.param_groups) > 1:
                if args.starouter_lr is not None:
                    starouter_lr_ratio = args.starouter_lr / args.learning_rate
                    starouter_lr = lr * starouter_lr_ratio
                else:
                    starouter_lr = lr
                optimizer.param_groups[1]["lr"] = starouter_lr

            X_eeg = X_eeg.float().to(device, non_blocking=True)
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)

            with torch.no_grad():
                with ctx:
                    inds_BN, rest_BND = eeg_tokenizer.get_codebook_msinds_and_msfeats(
                        X_eeg, input_chans, input_time, input_mask
                    )
                    inds_BN = inds_BN.detach()
                    rest_BND = rest_BND.detach()
                    
                    if step % 2000 == 0:
                        inds_flat = inds_BN.cpu().flatten().numpy()
                        unique_indices, counts = np.unique(inds_flat, return_counts=True)
                        num_unique = len(unique_indices)
                        num_total_codes = args.eeg_vocab_size
                        unused_codes = num_total_codes - num_unique
                        usage_rate = num_unique / num_total_codes * 100
                        
                        sorted_indices = np.argsort(counts)[::-1]
                        top_10_indices = unique_indices[sorted_indices[:10]]
                        top_10_counts = counts[sorted_indices[:10]]
                        
                        probs = counts / counts.sum()
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        max_entropy = np.log(num_unique) if num_unique > 0 else 0
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        
                        print(
                            f"[Codebook Stats] Step {step}: "
                            f"Unique codes: {num_unique}/{num_total_codes} ({usage_rate:.2f}%), "
                            f"Unused: {unused_codes}, "
                            f"Entropy: {entropy:.2f} (normalized: {normalized_entropy:.3f}), "
                            f"Top3 codes: {top_10_indices[:3].tolist()} (counts: {top_10_counts[:3].tolist()})"
                        )

            suffix_input_list = []
            suffix_label_mask_list = []
            label_ids_list = label_ids.tolist() if isinstance(label_ids, torch.Tensor) else label_ids
            for ds_name, y in zip(dataset_names, label_ids_list):
                text_ids, label_mask = build_instruction_text(
                    tokenizer, ds_name, int(y)
                )
                suffix_input_list.append(text_ids)
                suffix_label_mask_list.append(label_mask)

            max_suffix_len = max(t.size(0) for t in suffix_input_list) if suffix_input_list else 1
            pad_id = tokenizer.pad_token_id
            suffix_input_ids = torch.full(
                (len(suffix_input_list), max_suffix_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            suffix_label_mask = torch.zeros(
                (len(suffix_input_list), max_suffix_len),
                dtype=torch.long,
                device=device,
            )
            for i, (t_ids, m) in enumerate(zip(suffix_input_list, suffix_label_mask_list)):
                length = t_ids.size(0)
                suffix_input_ids[i, :length] = t_ids.to(device)
                suffix_label_mask[i, :length] = m.to(device)


            model_dtype = (
                torch.float16
                if dtype == "float16"
                else torch.bfloat16
                if dtype == "bfloat16"
                else torch.float32
            )
            rest_BND = rest_BND.to(dtype=model_dtype)
            eeg_codes = torch.clamp(
                inds_BN.clone(), min=0, max=args.eeg_vocab_size - 1
            )

            with accelerator.accumulate(unified_model):
                sample_weights_list = []
                for ds_name, label_id in zip(dataset_names, label_ids_list):
                    ds_name_norm = normalize_dataset_name(ds_name)
                    class_id = int(label_id)
                    weight = class_weights.get(ds_name_norm, {}).get(class_id, 1.0)
                    sample_weights_list.append(weight)
                
                sample_weights_tensor = torch.tensor(
                    sample_weights_list, 
                    device=device, 
                    dtype=torch.float32
                )
                
                with ctx:
                    outputs = unified_model(
                        rest_BND,
                        prefix_text_input_ids,
                        prefix_text_mask,
                        eeg_codes,
                        suffix_input_ids=suffix_input_ids,
                        suffix_label_mask=suffix_label_mask,
                        sample_weights=sample_weights_tensor,
                        use_eeg_code_segment=True,
                    )
                    
                    loss = outputs.loss / args.gradient_accumulation_steps

                if (
                    torch.isnan(loss)
                    or torch.isinf(loss)
                    or loss.item() > 150
                ):
                    skip_count += 1
                    if master_process:
                        print(
                            f"[Warning] skip batch {skip_count}, abnormal loss={loss.item()}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients and args.grad_clip != 0.0:
                    grad_norm = accelerator.clip_grad_norm_(
                        unified_model.parameters(), args.grad_clip
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                balanced_acc = outputs.instruction_acc
                
                print(
                    f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: "
                    f"Total: {loss.item()*args.gradient_accumulation_steps:.4f}, "
                    f"Main: {outputs.loss.item():.4f}, "
                    f"Text: {float(outputs.text_loss.detach()):.4f}, "
                    f"EEG: {float(outputs.eeg_loss.detach()):.4f} (Acc: {outputs.eeg_acc:.2%}), "
                    f"Instruction: {float(outputs.instruction_loss.detach()):.4f} (Acc: {outputs.instruction_acc:.2%}, BalAcc: {balanced_acc:.2%}), "
                    f"Skipped: {skip_count}"
                )
                if args.wandb_log:
                    import wandb

                    log_dict = {
                        "iter": iter_num,
                        "train/total_loss": loss.item() * args.gradient_accumulation_steps,
                        "train/instruction_loss": float(outputs.instruction_loss.detach()),
                        "train/instruction_accuracy": outputs.instruction_acc,
                        "lr": lr,
                        "epoch": epoch,
                    }
                    if args.train_starouter and len(optimizer.param_groups) > 1:
                        log_dict["lr_starouter"] = optimizer.param_groups[1]["lr"]
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = (
                            grad_norm.item()
                            if torch.is_tensor(grad_norm)
                            else grad_norm
                        )
                    wandb.log(log_dict)

            t1 = time.time()
            _ = t1 - t0
            t0 = t1
            iter_num += 1

        accelerator.wait_for_everyone()
        if master_process:
            print(f"\n{'='*70}")
            print(f"Evaluating on epoch {epoch}...")
            print(f"{'='*70}")
        
        eval_results = evaluate_model_normal(
            unified_model,
            data_loader_train,
            eeg_tokenizer,
            tokenizer,
            device,
            ctx,
            args,
            get_batch,
            split_name='eval',
        )
        
        if master_process:
            print(f"Validation results (epoch {epoch}):")
            print(f"  Loss: {eval_results['loss']:.4f}")
            print(f"  Instruction Loss: {eval_results['instruction_loss']:.4f}")
            print(f"  Instruction Acc: {eval_results['instruction_acc']:.2%}")
            print(f"  Balanced Acc: {eval_results['balanced_acc']:.2%}")
            print(f"  Balanced accuracy per dataset:")
            for ds_name, acc in eval_results.get('dataset_balanced_accs', {}).items():
                print(f"    {ds_name}: {acc:.2%}")
            if 'overall_binary_metrics' in eval_results:
                print(f"  Overall binary: BalAcc {eval_results['overall_binary_metrics'].get('balanced_acc', 0.0):.2%}, "
                    f"AUROC {eval_results['overall_binary_metrics'].get('auroc', 0.0):.2%}, "
                    f"AUC-PR {eval_results['overall_binary_metrics'].get('auc_pr', 0.0):.2%}")
            if 'overall_multi_metrics' in eval_results:
                print(f"  Overall multi-class: BalAcc {eval_results['overall_multi_metrics'].get('balanced_acc', 0.0):.2%}, "
                    f"Kappa {eval_results['overall_multi_metrics'].get('cohen_kappa', 0.0):.2%}, "
                    f"F1-macro {eval_results['overall_multi_metrics'].get('f1_macro', 0.0):.2%}, "
                    f"F1-weighted {eval_results['overall_multi_metrics'].get('f1_weighted', 0.0):.2%}")
            if 'dataset_metrics' in eval_results:
                print(f"  Detailed metrics per dataset:")
                for ds_name, metrics in eval_results['dataset_metrics'].items():
                    if is_binary_classification(ds_name):
                        print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                            f"AUROC {metrics.get('auroc', 0.0):.2%}, "
                            f"AUC-PR {metrics.get('auc_pr', 0.0):.2%}")
                    else:
                        print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                            f"Kappa {metrics.get('cohen_kappa', 0.0):.2%}, "
                            f"F1-macro {metrics.get('f1_macro', 0.0):.2%}, "
                            f"F1-weighted {metrics.get('f1_weighted', 0.0):.2%}")
            print(f"{'='*70}\n")
        
        if args.wandb_log and master_process:
            import wandb
            log_dict = {
                "epoch": epoch,
                "eval/loss": eval_results['loss'],
                "eval/instruction_loss": eval_results['instruction_loss'],
                "eval/instruction_accuracy": eval_results['instruction_acc'],
                "eval/balanced_accuracy": eval_results['balanced_acc'],
                "eval/binary_balanced_acc": eval_results.get('overall_binary_metrics', {}).get('balanced_acc', 0.0),
                "eval/binary_auroc": eval_results.get('overall_binary_metrics', {}).get('auroc', 0.0),
                "eval/binary_auc_pr": eval_results.get('overall_binary_metrics', {}).get('auc_pr', 0.0),
                "eval/multi_balanced_acc": eval_results.get('overall_multi_metrics', {}).get('balanced_acc', 0.0),
                "eval/multi_cohen_kappa": eval_results.get('overall_multi_metrics', {}).get('cohen_kappa', 0.0),
                "eval/multi_f1_macro": eval_results.get('overall_multi_metrics', {}).get('f1_macro', 0.0),
                "eval/multi_f1_weighted": eval_results.get('overall_multi_metrics', {}).get('f1_weighted', 0.0),
            }
            for ds_name, acc in eval_results['dataset_balanced_accs'].items():
                log_dict[f"eval/balanced_acc_{ds_name}"] = acc
            for ds_name, metrics in eval_results['dataset_metrics'].items():
                if is_binary_classification(ds_name):
                    log_dict[f"eval/{ds_name}_auroc"] = metrics.get("auroc", 0.0)
                    log_dict[f"eval/{ds_name}_auc_pr"] = metrics.get("auc_pr", 0.0)
                else:
                    log_dict[f"eval/{ds_name}_cohen_kappa"] = metrics.get("cohen_kappa", 0.0)
                    log_dict[f"eval/{ds_name}_f1_macro"] = metrics.get("f1_macro", 0.0)
                    log_dict[f"eval/{ds_name}_f1_weighted"] = metrics.get("f1_weighted", 0.0)
            wandb.log(log_dict)

        accelerator.wait_for_everyone()
        save_dir = os.path.join(checkpoint_out_dir, f"checkpoint-{epoch}")
        if (epoch + 1) % args.save_ckpt_freq == 0:
            if master_process:
                print(f"Saving ZeRO-2/3 sharded checkpoint to {save_dir}...")
            accelerator.save_state(save_dir)
            if master_process:
                metadata = {
                    "model_args": {
                        "text_model_path": args.text_model_path,
                        "eeg_vocab_size": args.eeg_vocab_size,
                    },
                    "iter_num": iter_num,
                    "epoch": epoch,
                }
                torch.save(metadata, os.path.join(save_dir, "training_metadata.pt"))

                print(f"Exporting LoRA adapter to {checkpoint_out_dir}...")
                unwrapped_model = accelerator.unwrap_model(unified_model)
                save_path_lora = os.path.join(
                    checkpoint_out_dir, f"lora_instruction-{epoch}"
                )
                try:
                    unwrapped_model.llm_model.llm.save_pretrained(save_path_lora)
                    tokenizer.save_pretrained(save_path_lora)
                    print("LoRA export success.")
                except RuntimeError as e:
                    print(f"LoRA export failed: {e}")
    
    accelerator.wait_for_everyone()
    if master_process:
        print(f"\n{'='*70}")
        print(f"Training finished, starting final test...")
        print(f"{'='*70}")
    
    test_results = evaluate_model_normal(
        unified_model,
        data_loader_test,
        eeg_tokenizer,
        tokenizer,
        device,
        ctx,
        args,
        get_batch,
        split_name='test',
    )
    
    if master_process:
        print(f"\nFinal test results:")
        print(f"  Loss: {test_results['loss']:.4f}")
        print(f"  Instruction Loss: {test_results['instruction_loss']:.4f}")
        print(f"  Instruction Acc: {test_results['instruction_acc']:.2%}")
        print(f"  Balanced Acc: {test_results['balanced_acc']:.2%}")
        print(f"  Balanced accuracy per dataset:")
        for ds_name, acc in test_results.get('dataset_balanced_accs', {}).items():
            print(f"    {ds_name}: {acc:.2%}")
        if 'overall_binary_metrics' in test_results:
            print(f"  Overall binary: BalAcc {test_results['overall_binary_metrics'].get('balanced_acc', 0.0):.2%}, "
                f"AUROC {test_results['overall_binary_metrics'].get('auroc', 0.0):.2%}, "
                f"AUC-PR {test_results['overall_binary_metrics'].get('auc_pr', 0.0):.2%}")
        if 'overall_multi_metrics' in test_results:
            print(f"  Overall multi-class: BalAcc {test_results['overall_multi_metrics'].get('balanced_acc', 0.0):.2%}, "
                f"Kappa {test_results['overall_multi_metrics'].get('cohen_kappa', 0.0):.2%}, "
                f"F1-macro {test_results['overall_multi_metrics'].get('f1_macro', 0.0):.2%}, "
                f"F1-weighted {test_results['overall_multi_metrics'].get('f1_weighted', 0.0):.2%}")
        if 'dataset_metrics' in test_results:
            print(f"  Detailed metrics per dataset:")
            for ds_name, metrics in test_results['dataset_metrics'].items():
                if is_binary_classification(ds_name):
                    print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                        f"AUROC {metrics.get('auroc', 0.0):.2%}, "
                        f"AUC-PR {metrics.get('auc_pr', 0.0):.2%}")
                else:
                    print(f"    {ds_name}: BalAcc {metrics.get('balanced_acc', 0.0):.2%}, "
                        f"Kappa {metrics.get('cohen_kappa', 0.0):.2%}, "
                        f"F1-macro {metrics.get('f1_macro', 0.0):.2%}, "
                        f"F1-weighted {metrics.get('f1_weighted', 0.0):.2%}")
        print(f"  Number of test samples: {test_results['num_samples']}")
        print(f"{'='*70}\n")
        
        test_results_path = os.path.join(checkpoint_out_dir, "test_results.pt")
        torch.save(test_results, test_results_path)
        print(f"Test results saved to: {test_results_path}")
    
    if args.wandb_log and master_process:
        import wandb
        log_dict = {
            "test/loss": test_results['loss'],
            "test/instruction_loss": test_results['instruction_loss'],
            "test/instruction_accuracy": test_results['instruction_acc'],
            "test/balanced_accuracy": test_results['balanced_acc'],
            "test/binary_balanced_acc": test_results.get('overall_binary_metrics', {}).get('balanced_acc', 0.0),
            "test/binary_auroc": test_results.get('overall_binary_metrics', {}).get('auroc', 0.0),
            "test/binary_auc_pr": test_results.get('overall_binary_metrics', {}).get('auc_pr', 0.0),
            "test/multi_balanced_acc": test_results.get('overall_multi_metrics', {}).get('balanced_acc', 0.0),
            "test/multi_cohen_kappa": test_results.get('overall_multi_metrics', {}).get('cohen_kappa', 0.0),
            "test/multi_f1_macro": test_results.get('overall_multi_metrics', {}).get('f1_macro', 0.0),
            "test/multi_f1_weighted": test_results.get('overall_multi_metrics', {}).get('f1_weighted', 0.0),
        }
        for ds_name, acc in test_results['dataset_balanced_accs'].items():
            log_dict[f"test/balanced_acc_{ds_name}"] = acc
        for ds_name, metrics in test_results['dataset_metrics'].items():
            if is_binary_classification(ds_name):
                log_dict[f"test/{ds_name}_auroc"] = metrics.get("auroc", 0.0)
                log_dict[f"test/{ds_name}_auc_pr"] = metrics.get("auc_pr", 0.0)
            else:
                log_dict[f"test/{ds_name}_cohen_kappa"] = metrics.get("cohen_kappa", 0.0)
                log_dict[f"test/{ds_name}_f1_macro"] = metrics.get("f1_macro", 0.0)
                log_dict[f"test/{ds_name}_f1_weighted"] = metrics.get("f1_weighted", 0.0)
        wandb.log(log_dict)


if __name__ == "__main__":
    args = get_args()
    main(args)


