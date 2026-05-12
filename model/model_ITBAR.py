import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from liger_kernel.transformers.functional import liger_cross_entropy
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional
LIGER_AVAILABLE = False

@dataclass
class EEGCausalLMOutput(CausalLMOutputWithPast):
    text_loss: Optional[torch.FloatTensor] = None
    eeg_loss: Optional[torch.FloatTensor] = None
    eeg_acc: Optional[float] = None
    instruction_loss: Optional[torch.FloatTensor] = None
    instruction_acc: Optional[float] = None
    per_sample_instruction_loss: Optional[torch.FloatTensor] = None
    per_sample_instruction_acc: Optional[torch.FloatTensor] = None

class EEGBARLLM(nn.Module):
    def __init__(self, llm_model, tokenizer, eeg_start_id, eeg_vocab_size=8192, use_eeg_code_loss: bool = True):
        super().__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.eeg_start_id = eeg_start_id 
        self.use_eeg_code_loss = use_eeg_code_loss
        
        self.frozen_text_embed = self.llm.get_input_embeddings()
        self.frozen_text_embed.weight.requires_grad = False
        
        embed_dim = self.frozen_text_embed.weight.shape[1]
        self.trainable_eeg_embed = nn.Embedding(eeg_vocab_size, embed_dim)
        with torch.no_grad():
            self.trainable_eeg_embed.weight.data.normal_(mean=0.0, std=0.02)

        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
             qwen_causal_lm = self.llm.base_model.model
        else:
             qwen_causal_lm = self.llm
        
        if hasattr(qwen_causal_lm, "model"):
            self.transformer_body = qwen_causal_lm.model
        else:
            self.transformer_body = qwen_causal_lm

        self.transformer_body.gradient_checkpointing = True
        if hasattr(self.transformer_body, "config"):
            self.transformer_body.config.gradient_checkpointing = True
            self.transformer_body.config.use_cache = False
        
        print(f"DEBUG: Transformer Body Checkpointing Enabled: {self.transformer_body.gradient_checkpointing}")
        
        self._update_lm_head_ref()
    
    def _update_lm_head_ref(self):
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            self.lm_head = self.llm.base_model.model.lm_head
        else:
            self.lm_head = self.llm.lm_head

    def forward(
        self,
        text_input_ids: torch.Tensor,
        eeg_feats: torch.Tensor,
        eeg_codes: torch.Tensor,
        text_label_mask: Optional[torch.Tensor] = None,
    ):
        device = eeg_feats.device
        batch_size = eeg_feats.shape[0]
        
        seq_len_text = text_input_ids.shape[1]
        seq_len_feat = eeg_feats.shape[1]
        seq_len_code = eeg_codes.shape[1]
        
        with torch.no_grad():
            text_embeds = self.frozen_text_embed(text_input_ids)
        
        eeg_feat_embeds = eeg_feats.to(text_embeds.dtype)
        target_ids = eeg_codes + self.eeg_start_id
        target_embeds = self.trainable_eeg_embed(eeg_codes).to(text_embeds.dtype)
        
        inputs_embeds = torch.cat([text_embeds, eeg_feat_embeds, target_embeds], dim=1)
        
        labels_text = text_input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels_text[labels_text == self.tokenizer.pad_token_id] = -100

        if text_label_mask is not None:
            labels_text[text_label_mask == 0] = -100

        labels_feat = torch.full((batch_size, seq_len_feat), -100, dtype=torch.long, device=device)
        if self.use_eeg_code_loss:
            labels_code = target_ids.clone()
        else:
            labels_code = torch.full((batch_size, seq_len_code), -100, dtype=torch.long, device=device)
        labels = torch.cat([labels_text, labels_feat, labels_code], dim=1)
        
        text_mask = (text_input_ids != self.tokenizer.pad_token_id).long()
        feat_mask = torch.ones((batch_size, seq_len_feat), device=device, dtype=torch.long)
        code_mask = torch.ones((batch_size, seq_len_code), device=device, dtype=torch.long)
        attention_mask = torch.cat([text_mask, feat_mask, code_mask], dim=1)
        
        if self.training and not self.transformer_body.gradient_checkpointing:
            self.transformer_body.gradient_checkpointing = True

        outputs = self.transformer_body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False
        )
        hidden_states = outputs[0]

        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        B, Tm1, H = shift_hidden.shape
        hidden_flat = shift_hidden.reshape(-1, H)
        labels_flat = shift_labels.reshape(-1)
        
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            lm_head = self.llm.base_model.model.lm_head
        else:
            lm_head = self.llm.lm_head
        
        chunk_size = 512
        loss_chunks = []
        n_correct_eeg = 0
        n_total_eeg = 0
        
        if not LIGER_AVAILABLE:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        for i in range(0, hidden_flat.size(0), chunk_size):
            h_chunk = hidden_flat[i : i + chunk_size]
            target_chunk = labels_flat[i : i + chunk_size]
            
            logits_chunk = lm_head(h_chunk)
            
            if LIGER_AVAILABLE:
                loss_chunk = liger_cross_entropy(logits_chunk, target_chunk, ignore_index=-100, reduction='none')
            else:
                loss_chunk = loss_fct(logits_chunk, target_chunk)
            loss_chunks.append(loss_chunk)
        
            with torch.no_grad():
                eeg_mask = (target_chunk >= self.eeg_start_id) & (target_chunk != -100)
                
                if eeg_mask.any():
                    preds = torch.argmax(logits_chunk, dim=-1)
                    
                    correct = (preds == target_chunk) & eeg_mask
                    n_correct_eeg += correct.sum().item()
                    n_total_eeg += eeg_mask.sum().item()

        
        per_token_loss = torch.cat(loss_chunks, dim=0).view(B, Tm1)

        valid_mask = shift_labels.ne(-100)
        with torch.no_grad():
            is_eeg = valid_mask & shift_labels.ge(self.eeg_start_id)
            is_text = valid_mask & shift_labels.lt(self.eeg_start_id)

        def masked_mean(x, mask):
            denom = mask.sum().clamp(min=1)
            return (x * mask).sum() / denom

        total_loss = masked_mean(per_token_loss, valid_mask)
        text_loss = masked_mean(per_token_loss, is_text)
        eeg_loss = masked_mean(per_token_loss, is_eeg)
        eeg_acc = n_correct_eeg / n_total_eeg if n_total_eeg > 0 else 0.0

        return EEGCausalLMOutput(
            loss=total_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            text_loss=text_loss,
            eeg_loss=eeg_loss,
            eeg_acc=eeg_acc
        )


class EEGBARLLM_instruction(nn.Module):
    def __init__(self, llm_model, tokenizer, eeg_start_id, eeg_vocab_size=8192, use_eeg_code_loss: bool = True):
        super().__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.eeg_start_id = eeg_start_id 
        self.use_eeg_code_loss = use_eeg_code_loss
        
        self.frozen_text_embed = self.llm.get_input_embeddings()
        self.frozen_text_embed.weight.requires_grad = False
        
        embed_dim = self.frozen_text_embed.weight.shape[1]
        self.trainable_eeg_embed = nn.Embedding(eeg_vocab_size, embed_dim)
        with torch.no_grad():
            self.trainable_eeg_embed.weight.data.normal_(mean=0.0, std=0.02)

        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
             qwen_causal_lm = self.llm.base_model.model
        else:
             qwen_causal_lm = self.llm
        
        if hasattr(qwen_causal_lm, "model"):
            self.transformer_body = qwen_causal_lm.model
        else:
            self.transformer_body = qwen_causal_lm

        self.transformer_body.gradient_checkpointing = True
        if hasattr(self.transformer_body, "config"):
            self.transformer_body.config.gradient_checkpointing = True
            self.transformer_body.config.use_cache = False
        
        print(f"DEBUG: Transformer Body Checkpointing Enabled: {self.transformer_body.gradient_checkpointing}")
        
        self._update_lm_head_ref()
    
    def _update_lm_head_ref(self):
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            self.lm_head = self.llm.base_model.model.lm_head
        else:
            self.lm_head = self.llm.lm_head

    def forward(
        self,
        text_input_ids: torch.Tensor,
        eeg_feats: torch.Tensor,
        eeg_codes: torch.Tensor,
        text_label_mask: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.Tensor] = None,
        suffix_label_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        use_eeg_code_segment: bool = True,
    ):
        device = eeg_feats.device
        batch_size = eeg_feats.shape[0]
        
        seq_len_text = text_input_ids.shape[1]
        seq_len_feat = eeg_feats.shape[1]
        seq_len_code = eeg_codes.shape[1] if use_eeg_code_segment else 0
        
        with torch.no_grad():
            text_embeds = self.frozen_text_embed(text_input_ids)
        
        eeg_feat_embeds = eeg_feats.to(text_embeds.dtype)
        if use_eeg_code_segment:
            target_ids = eeg_codes + self.eeg_start_id
            target_embeds = self.trainable_eeg_embed(eeg_codes).to(text_embeds.dtype)
            inputs_embeds = torch.cat([text_embeds, eeg_feat_embeds, target_embeds], dim=1)
        else:
            target_ids = None
            target_embeds = None
            inputs_embeds = torch.cat([text_embeds, eeg_feat_embeds], dim=1)
        if suffix_input_ids is not None:
            with torch.no_grad():
                suffix_embeds = self.frozen_text_embed(suffix_input_ids).to(text_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, suffix_embeds], dim=1)
        
        labels_text = text_input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels_text[labels_text == self.tokenizer.pad_token_id] = -100

        if text_label_mask is not None:
            labels_text[text_label_mask == 0] = -100
        else:
            if suffix_input_ids is not None: 
                labels_text[:] = -100

        labels_feat = torch.full((batch_size, seq_len_feat), -100, dtype=torch.long, device=device)
        if use_eeg_code_segment:
            if self.use_eeg_code_loss:
                labels_code = target_ids.clone()
            else:
                labels_code = torch.full((batch_size, seq_len_code), -100, dtype=torch.long, device=device)
            labels = torch.cat([labels_text, labels_feat, labels_code], dim=1)
        else:
            labels_code = None
            labels = torch.cat([labels_text, labels_feat], dim=1)
        if suffix_input_ids is not None:
            labels_suffix = suffix_input_ids.clone()
            if self.tokenizer.pad_token_id is not None:
                labels_suffix[labels_suffix == self.tokenizer.pad_token_id] = -100
            if suffix_label_mask is not None:
                labels_suffix[suffix_label_mask == 0] = -100
            labels = torch.cat([labels, labels_suffix], dim=1)

        text_mask = (text_input_ids != self.tokenizer.pad_token_id).long()
        feat_mask = torch.ones((batch_size, seq_len_feat), device=device, dtype=torch.long)
        if use_eeg_code_segment:
            code_mask = torch.ones((batch_size, seq_len_code), device=device, dtype=torch.long)
            attention_mask = torch.cat([text_mask, feat_mask, code_mask], dim=1)
        else:
            attention_mask = torch.cat([text_mask, feat_mask], dim=1)
        if suffix_input_ids is not None:
            suffix_mask = (suffix_input_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        
        if self.training and not self.transformer_body.gradient_checkpointing:
            self.transformer_body.gradient_checkpointing = True

        outputs = self.transformer_body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False
        )
        hidden_states = outputs[0]

        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        B, Tm1, H = shift_hidden.shape
        hidden_flat = shift_hidden.reshape(-1, H)
        labels_flat = shift_labels.reshape(-1)
        
        if hasattr(self.llm, "base_model") and hasattr(self.llm.base_model, "model"):
            lm_head = self.llm.base_model.model.lm_head
        else:
            lm_head = self.llm.lm_head
        
        chunk_size = 512
        loss_chunks = []
        n_correct_eeg = 0
        n_total_eeg = 0
        
        if not LIGER_AVAILABLE:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        for i in range(0, hidden_flat.size(0), chunk_size):
            h_chunk = hidden_flat[i : i + chunk_size]
            target_chunk = labels_flat[i : i + chunk_size]
            
            logits_chunk = lm_head(h_chunk)
            
            if LIGER_AVAILABLE:
                loss_chunk = liger_cross_entropy(logits_chunk, target_chunk, ignore_index=-100, reduction='none')
            else:
                loss_chunk = loss_fct(logits_chunk, target_chunk)
            loss_chunks.append(loss_chunk)
        
            with torch.no_grad():
                eeg_mask = (target_chunk >= self.eeg_start_id) & (target_chunk != -100)
                
                if eeg_mask.any():
                    preds = torch.argmax(logits_chunk, dim=-1)
                    
                    correct = (preds == target_chunk) & eeg_mask
                    n_correct_eeg += correct.sum().item()
                    n_total_eeg += eeg_mask.sum().item()

        
        per_token_loss = torch.cat(loss_chunks, dim=0).view(B, Tm1)

        valid_mask = shift_labels.ne(-100)
        with torch.no_grad():
            is_eeg = valid_mask & shift_labels.ge(self.eeg_start_id)
            is_text = valid_mask & shift_labels.lt(self.eeg_start_id)
            
            is_instruction = torch.zeros_like(valid_mask, dtype=torch.bool)
            
            if suffix_input_ids is not None and suffix_label_mask is not None:
                suffix_start_in_shift = seq_len_text + seq_len_feat + seq_len_code - 1
                
                seq_len_suffix = suffix_input_ids.shape[1]
                
                if 0 <= suffix_start_in_shift < Tm1:
                    remaining_len = Tm1 - suffix_start_in_shift
                    actual_suffix_len = min(seq_len_suffix, remaining_len)
                    
                    if actual_suffix_len > 0:
                        suffix_mask_segment = suffix_label_mask[:, :actual_suffix_len].to(device)
                        
                        target_labels_segment = shift_labels[:, suffix_start_in_shift : suffix_start_in_shift + actual_suffix_len]
                        
                        is_instruction[:, suffix_start_in_shift : suffix_start_in_shift + actual_suffix_len] = (
                            suffix_mask_segment == 1
                        ) & (target_labels_segment != -100)
                        
        def masked_mean(x, mask):
            denom = mask.sum().clamp(min=1)
            return (x * mask).sum() / denom

        total_loss = masked_mean(per_token_loss, valid_mask)
        text_loss = masked_mean(per_token_loss, is_text)
        eeg_loss = masked_mean(per_token_loss, is_eeg)
        eeg_acc = n_correct_eeg / n_total_eeg if n_total_eeg > 0 else 0.0
        
        if is_instruction.any():
            per_sample_instruction_loss = []
            for b in range(B):
                sample_instruction_mask = is_instruction[b]
                if sample_instruction_mask.any():
                    sample_loss = masked_mean(per_token_loss[b], sample_instruction_mask)
                    per_sample_instruction_loss.append(sample_loss)
                else:
                    per_sample_instruction_loss.append(torch.tensor(0.0, device=device, dtype=per_token_loss.dtype))
            
            per_sample_instruction_loss = torch.stack(per_sample_instruction_loss)
            
            if sample_weights is not None:
                if sample_weights.device != device:
                    sample_weights = sample_weights.to(device)
                if sample_weights.dtype != per_sample_instruction_loss.dtype:
                    sample_weights = sample_weights.to(per_sample_instruction_loss.dtype)
                weighted_loss_sum = (per_sample_instruction_loss * sample_weights).sum()
                weight_sum = sample_weights.sum()
                instruction_loss = weighted_loss_sum / weight_sum.clamp(min=1e-8)
            else:
                instruction_loss = per_sample_instruction_loss.mean()
        else:
            instruction_loss = torch.tensor(0.0, device=device, dtype=per_token_loss.dtype)
            per_sample_instruction_loss = torch.zeros(B, device=device, dtype=per_token_loss.dtype)
        
        n_correct_instruction = 0
        n_total_instruction = 0
        per_sample_instruction_acc = []
        
        if is_instruction.any():
            with torch.no_grad():
                instruction_mask_flat = is_instruction.view(-1)
                if instruction_mask_flat.any():
                    instruction_hidden = shift_hidden[is_instruction]
                    instruction_labels = shift_labels[is_instruction]
                    
                    instruction_preds = []
                    for i in range(0, instruction_hidden.size(0), chunk_size):
                        h_chunk = instruction_hidden[i : i + chunk_size]
                        logits_chunk = lm_head(h_chunk)
                        preds_chunk = torch.argmax(logits_chunk, dim=-1)
                        instruction_preds.append(preds_chunk)
                    
                    instruction_preds = torch.cat(instruction_preds, dim=0)
                    correct_instruction = (instruction_preds == instruction_labels)
                    n_correct_instruction = correct_instruction.sum().item()
                    n_total_instruction = instruction_labels.numel()
                    
                    nonzero_indices = torch.nonzero(is_instruction, as_tuple=False)
                    sample_indices = nonzero_indices[:, 0]
                    
                    for b in range(B):
                        sample_mask = (sample_indices == b)
                        if sample_mask.any():
                            sample_correct = correct_instruction[sample_mask].sum().item()
                            sample_total = sample_mask.sum().item()
                            sample_acc = sample_correct / sample_total if sample_total > 0 else 0.0
                            per_sample_instruction_acc.append(torch.tensor(sample_acc, device=device, dtype=torch.float32))
                        else:
                            per_sample_instruction_acc.append(torch.tensor(0.0, device=device, dtype=torch.float32))
                else:
                    per_sample_instruction_acc = [torch.tensor(0.0, device=device, dtype=torch.float32) for _ in range(B)]
        else:
            per_sample_instruction_acc = [torch.tensor(0.0, device=device, dtype=torch.float32) for _ in range(B)]
        
        per_sample_instruction_acc = torch.stack(per_sample_instruction_acc)
        instruction_acc = n_correct_instruction / n_total_instruction if n_total_instruction > 0 else 0.0

        if sample_weights is not None and is_instruction.any():
            weighted_total_loss = instruction_loss
        else:
            weighted_total_loss = total_loss

        return EEGCausalLMOutput(
            loss=weighted_total_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            text_loss=text_loss,
            eeg_loss=eeg_loss,
            eeg_acc=eeg_acc,
            instruction_loss=instruction_loss,
            instruction_acc=instruction_acc,
            per_sample_instruction_loss=per_sample_instruction_loss,
            per_sample_instruction_acc=per_sample_instruction_acc
        )

    