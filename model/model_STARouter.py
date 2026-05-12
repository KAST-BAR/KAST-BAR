from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM


class STARouter(nn.Module):
    def __init__(self,
                 d_model=768,
                 num_queries=16,
                 eeg_dim=128,
                 num_eeg_channels=91,
                 nhead=8,
                 dropout=0.1,
                 text_embedding_config=None,
                 text_embedding_module=None,
                 freeze_text=True):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_eeg_channels = num_eeg_channels
        self.freeze_text = freeze_text
        self.eeg_input_proj = nn.Sequential(
            nn.Linear(eeg_dim, eeg_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eeg_dim * 8, d_model),
            nn.Dropout(dropout)
        )
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_queries, d_model))

        self.text_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.eeg_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        if text_embedding_module is not None:
            self.wte = text_embedding_module
        else:
            self.wte = self._build_text_embedding(text_embedding_config)
        
        if self.freeze_text and text_embedding_module is None:
             for param in self.wte.parameters():
                 param.requires_grad = False
        self._init_weights()

    def _build_text_embedding(self, text_embedding_config):
        default_cfg = {
            'model_path': 'gpt2',
            'embedding_key': 'transformer.wte.weight',
            'embedding_key_2': 'model.embed_tokens.weight',
            'model_type': 'causal_lm',
            'freeze': True
        }
        if text_embedding_config is not None:
            default_cfg.update(text_embedding_config)

        model_path = default_cfg['model_path']
        embedding_key = default_cfg['embedding_key']
        embedding_key_2 = default_cfg['embedding_key_2']
        model_type = default_cfg['model_type']
        freeze = default_cfg['freeze']

        loader_cls = AutoModelForCausalLM if model_type == 'causal_lm' else AutoModel
        model_hf = loader_cls.from_pretrained(
            model_path, local_files_only=True)
        state_dict = model_hf.state_dict()
        if embedding_key in state_dict:
            embedding_weight = state_dict[embedding_key]
        elif embedding_key_2 in state_dict:
            embedding_weight = state_dict[embedding_key_2]

        vocab_size, embed_dim = embedding_weight.shape
        wte = nn.Embedding(vocab_size, embed_dim, _freeze=freeze)
        wte.weight.data.copy_(embedding_weight)
        return wte

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and p.requires_grad and not name.startswith("wte."):
                nn.init.xavier_uniform_(p)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x_eeg, x_text, text_mask=None):
        x_eeg = self.eeg_input_proj(x_eeg)
        B, total_eeg_len, D = x_eeg.shape

        x_text = self.wte(x_text).detach()

        _, L_text, D_text = x_text.shape
        window_num = total_eeg_len // self.num_eeg_channels

        eeg_reshaped = x_eeg.view(B * window_num, self.num_eeg_channels, D)

        text_expanded = x_text.repeat_interleave(window_num, dim=0)

        key_padding_mask = None
        if text_mask is not None:
            mask_expanded = text_mask.repeat_interleave(window_num, dim=0)
            key_padding_mask = (mask_expanded == 0)

        latents = self.latent_queries.expand(B * window_num, -1, -1)

        q_aware, _ = self.text_attn(
            query=latents,
            key=text_expanded,
            value=text_expanded,
            key_padding_mask=key_padding_mask
        )
        q_aware = self.norm1(latents + q_aware)

        routed_feats, _ = self.eeg_attn(
            query=q_aware,
            key=eeg_reshaped,
            value=eeg_reshaped
        )
        routed_feats = self.norm2(q_aware + routed_feats)

        ffn_out = self.ffn(routed_feats)
        output_flat = self.norm3(routed_feats + ffn_out)

        output = output_flat.view(B, window_num * self.num_queries, D)

        return output
        
    def get_orthogonality_loss(self):
        Q = self.latent_queries.squeeze(0) 
        
        Q_norm = F.normalize(Q, p=2, dim=1)
        
        gram_matrix = torch.matmul(Q_norm, Q_norm.t())
        
        identity = torch.eye(self.num_queries, device=Q.device)
        
        orth_loss = torch.norm(gram_matrix - identity, p='fro')
        
        return orth_loss