import torch
from torch import nn
import torch.nn.functional as F
import inspect
import os
import numpy as np
import math
import time


from model.model_neural_transformer import NeuralTransformer
from model.token_based_quantizer import TokenBasedMSVectorQuantizer
from model.multiscale_channel_pooling import MultiScaleChannelPooling

from torch.autograd import Function
from transformers import AutoModel, AutoModelForCausalLM
from model.standard_1020_chorder import standard_1020, ch_nums, ms_ch_lst, ms_ch_dict



class CrossScaleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, main_stream_q, branch_stream_kv):
        B, N_q, E = main_stream_q.shape
        _, N_kv, _ = branch_stream_kv.shape
        
        Q = self.q_linear(main_stream_q)
        K = self.k_linear(branch_stream_kv)
        V = self.v_linear(branch_stream_kv)
        
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_q, E)
        
        output = self.out_linear(attn_output)
        
        return output


class CrossScaleFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_cross_attn = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        
        self.cross_scale_attn = CrossScaleMultiHeadAttention(embed_dim, num_heads, dropout)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, main_stream, branch_stream):
        norm_main = self.norm_cross_attn(main_stream)
        cross_attn_out = self.cross_scale_attn(norm_main, branch_stream)
        main_stream = main_stream + self.dropout(cross_attn_out)
        
        norm_main = self.norm_mlp(main_stream)
        mlp_out = self.mlp(norm_main)
        main_stream = main_stream + mlp_out
        
        return main_stream


class BTH_DualStream(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=128,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss=False,
                 num_layers=5,
                 share_cross_scale=False,
                 data_pooling=False,
                 tokens_per_window=91,
                 normalize_reconstruction=True,
                 **kwargs):
        super().__init__()
        print(kwargs)
        
        if decoder_config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config.in_chans} to {embed_dim}")
            decoder_config.in_chans = embed_dim

        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_freq = NeuralTransformer(decoder_config)
        self.decoder_raw = NeuralTransformer(decoder_config)
        
        self.quantize = TokenBasedMSVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, 
            kmeans_init=quantize_kmeans_init, decay=decay,
            tokens_per_window=tokens_per_window
        )

        self.decoder_out_dim = decoder_out_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.data_pooling = data_pooling
        self.tokens_per_window = tokens_per_window
        self.normalize_reconstruction = normalize_reconstruction
        self.standard_1020 = standard_1020
        self.share_cross_scale = share_cross_scale
        self.ms_ch_dict = ms_ch_dict

        
        self.main_stream_s1_layers = nn.ModuleList()
        self.main_stream_s5_layers = nn.ModuleList()
        
        shared_s1_fusion = None
        shared_s5_fusion = None
        if self.share_cross_scale:
            shared_s1_fusion = CrossScaleFusion(encoder_config.n_embd, encoder_config.n_head, dropout=0.1)
            shared_s5_fusion = CrossScaleFusion(encoder_config.n_embd, encoder_config.n_head, dropout=0.1)
        for i in range(num_layers):
            if self.share_cross_scale:
                self.main_stream_s1_layers.append(shared_s1_fusion)
            else:
                self.main_stream_s1_layers.append(
                    CrossScaleFusion(encoder_config.n_embd, encoder_config.n_head, dropout=0.1)
                )
            
            if self.share_cross_scale:
                self.main_stream_s5_layers.append(shared_s5_fusion)
            else:
                self.main_stream_s5_layers.append(
                    CrossScaleFusion(encoder_config.n_embd, encoder_config.n_head, dropout=0.1)
                )
            
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config.n_embd, encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(encoder_config.n_embd, embed_dim)
        )
        
        self.decode_task_layer_freq = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim // 2),
        )
        
        self.decode_task_layer_raw = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim),
        )
        
        
        self.token_to_chans = nn.Sequential(
            nn.Linear(self.tokens_per_window, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 90)
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_freq.apply(self._init_weights)
        self.decode_task_layer_raw.apply(self._init_weights)
        self.token_to_chans.apply(self._init_weights)
        self.main_stream_s1_layers.apply(self._init_weights)
        self.main_stream_s5_layers.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        
        self.timing_stats = {}
        self.enable_timing = False
    
    def set_timing(self, enable=True):
        self.enable_timing = enable
        if enable:
            self.timing_stats = {}
    
    def get_timing_stats(self):
        return self.timing_stats
    
    def print_timing_stats(self):
        if not self.timing_stats:
            print("No timing stats available. Please enable timing first.")
            return
        
        print("\n" + "="*60)
        print("Performance Timing Statistics")
        print("="*60)
        total_time = sum(self.timing_stats.values())
        for key, value in sorted(self.timing_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (value / total_time * 100) if total_time > 0 else 0
            print(f"{key:40s}: {value*1000:8.2f} ms ({percentage:5.1f}%)")
        print("="*60)
        print(f"{'Total':40s}: {total_time*1000:8.2f} ms (100.0%)")
        print("="*60 + "\n")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def ch_trans(self, tensor, input_chans, input_time, num_standard_ch):

        B, N, C = tensor.shape

        num_times = torch.max(input_time, dim=1).values + 1
        num_chs = torch.sum(input_time == 1, dim=1)
        W = int(num_times.max().item())

        output_tensor = torch.zeros(
            (B, W, num_standard_ch, C),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for i, (num_ch, num_time) in enumerate(zip(num_chs, num_times)):
            num_ch = int(num_ch.item())
            num_time = int(num_time.item())

            i_input_chans_id = input_chans[i, :num_ch]
            x = tensor[i, :num_ch * num_time].view(num_time, num_ch, C)

            output_tensor[i, :num_time, i_input_chans_id] = x
        return output_tensor

    def group_chans_by_mean(self, data, original_channels, final_groups):
        chan_to_idx = {ch: i for i, ch in enumerate(original_channels)}
        B, A, S, E = data.shape

        final_data = torch.empty(B, A, len(final_groups), E, dtype=data.dtype, device=data.device)

        for i, group in enumerate(final_groups):
            indices = torch.tensor(
                [chan_to_idx[ch] for ch in group],
                dtype=torch.long,
                device=data.device,
            )
            group_data = data.index_select(dim=2, index=indices)
            group_mean = group_data.mean(dim=2)
            final_data[:, :, i, :] = group_mean

        return final_data

    def encode_onescale_uptomultiscale(self, x, input_chans, input_time, mask=None):
        x_feature = self.encoder(x, input_chans, input_time, mask=mask, return_all_tokens=True)

        x_feature_standard = self.ch_trans(x_feature, input_chans, input_time, len(self.standard_1020)).contiguous()
        B, W, S, E = x_feature_standard.shape

        scale_features = {}
        SN = len(self.ms_ch_dict)
        for si in range(SN):
            scale_name = f's{si+1}'
            scale_features[scale_name] = self.group_chans_by_mean(
                x_feature_standard, self.ms_ch_dict[f'ch_lst_scale0'][0], self.ms_ch_dict[f'ch_lst_scale{si}']
            ).reshape(B, -1, E) if (si != SN-1) else x_feature_standard.reshape(B, -1, E)


        return scale_features, x_feature

    
    def dual_stream_forward(self, encoder_features=None, input_chans=None, input_time=None, mask=None, scale_features=None):
        for i in range(1, self.num_layers + 1):
            if self.enable_timing:
                torch.cuda.synchronize()
                t0 = time.time()
            
            if 1 + i <= 5:
                branch_key = f's{1+i}'
                scale_features['s1'] = self.main_stream_s1_layers[i-1](
                    scale_features['s1'], 
                    scale_features[branch_key]
                )
            else:
                scale_features['s1'] = self.main_stream_s1_layers[i-1](
                    scale_features['s1'], 
                    scale_features['s1']
                )
            
            if self.enable_timing:
                torch.cuda.synchronize()
                self.timing_stats[f'3_dual_stream_s1_layer_{i}'] = time.time() - t0
                torch.cuda.synchronize()
                t0 = time.time()
            
            if 5 - i >= 1:
                branch_key = f's{5-i}'
                scale_features['s5'] = self.main_stream_s5_layers[i-1](
                    scale_features['s5'], 
                    scale_features[branch_key]
                )
            else:
                scale_features['s5'] = self.main_stream_s5_layers[i-1](
                    scale_features['s5'], 
                    scale_features['s5']
                )
            
            if self.enable_timing:
                torch.cuda.synchronize()
                self.timing_stats[f'3_dual_stream_s5_layer_{i}'] = time.time() - t0
        
        return scale_features['s1'], scale_features['s5']

    def get_tokens(self, x, input_chans=None, input_time=None, mask=None, **kwargs):
        scale_features, _ = self.encode_onescale_uptomultiscale(x, input_chans, input_time, mask)
        
        s1_features, s5_features = self.dual_stream_forward(scale_features=scale_features)
        
        B, N1, E = s1_features.shape
        N5 = s5_features.shape[1]
        
        window_num = N1
        num_groups_s5 = N5 // window_num
        
        s1_4d = s1_features.view(B, window_num, 1, E)
        s5_4d = s5_features.view(B, window_num, num_groups_s5, E)
        
        fused_4d = torch.cat([s1_4d, s5_4d], dim=2)
        
        residual_4d_list = []
        
        if 's2' in scale_features:
            N2 = scale_features['s2'].shape[1]
            if N2 % window_num == 0:
                num_groups_s2 = N2 // window_num
                s2_4d = scale_features['s2'].view(B, window_num, num_groups_s2, E)
                residual_4d_list.append(s2_4d)
        
        if 's3' in scale_features:
            N3 = scale_features['s3'].shape[1]
            if N3 % window_num == 0:
                num_groups_s3 = N3 // window_num
                s3_4d = scale_features['s3'].view(B, window_num, num_groups_s3, E)
                residual_4d_list.append(s3_4d)
        
        if 's4' in scale_features:
            N4 = scale_features['s4'].shape[1]
            if N4 % window_num == 0:
                num_groups_s4 = N4 // window_num
                s4_4d = scale_features['s4'].view(B, window_num, num_groups_s4, E)
                residual_4d_list.append(s4_4d)
        
        if residual_4d_list:
            residual_concat = torch.cat(residual_4d_list, dim=2)
            residual_pooled = residual_concat.mean(dim=2, keepdim=True)
            fused_4d = fused_4d + 0.1 * residual_pooled
        
        fused_features = fused_4d.reshape(B,-1, E)
        
        with torch.amp.autocast('cuda', enabled=False):
            to_quantizer_features = self.encode_task_layer(fused_features.type_as(self.encode_task_layer[-1].weight))
        
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)
        
        inds_LmsN = embed_ind
        rest_LmsND = quantize
        
        return inds_LmsN, rest_LmsND

    def encode(self, x, scale_data, input_chans=None, input_time=None, mask=None):
        scale_features, x_features = self.encode_onescale_uptomultiscale(x, input_chans, input_time, mask)

        s1_features, s5_features = self.dual_stream_forward(scale_features=scale_features)
        
        if self.enable_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        
        B, N1, E = s1_features.shape
        N5 = s5_features.shape[1]
        
        window_num = N1
        num_groups_s5 = N5 // window_num
        
        s1_4d = s1_features.view(B, window_num, 1, E)
        s5_4d = s5_features.view(B, window_num, num_groups_s5, E)
        
        fused_4d = torch.cat([s1_4d, s5_4d], dim=2)
        residual_scales = []
        residual_4d_list = []
        
        if 's2' in scale_features:
            N2 = scale_features['s2'].shape[1]
            if N2 % window_num == 0:
                num_groups_s2 = N2 // window_num
                s2_4d = scale_features['s2'].view(B, window_num, num_groups_s2, E)
                residual_4d_list.append(s2_4d)
                residual_scales.append('s2')
        
        if 's3' in scale_features:
            N3 = scale_features['s3'].shape[1]
            if N3 % window_num == 0:
                num_groups_s3 = N3 // window_num
                s3_4d = scale_features['s3'].view(B, window_num, num_groups_s3, E)
                residual_4d_list.append(s3_4d)
                residual_scales.append('s3')
        
        if 's4' in scale_features:
            N4 = scale_features['s4'].shape[1]
            if N4 % window_num == 0:
                num_groups_s4 = N4 // window_num
                s4_4d = scale_features['s4'].view(B, window_num, num_groups_s4, E)
                residual_4d_list.append(s4_4d)
                residual_scales.append('s4')
        
        if residual_4d_list:
            residual_concat = torch.cat(residual_4d_list, dim=2)
            residual_pooled = residual_concat.mean(dim=2, keepdim=True)
            fused_4d = fused_4d + 0.1 * residual_pooled
        
        fused_features = fused_4d.reshape(B, -1, E)
        

        
        

        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['4_s3_residual'] = time.time() - t0
            torch.cuda.synchronize()
            t0 = time.time()
        
        with torch.amp.autocast('cuda', enabled=False):
            to_quantizer_features = self.encode_task_layer(fused_features.type_as(self.encode_task_layer[-1].weight))
        
        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['5_encode_task_layer'] = time.time() - t0
            torch.cuda.synchronize()
            t0 = time.time()
        
   
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['5_quantization'] = time.time() - t0
        
        return quantize, embed_ind, loss, scale_features, x_features
        
    def decode(self, quantize, input_chans=None, input_time=None, real_chans=None, mask=None, **kwargs):
        B,N,E = quantize.shape
        window_num = N // self.tokens_per_window

        quantize = quantize.view(B, window_num, self.tokens_per_window, E)
        quantize = quantize.permute(0, 1, 3, 2)
        quantize = self.token_to_chans(quantize)
        quantize = quantize.permute(0, 1, 3, 2)
        quantize = quantize.reshape(B, -1, E)
        
        num_standard_ch = len(self.standard_1020)
        seq_len = window_num * num_standard_ch
        
        if input_chans is None:
            channel_indices = torch.arange(num_standard_ch, device=quantize.device, dtype=torch.long)
            input_chans = channel_indices.repeat(window_num).unsqueeze(0).repeat(B, 1)
        
        if input_time is None:
            time_indices = torch.arange(window_num, device=quantize.device, dtype=torch.long)
            input_time = time_indices.repeat_interleave(num_standard_ch).unsqueeze(0).repeat(B, 1)

        if self.enable_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        raw_chans_num = 1024 // window_num
        raw_chans_index = real_chans[0, :raw_chans_num]
        
        decoder_features_freq = self.decoder_freq(quantize, input_chans, input_time, mask=None, return_all_tokens=True)
        B,_,Ed = decoder_features_freq.shape
        decoder_features_freq = decoder_features_freq.view(B, window_num, num_standard_ch, Ed)
        decoder_features_freq = decoder_features_freq[:, :, raw_chans_index, :]
        decoder_features_freq = decoder_features_freq.reshape(B, -1, Ed)
        decoder_features_freq = self._pad_decoder_features(decoder_features_freq)

        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['6_decoder_freq'] = time.time() - t0
            torch.cuda.synchronize()
            t0 = time.time()
        
        decoder_features_raw = self.decoder_raw(quantize, input_chans, input_time, mask=None, return_all_tokens=True)
        B,_,Ed = decoder_features_raw.shape
        decoder_features_raw = decoder_features_raw.view(B, window_num, num_standard_ch, Ed)
        decoder_features_raw = decoder_features_raw[:, :, raw_chans_index, :]
        decoder_features_raw = decoder_features_raw.reshape(B, -1, Ed)
        decoder_features_raw = self._pad_decoder_features(decoder_features_raw)
        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['6_decoder_raw'] = time.time() - t0
            torch.cuda.synchronize()
            t0 = time.time()
        
        rec_freq = self.decode_task_layer_freq(decoder_features_freq)
        rec_raw = self.decode_task_layer_raw(decoder_features_raw)

        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['6_decode_task_layers'] = time.time() - t0

        return rec_freq, rec_raw
    
    def _pad_decoder_features(self, decoder_features_freq):
        if decoder_features_freq.size(1) >= 1024:
            return decoder_features_freq
        B, _, E = decoder_features_freq.shape
        pad_len = 1024 - decoder_features_freq.size(1)
        pad_tensor = decoder_features_freq.new_zeros(B, pad_len, E)
        return torch.cat([decoder_features_freq, pad_tensor], dim=1)
    
    
    def get_codebook_indices(self, x, input_chans=None, input_time=None, input_mask=None, **kwargs):
        if input_mask is None:
            mask = None
        else:
            mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        inds_LmsN, _ = self.get_tokens(x, input_chans, input_time, mask, **kwargs)
        return inds_LmsN
    
    def get_codebook_msinds_and_msfeats(self, x, input_chans=None, input_time=None, input_mask=None, **kwargs):
        if input_mask is None:
            mask = None
        else:
            mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        inds_BN, rest_BND = self.get_tokens(x, input_chans, input_time, mask, **kwargs)
        B, _, emb = rest_BND.shape
        inds_BN = inds_BN.view(B,-1)
        return inds_BN, rest_BND
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def mean_pcc(self, tensor1, tensor2):
        assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape."
        tensor1_centered = tensor1 - tensor1.mean(dim=-1, keepdim=True)
        tensor2_centered = tensor2 - tensor2.mean(dim=-1, keepdim=True)
        numerator = (tensor1_centered * tensor2_centered).sum(dim=-1)
        denominator = torch.sqrt((tensor1_centered ** 2).sum(dim=-1) * (tensor2_centered ** 2).sum(dim=-1))
        pcc = numerator / (denominator + 1e-8)
        mean_pcc = pcc.mean()
        return mean_pcc

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        x = (x - mean) / std
        return x

    def ori_data_std_norm(self, x):
        mean = torch.mean(x, dim=(0, 1, 2), keepdim=True)
        std = torch.std(x, dim=(0, 1, 2), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, y_freq, y_raw, scale_data=None, input_chans=None, input_time=None, input_mask=None, **kwargs):
        if self.enable_timing:
            torch.cuda.synchronize()
            forward_start = time.time()
        if input_mask is not None and not isinstance(input_mask, str):
            mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        else:
            mask = None
        
        if self.enable_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        
        quantize, embed_ind, emb_loss, scale_features, aligned_fused_features = self.encode(x, scale_data, input_chans, input_time, mask)
        
        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['0_total_encode'] = time.time() - t0
        
        
        
        if self.enable_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        if scale_data is not None:
            xrec_freq, xrec_raw = self.decode(quantize, scale_data['s5'][1], scale_data['s5'][2], mask)
        else:
            xrec_freq, xrec_raw = self.decode(quantize, None, None, input_chans, mask)
        
        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['0_total_decode'] = time.time() - t0
            torch.cuda.synchronize()
            t0 = time.time()

        
        if input_mask is not None and not isinstance(input_mask, str):
            loss_freq_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_freq.size(-1))
            loss_raw_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_raw.size(-1))
            rec_freq_loss = self.calculate_rec_loss(xrec_freq * loss_freq_mask, y_freq * loss_freq_mask)
            rec_raw_loss = self.calculate_rec_loss(xrec_raw * loss_raw_mask, y_raw * loss_raw_mask)
        else:
            rec_freq_loss = self.calculate_rec_loss(xrec_freq, y_freq)
            rec_raw_loss = self.calculate_rec_loss(xrec_raw, y_raw)

        loss = emb_loss + rec_freq_loss + rec_raw_loss

        pcc = self.mean_pcc(y_raw, xrec_raw)

        if self.enable_timing:
            torch.cuda.synchronize()
            self.timing_stats['7_loss_calculation'] = time.time() - t0
            torch.cuda.synchronize()
            self.timing_stats['0_total_forward'] = time.time() - forward_start

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_freq_loss'] = rec_freq_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()
        log[f'{split}/pcc'] = pcc.detach().mean()

        return loss, aligned_fused_features, log
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num all parameter tensors: {len(param_dict)}, with {num_decay_params + num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
