#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


OMP_NUM_THREADS=1 torchrun --master_port 29501 --nnodes=1 --nproc_per_node=8 analyze_eeg_directory_multi_ddp.py \
    --eeg_dir \
    --output_dir  \
    --model_path ./model/qwen/qwen2.5_7b\
    --model_type qwen \
    --max_new_tokens 2048 \
    --temperature 0.6 \
    --top_k_channels 6 \
    --attn_implementation flash_attention_2 

