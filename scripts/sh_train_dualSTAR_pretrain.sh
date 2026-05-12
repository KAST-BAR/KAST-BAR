export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OMP_NUM_THREADS=1 accelerate launch --config_file accelerate_config.yaml --num_processes=8 train_pretrain_KAST_BAR.py \
    --dataset_dir  \
    --text_dataset_dir  \
    --text_model_path  \
    --out_dir  \
    --EEG_tokenizer_path \
    --eeg_batch_size 16 \
    --gradient_accumulation_steps 8 
