export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OMP_NUM_THREADS=1 torchrun --master_port 29505 --nnodes=1 --nproc_per_node=8 train_DSHA_tokenizer.py \
    --dataset_dir  \
    --out_dir  \
    --text_dataset_dir \
    --text_model_path  \
    --batch_size 64 \
    --gradient_accumulation_steps 2 
