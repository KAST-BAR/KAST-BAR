export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29510
OMP_NUM_THREADS=1 accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --num_processes=8 \
  train_instruction_STAR_BAR.py \
  --out_dir  \
  --dataset_dir \
  --text_model_path  \
  --text_dataset_dir  \
  --EEG_tokenizer_path   \
  --pretrained_lora_path  \
  --pretrained_ckpt_dir  \
  --eeg_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.1 \
  --train_starouter