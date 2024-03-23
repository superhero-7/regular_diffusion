export MODEL_NAME="/data/peiyu/zlc/codebase/modelscope/cache/AI-ModelScope/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="/data/peiyu/zlc/codebase/layoutcontrol/data/spatial.json"

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=20000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="spatial_ft_lr1e-6_bs4" 