export MODEL_NAME="/data/peiyu/zlc/codebase/modelscope/cache/AI-ModelScope/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout_mask.json"

export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --gradient_checkpointing \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=20000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="../ckpts/spatial_msr-loss-test_lr1e-5_bs4" \
  --enable_mar_loss