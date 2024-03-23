import os
import os.path as osp
import torch
import argparse
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from PIL import Image

def parse_args():
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/data/peiyu/zlc/codebase/modelscope/cache/AI-ModelScope/stable-diffusion-v1-5")
    parser.add_argument("--weight_dtype", type=str, default="torch.float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_root", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/src/test_output")
    parser.add_argument("--unet_model_path", type=str, default=None)
    parser.add_argument("--task_name_for_save", type=str, default="spatial_ft_ckpt5000" ,help="format: [experiment_name]_[ckpt_step]")
    parser.add_argument("--test_prompt_file_dir", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/data/T2I-CompBench_dataset/spatial_val.txt")
    parser.add_argument("--test_num", type=int, default=None)
    
    args = parser.parse_args()
    
    return args
    

def main():
    args = parse_args()
    
    weight_dtype = torch.float16 if args.weight_dtype == "torch.float16" else torch.float32
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    if args.unet_model_path is not None:
        pipeline.unet.load_state_dict(load_file(args.unet_model_path))
    pipeline.to(args.device)
    
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with open(args.test_prompt_file_dir, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    
    if args.test_num is not None:
        prompts = prompts[:args.test_num]
    
    
    save_path = osp.join(args.save_root, args.task_name_for_save)
    os.makedirs(save_path, exist_ok=True)
    for index, prompt in enumerate(prompts):
        images = []
        for k in range(4):
            with torch.autocast("cuda"):
                image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]
            images.append(image)
        width, height = images[0].size
        total_width = 2*width
        total_height = 2*height
        new_image = Image.new('RGB', (total_width, total_height))
        
        new_image.paste(images[0], (0, 0))
        new_image.paste(images[1], (width, 0))
        new_image.paste(images[2], (0, height))
        new_image.paste(images[3], (width, height))
        
        file_name = f"{prompt}.png"
        new_image.save(osp.join(save_path,file_name))


if __name__ == "__main__":
    main()


