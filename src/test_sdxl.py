# %%
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "/data/peiyu/zlc/codebase/modelscope/cache/AI-ModelScope/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
# %%
image = pipeline_text2image(prompt="a green bench and a red car").images[0]
image
# %%
