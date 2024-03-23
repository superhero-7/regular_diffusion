import torch
from diffusers import StableDiffusionGLIGENPipeline
import json
from tqdm import tqdm

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "/data/peiyu/zlc/codebase/layoutcontrol/cache/models--masterful--gligen-1-4-generation-text-box/snapshots/d2820dc1e9ba6ca082051ce79cfd3eb468ae2c83", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda:1")
pipe.set_progress_bar_config(disable=True)
pipe.safety_checker = None

data_path = "/data/peiyu/zlc/codebase/layoutcontrol/src/llm_output/spatial/gpt4.spatial.k-similar.k_8.px_64.json"
with open(data_path, 'r') as fn:
    data = json.load(fn)
    
save_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/gligen_generate_images"


for d in tqdm(data,desc="Generating:"):
    iter = d["iter"]
    query_id = d["query_id"]
    prompt = d['prompt']
    locations = [loc for _, loc in d['object_list'] if loc != None]
    phrases = ["a "+phrase for phrase, _ in d['object_list'] if phrase != None]


    images = pipe(
        prompt=[prompt]*4,
        gligen_phrases=phrases,
        gligen_boxes=locations,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=50,
    ).images
    
    for idx, image in enumerate(images):
        image.save(save_path+'/'+"id-{}_iter-{}_image-{}.jpg".format(query_id,iter,idx))

print("^_^!!! Finish")
