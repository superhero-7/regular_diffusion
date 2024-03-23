# %%
import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
# pipe = StableDiffusionGLIGENPipeline.from_pretrained(
#     "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
# )

pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "/data/peiyu/zlc/codebase/layoutcontrol/cache/models--masterful--gligen-1-4-generation-text-box/snapshots/d2820dc1e9ba6ca082051ce79cfd3eb468ae2c83", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
# %%
prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage"
boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
phrases = ["a waterfall", "a modern high speed train running through the tunnel"]
images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
).images

images[0].save("./gligen-1-4-generation-text-box.jpg")

# %%
# test = {'query_id': 471, 'iter': 1, 'prompt': 'a fish on the right of a airplane', 'object_list': [['fish', [0.625, 0.3125, 0.9375, 0.546875]], ['airplane', [0.0, 0.3125, 0.46875, 0.546875]]]}
test = {'query_id': 0, 'iter': 0, 'prompt': 'a balloon on the top of a giraffe', 'object_list': [['balloon', [0.453125, 0.0, 0.640625, 0.1875]], ['giraffe', [0.375, 0.1875, 0.625, 1.0]]]}

prompt = test['prompt']
locations = [loc for _, loc in test['object_list'] if loc != None]
phrases = ["a "+phrase for phrase, _ in test['object_list'] if phrase != None]

images = pipe(
    prompt=[prompt]*4,
    gligen_phrases=phrases,
    gligen_boxes=locations,
    gligen_scheduled_sampling_beta=0.6,
    output_type="pil",
    num_inference_steps=50,
).images

# %%
images[0].show()
images[1].show()
images[2].show()
images[3].show()
# %%
for idx, image in enumerate(images):
    image.save("{}.jpg".format(idx))
# %%
