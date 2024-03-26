# %%
import json

data_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout_mask.json"

with open(data_path, 'r') as fn:
    data = json.load(fn)

# %%
data[0]
# %%
from PIL import Image
import numpy as np
import torch

prompt = data[0]['prompt']
phrases = []
mask_images = []

for object in data[0]['object_list']:
    phrases.append(object[0])
    mask_images.append(Image.open(object[2]).convert('1'))
# %%
phrases
# %%
mask_images[0]
# %%
np.array(mask_images[0])
# %%
resized_mask_images = []
for mask_image in mask_images:
    resized_mask_images.append(mask_image.resize((16,16)))
# %%
numpy_mask_images = []

for resized_mask_image in  resized_mask_images:
    numpy_mask_images.append(np.array(resized_mask_image).astype(float))
# %%
attn = torch.rand(16,16).to("cuda")
torch_mask_images = []
for numpy_mask_image in numpy_mask_images:
    torch_mask_images.append(torch.tensor(numpy_mask_image).to(attn.device).to(attn.dtype))

# %%
attn.dtype
# %%
torch_mask_images[0]
# %%
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "/data/peiyu/zlc/codebase/modelscope/cache/AI-ModelScope/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer"
)
# %%
prompt
phrases
# %%
words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(prompt)][1:-1]

# %%
words_encode
# %%
# 这个函数的主要用途是在给定一段文本和一个单词时，找到这个单词在分词器处理后的文本中的位置。
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        # ptr指的是没分词的单词位置
        # cur_len用来统计当前token的累积长度（因此存在一个词被拆分的情况）
        # 循环中的i就是分词后的单词位置（注意去掉了'<|startoftext|>'和'<|endoftext|>'）
        # 所以out append的时候要+1
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)
# %%
id1 = get_word_inds(prompt, phrases[1], tokenizer)
# %%
id1
# %%
import torch.nn.functional as F

attn.shape, torch_mask_images[0].shape

# %%
loss = F.l1_loss(torch_mask_images[0], attn)
# %%
loss = F.mse_loss(torch_mask_images[0], attn)
# %%
loss
# %%