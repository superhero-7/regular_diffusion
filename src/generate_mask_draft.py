# %%
import torch
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/data/peiyu/zlc/codebase/layoutcontrol/cache/sam_vit_h_4b8939.pth"

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam)

# %%
import numpy as np

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    # final_mask = np.array(result_masks[0])
    # for mask in result_masks:
    #     final_mask += np.array(mask)
    # return final_mask
    return np.array(result_masks)

# %%
import json

data_path = "/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout.json"

with open(data_path, 'r') as fn:
    data = json.load(fn)
data[0]
# %%
from PIL import Image

canvas_size = 512

image = Image.open(data[0]['image_path']).convert('RGB')
image_array = np.array(image)

objects = data[0]['object_list']

bboxes = [obj[1] for obj in objects]
new_bboxes = []
for bbox in bboxes:
    # 将边界框的坐标从相对坐标转换为绝对坐标
    x1, y1, x2, y2 = [coord * canvas_size for coord in bbox]
    new_bboxes.append([x1, y1, x2, y2])
new_bboxes

# %%
import cv2


mask = segment(
    sam_predictor=sam_predictor,
    image=image_array,
    xyxy=np.array(new_bboxes)
)
# %%
image
# %%
# 0是黑，1是白；False是黑，True是白；取反后，mask部分是黑色，背景是白色，即mask部分是False(0)，背景是True(1);
# 直接的输出是mask部分是True(1)，背景是False(0)；
mask_img = Image.fromarray((~mask[0]*255).astype(np.uint8), mode='L')
mask_img
# %%
mask_img_1 = Image.fromarray((~mask[1]*255).astype(np.uint8), mode='L')
mask_img_1
# %%
np.set_printoptions(threshold=np.inf)

# Now you can print the entire array
print(mask[0])
# %%
