import os
import torch
import json
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor


def parse_args():
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout.json")
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--sam_encoder_version", type=str, default="vit_h")
    parser.add_argument("--sam_checkpoint_path", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/cache/sam_vit_h_4b8939.pth")
    parser.add_argument("--mask_save_root", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/data/sam_generated_masks")
    parser.add_argument("--json_save_path", type=str, default="/data/peiyu/zlc/codebase/layoutcontrol/data/spatial_contain_layout_mask.json")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    
    return args


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



def main():
    
    args = parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION =  args.sam_encoder_version
    SAM_CHECKPOINT_PATH = args.sam_checkpoint_path
    
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
    sam_predictor = SamPredictor(sam)

    with open(args.data_path, 'r') as fn:
        data = json.load(fn)

    for idx, d in enumerate(tqdm(data, desc="Generating masks")):
        image = Image.open(d['image_path']).convert('RGB')
        image_array = np.array(image)

        objects = d['object_list']

        bboxes = [obj[1] for obj in objects]
        new_bboxes = []
        for bbox in bboxes:
            # 将边界框的坐标从相对坐标转换为绝对坐标
            x1, y1, x2, y2 = [coord * args.canvas_size for coord in bbox]
            new_bboxes.append([x1, y1, x2, y2])

        maskes = segment(
            sam_predictor=sam_predictor,
            image=image_array,
            xyxy=np.array(new_bboxes)
        )

        assert len(maskes) == len(objects), "Number of masks and objects do not match"
        os.makedirs(args.mask_save_root, exist_ok=True)
        
        for i, mask in enumerate(maskes):
            # 0是黑，1是白；False是黑，True是白；取反后，mask部分是黑色，背景是白色，即mask部分是False(0)，背景是True(1);
            # 直接的输出是mask部分是True(1)，背景是False(0)；感觉直接存成输出的mask图像更好，读取的时候转成array即可
            mask_img = Image.fromarray((mask*255).astype(np.uint8), mode='L')
            file_name = "id-{}_iter-{}_image-{}_mask-{}.jpg".format(d['id'],d['iter'],d['image_id'],i)
            save_path = osp.join(args.mask_save_root,file_name)
            mask_img.save(osp.join(save_path))
            d['object_list'][i].append(save_path) 

        if args.test and idx == 10:
            break
    
    with open(args.json_save_path, 'w') as fn:
        json.dump(data, fn)
    print("final data is saved to {}".format(args.json_save_path))    

if __name__ == "__main__":
    main()