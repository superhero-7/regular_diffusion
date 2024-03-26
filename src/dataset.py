import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
from PIL import Image
from utils import get_word_inds
import numpy as np

class LayoutDataset(Dataset):
    
    def __init__(self, data_path, resolution=512, center_crop=False, random_flip=False, tokenizer=None) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.tf = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = tokenizer
        
        # data format: [{'id': '94',
        # 'iter': '1',
        # 'image_id': '2',
        # 'image_path': '/data/peiyu/zlc/codebase/layoutcontrol/data/gligen_generate_images/id-94_iter-1_image-2.jpg',
        # 'prompt': 'a woman on the left of a television'},...]
        with open(self.data_path, 'r') as fn:
            self.data = json.load(fn)
            
        print("Sucessfully load data!!!")
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        example = {}
        try:
            caption = self.data[index]['prompt']
            img = Image.open(self.data[index]['image_path']).convert("RGB")
            example["pixel_values"] = self.tf(img)
            example["input_ids"] = self.tokenizer(
                [caption], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            
            return example
        except Exception as e:
            print('Bad idx of %d skipped bacause of %s' % (index, e))

class MaskDataset(Dataset):
    
    def __init__(self, data_path, resolution=512, attn_res=16, center_crop=False, random_flip=False, tokenizer=None) -> None:
        super().__init__()
        
        self.data_path = data_path
        self.tf = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = tokenizer
        self.attn_res = attn_res
        
        # {'id': '94',
        # 'iter': '1',
        # 'image_id': '2',
        # 'image_path': '/data/peiyu/zlc/codebase/layoutcontrol/data/gligen_generate_images/id-94_iter-1_image-2.jpg',
        # 'prompt': 'a woman on the left of a television',
        # 'object_list': [['woman',
        # [0.0, 0.109375, 0.4375, 0.890625],
        # '/data/peiyu/zlc/codebase/layoutcontrol/data/sam_generated_masks/id-94_iter-1_image-2_mask-0.jpg'],
        # ['television',
        # [0.546875, 0.234375, 0.921875, 0.75],
        # '/data/peiyu/zlc/codebase/layoutcontrol/data/sam_generated_masks/id-94_iter-1_image-2_mask-1.jpg']]}
        with open(self.data_path, 'r') as fn:
            self.data = json.load(fn)
            
        print("Sucessfully load data!!!")
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        example = {}
        try:
            caption = self.data[index]['prompt']
            img = Image.open(self.data[index]['image_path']).convert("RGB")
            example["pixel_values"] = self.tf(img).to(memory_format=torch.contiguous_format).float()
            example["input_ids"] = self.tokenizer(
                [caption], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            
            phrases = []
            torch_mask_images = []
            for object in self.data[index]['object_list']:
                phrases.append(object[0])
                mask_image = Image.open(object[2]).convert('1')
                resized_mask_image = mask_image.resize((self.attn_res, self.attn_res))
                numpy_mask_image = np.array(resized_mask_image).astype(float)
                torch_mask_images.append(torch.tensor(numpy_mask_image))
            
            attn_map_ids = []
            for phrase in phrases:
                phrase_inds = get_word_inds(caption, phrase, self.tokenizer)
                attn_map_ids.append(phrase_inds)
            example["attn_map_ids"] = attn_map_ids
            example["torch_mask_images"] = torch_mask_images
            
            return example
        except Exception as e:
            print('Bad idx of %d skipped bacause of %s' % (index, e))

            
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples], dim=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids}

# 暂时来为batch size=1过渡一下就好，后面再修成成支持pad
def tmp_collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples], dim=0)
    attn_map_ids = examples[0]["attn_map_ids"]
    torch_mask_images = examples[0]["torch_mask_images"]
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attn_map_ids": attn_map_ids, "torch_mask_images": torch_mask_images}
