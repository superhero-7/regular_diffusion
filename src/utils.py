import torch
import json
import numpy as np
import torch.nn.functional as F

def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def write_json(fname, data):
    with open(fname, "w") as file:
        json.dump(data, file, indent=4, separators=(",",":"), sort_keys=True)


def bb_relative_position(boxA, boxB):
    xA_c = (boxA[0]+boxA[2])/2
    yA_c = (boxA[1]+boxA[3])/2
    xB_c = (boxB[0]+boxB[2])/2
    yB_c = (boxB[1]+boxB[3])/2
    dist = np.sqrt((xA_c - xB_c)**2 + (yA_c - yB_c)**2)
    cosAB = (xA_c-xB_c) / dist
    sinAB = (yB_c-yA_c) / dist
    return cosAB, sinAB
    

def eval_spatial_relation(bbox1, bbox2):
    theta = np.sqrt(2)/2
    relation = 'diagonal'

    if bbox1 == bbox2:
        return relation
    
    cosine, sine = bb_relative_position(bbox1, bbox2)

    if cosine > theta:
        relation = 'right'
    elif sine > theta:
        relation = 'top'
    elif cosine < -theta:
        relation = 'left'
    elif sine < -theta:
        relation = 'bottom'
    
    return relation


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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
    return out

def compute_ca_loss(attn_down, attn_up, masks, idxs, attn_res):
    # down的res是64-->32-->16
    # up的res是16-->32-->64    
    attn_down_res_16 = attn_down[2] # （2,8,256,77）
    attn_up_res_16 = attn_up[0] # (2,8,256,77)
    attn_sum = attn_down_res_16[0] + attn_down_res_16[1] + attn_up_res_16[0] + attn_up_res_16[1] # (8,256,77)
    loss = 0
    for mask,idx in zip(masks,idxs):
        # mask 16*16
        # idx是个列表
        loss_inner = 0
        # attn = torch.zeros_like(attn_sum[:, :, 0]).to(attn_sum.device).to(attn_sum.dtype)
        for i in idx:
            attn = attn_sum[:, :, i].reshape(-1, attn_res, attn_res)
            attn = attn.sum(0) / attn.shape[0] # 对所有头做平均
            loss_inner += F.mse_loss(attn, mask.to(attn_sum.device).to(attn_sum.dtype))
        loss_inner /= len(idx)
        loss += loss_inner
    loss /= len(idxs)
    return loss