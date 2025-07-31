"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


#使得query可以从带有GT信息的噪声样本中学习，比随机初始化更好

def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]   #一个batch中标签数量
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts)   #最大标签数量
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num #分5组
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    #初始化
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device) #初始化为背景2 [2,20]
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)  #初始化为0 [2,20,4]
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)    #标注掩膜初始化0    [2,20]
    #赋初值标注
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']    #赋值gt
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1     #填充掩膜：前目标数个1，其余为0
    # each group has positive and negative queries.复制2 x num_group次
    input_query_class = input_query_class.tile([1, 2 * num_group])  #[2,200]
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1]) #[2,200,4]
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])  #[2,200]
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)  #[2,40,1]
    negative_gt_mask[:, max_gt_num:] = 1    #后一半变1:引入噪声
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])     #[2,200,1]
    positive_gt_mask = 1 - negative_gt_mask #所有样本：10对半分（各20）
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask   #[2,200] 将正负样本掩膜和填充掩膜结合.目标最多的样本：40个中1/0对半。其余样本：15个1，25个0
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1] #[175]  torch.nonzero(positive_gt_mask)[:, 1]返回的是所有非零值的下标，但这里只用了第二维
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])    #元组，每个元素是样本的非零下标（正样本）
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group) #200

    #生成类别噪声
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)    #随机掩码
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype) #0或1
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)   #随机选新（随机标签）/真实标签

    #生成bbox噪声
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)       #[2,200,4]
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale #缩放程度 [2,200,4]
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  #随机正负，代表偏移方向
        rand_part = torch.rand_like(input_query_bbox)   #随机小数
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)   #正负掩码加权
        known_bbox += (rand_sign * rand_part * diff)        #加噪
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)          #查询向量是unact的

    input_query_logits = class_embed(input_query_class) #[2,200,256]

    tgt_size = num_denoising + num_queries  #300+200=500
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction  正常query不能看到DN query
    attn_mask[num_denoising:, :num_denoising] = True        #左下角
    
    # reconstruct cannot see each other 每组DN query不可见
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

# input_query_logits是加噪的类别查询向量[2,200,256]
# input_query_bbox_unact是加噪的bbox查询向量[2,200,4]
