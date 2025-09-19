import torch
import torch.nn.functional as F
import torch.nn as nn


    
def grouped_self_attn(tgt, temp_cls_head, group_attn, num_queries, score_threshold=0.5, min_keep=10):
    """
    分组注意力（带掩码 + 顺序还原）

    Args:
        tgt: [B, Q, C] decoder 输入
        temp_cls_head: 临时分类头 nn.Linear(C, 1)
        group_attn: 组内自注意力模块 (支持 attn_mask)
        num_queries: 每个 batch 的 query 数
        score_threshold: 前景阈值
        min_keep: 至少保留的前景 query 数

    Return:
        tgt_new: [B, Q, C] 更新后的 decoder 输入
    """
    B, Q, C = tgt.shape

    # 1. 拿出后 num_queries 个 → decoder 专用 queries
    tgt0 = tgt[:, -num_queries:, :]   # [B, num_queries, C]
    
    # 2. 计算前景分数
    scores = torch.sigmoid(temp_cls_head(tgt0)).squeeze(-1)  # [B, num_queries]

    # 3. 前景 mask
    fg_mask = scores > score_threshold                      # [B, num_queries]
    
    # 保证至少 min_keep
    topk_scores, topk_idx = torch.topk(scores, min_keep, dim=1)  # [B, min_keep]
    fg_mask.scatter_(1, topk_idx, True)
    bg_mask = ~fg_mask

    # 4. tgt分组（保持索引）
    tgt_fg = [tgt0[i][fg_mask[i]] for i in range(B)]  # list of [k, C]
    tgt_bg = [tgt0[i][bg_mask[i]] for i in range(B)]  # list of [num_queries-k, C]
    idx_fg = [torch.nonzero(fg_mask[i], as_tuple=False).squeeze(-1) for i in range(B)]  #对应的下标
    idx_bg = [torch.nonzero(bg_mask[i], as_tuple=False).squeeze(-1) for i in range(B)]

    # 5. 所有batch的tgt数量补齐 ,补零向量
    max_fg = max(x.size(0) for x in tgt_fg)
    max_bg = max(x.size(0) for x in tgt_bg)

    padded_tgt_fg = torch.stack([F.pad(x, (0, 0, 0, max_fg - x.size(0))) for x in tgt_fg], dim=0)  # [B, max_fg, C]
    padded_tgt_bg = torch.stack([F.pad(x, (0, 0, 0, max_bg - x.size(0))) for x in tgt_bg], dim=0)  # [B, max_bg, C]

    # 6. 生成 attn_mask：True 表示不能交互
    # 下标[0,最大长度)  >=  真实长度
    attn_mask_fg = torch.arange(max_fg, device=tgt.device)[None, :].expand(B, -1) >= torch.tensor(  
        [x.size(0) for x in tgt_fg], device=tgt.device)[:, None]
    attn_mask_bg = torch.arange(max_bg, device=tgt.device)[None, :].expand(B, -1) >= torch.tensor(
        [x.size(0) for x in tgt_bg], device=tgt.device)[:, None]

    # 7. 组内自注意力（带掩码）
    assert not torch.isnan(padded_tgt_fg).any(), "NaN detected in tensor"
    assert not torch.isnan(padded_tgt_bg).any(), "NaN detected in tensor"
    
    padded_tgt_fg = group_attn(padded_tgt_fg, attn_mask=attn_mask_fg)  # [B, max_fg, C]
    padded_tgt_bg = group_attn(padded_tgt_bg, attn_mask=attn_mask_bg)  # [B, max_bg, C]
    
    assert not torch.isnan(padded_tgt_fg).any(), "NaN detected in tensor"
    assert not torch.isnan(padded_tgt_bg).any(), "NaN detected in tensor"
    # 8. 按原始顺序还原
    tgt0_new = torch.zeros_like(tgt0)  # [B, num_queries, C]
    for i in range(B):
        tgt0_new[i, idx_fg[i]] = padded_tgt_fg[i, :idx_fg[i].size(0)]       #前景对应的索引处,填上pad的那些向量
        tgt0_new[i, idx_bg[i]] = padded_tgt_bg[i, :idx_bg[i].size(0)]

    # 9. 拼回 decoder 输入
    tgt_new = torch.cat([tgt[:, :-num_queries, :], tgt0_new], dim=1)  # [B, Q, C]

    return tgt_new




