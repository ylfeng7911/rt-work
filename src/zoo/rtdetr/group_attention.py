import torch
import torch.nn.functional as F
import torch.nn as nn


class GNNMixer(nn.Module):
    """
    轻量图神经网络 (GAT风格) 用于 TGT 分组交互
    输入: [B, L, C]
    输出: [B, L, C]
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # x: [B, L, C]
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, C//H]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, H, L]

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].expand(-1, self.heads, L, -1)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)  # [B, L, C]

        out = self.proj(out)
        out = self.dropout(out)
        return self.norm(x + out)   # 残差+归一化



# 定义一个简单的 1D 卷积交互模块
class Conv1dGroupMixer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        """
        dim: 特征维度 (C)
        kernel_size: 卷积核大小（建议 3 或 5）
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,   # 如果想分通道做卷积，可以设为 dim
            bias=False
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, L, C]
        key_padding_mask: [B, L]，True 表示该位置是 padding
        """
        B, L, C = x.shape
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, C]
        out = self.norm(out)

        if key_padding_mask is not None:
            # 把 padding 的位置置零，避免污染
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return out
    


class GatedConv1d(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim*2, kernel_size, padding=kernel_size//2)

    def forward(self, x):  # x: [B, C, L]
        out = self.conv(x)               # [B, 2C, L]
        out, gate = out.chunk(2, dim=1)  # 划分为信号和门
        return out * torch.sigmoid(gate) # 门控调节


class MultiScaleGatedConv1d(nn.Module):
    def __init__(self, dim, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim * 2, k, padding=k // 2) for k in kernels
        ])
        self.proj = nn.Conv1d(dim * 2 * len(kernels), dim * 2, 1)  # 融合多尺度

    def forward(self, x):  # x: [B, C, L]
        outs = [conv(x) for conv in self.convs]   # 多尺度卷积 [B, 2C, L]
        out = torch.cat(outs, dim=1)              # [B, 2C*len(kernels), L]
        out = self.proj(out)                      # 压缩回 [B, 2C, L]
        signal, gate = out.chunk(2, dim=1)        # 划分为信号和门
        return signal * torch.sigmoid(gate)       # 门控调节


class GatedResidualBlock1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gated_conv = MultiScaleGatedConv1d(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask):  # [B, L, C]
        residual = x
        x = x.transpose(1, 2)        # [B, C, L]
        out = self.gated_conv(x)     # [B, C, L]
        out = out.transpose(1, 2)    # [B, L, C]
        out = self.norm(out)
        
        if key_padding_mask is not None:
            # 扩展成 [B, L, 1]，使其能广播到 [B, L, C]
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return residual + out        # 残差加法


    
def kmeans_torch(x, K=2, num_iters=10, eps=1e-6, seed=0):
    """
    简单的 KMeans (单样本). x: [N, C]
    返回: labels [N] (long), centers [K, C]
    说明: 适用于 N ~ 300, K 小（2~8）。
    """
    device = x.device
    N, C = x.shape
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty((K, C), device=device)

    K = min(K, max(1, N))
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    # 随机初始化 centers 从 x 采样
    if N >= K:
        perm = torch.randperm(N, generator=rng, device=device)[:K]
        centers = x[perm].clone()
    else:
        centers = torch.zeros((K, C), device=device)
        centers[:N] = x.clone()

    labels = torch.zeros(N, dtype=torch.long, device=device)
    for it in range(num_iters):
        # 计算平方距离: ||x||^2 - 2 x c^T + ||c||^2
        x_norm2 = (x * x).sum(dim=1, keepdim=True)           # [N,1]
        c_norm2 = (centers * centers).sum(dim=1, keepdim=True).T  # [1,K]
        dist = x_norm2 - 2.0 * (x @ centers.T) + c_norm2     # [N,K]
        new_labels = torch.argmin(dist, dim=1)               # [N]
        if it > 0 and torch.equal(new_labels, labels):
            break
        labels = new_labels
        centers_prev = centers.clone()
        # 更新 centers
        for k in range(K):
            mask = (labels == k)
            if mask.any():
                centers[k] = x[mask].mean(dim=0)
            else:
                # 空簇：重新随机初始化为一个样本点，避免中心为 zero 导致退化
                centers[k] = x[torch.randint(0, N, (1,), device=device)].squeeze(0)
        if torch.max((centers - centers_prev).abs()) < eps:
            break
    return labels, centers




def grouped_self_conv(
    tgt,                       # [B, Q, C]
    enc_topk_logits,           # [B, Q, num_classes]
    conv_mixer,
    num_queries=300,
    K=2,                       # 聚类簇数
    kmeans_iters=10,
    min_keep=10,
    min_keep_bg=1,
    use_pos=False,
    ref_points=None,
    use_sklearn=False,
):
    """
    基于聚类的分组卷积交互：
    - 在 tgt 的最后 num_queries 个 queries 上做聚类
    - 用 enc_topk_logits 的平均前景概率判定前/背景簇
    - 对每个簇做 1D 卷积交互
    """
    B, Q, C = tgt.shape
    device = tgt.device
    M = num_queries

    tgt0 = tgt[:, -M:, :].contiguous()
    logits0 = enc_topk_logits[:, -M:, :].contiguous()

    if logits0.shape[-1] == 1:
        scores_all = torch.sigmoid(logits0.view(B, M))
    else:
        probs = torch.softmax(logits0, dim=-1)
        fg_idx = 1 if logits0.shape[-1] > 1 else 0
        scores_all = probs[..., fg_idx]

    tgt0_new = torch.zeros_like(tgt0)

    tgt_fg_list, tgt_bg_list = [], []
    idx_fg_list, idx_bg_list = [], []
    len_fg_list, len_bg_list = [], []

    from sklearn.cluster import KMeans
    use_sklearn_available = use_sklearn

    for b in range(B):
        x = tgt0[b]
        feats = F.normalize(x.detach(), p=2, dim=1)

        K_b = min(K, max(1, M))

        if use_sklearn_available:
            feats_np = feats.cpu().numpy()
            kmeans = KMeans(n_clusters=K_b, n_init=1, max_iter=kmeans_iters, random_state=b)
            labels_np = kmeans.fit_predict(feats_np)
            labels = torch.from_numpy(labels_np).to(device)
        else:
            labels, centers = kmeans_torch(feats, K=K_b, num_iters=kmeans_iters, seed=b+1)

        cluster_scores = []
        for k in range(K_b):
            mask_k = (labels == k)
            if mask_k.sum() == 0:
                cluster_scores.append(torch.tensor(-1.0, device=device))
            else:
                cluster_scores.append(scores_all[b][mask_k].mean().detach())
        cluster_scores = torch.stack(cluster_scores)

        fg_cluster_idx = int(torch.argmax(cluster_scores).item())
        fg_mask_b = (labels == fg_cluster_idx)
        bg_mask_b = ~fg_mask_b

        if fg_mask_b.sum() < min_keep:
            _, topk_idx = torch.topk(scores_all[b], min(min_keep, M), dim=0)
            new_fg = torch.zeros_like(fg_mask_b)
            new_fg[topk_idx] = True
            fg_mask_b = new_fg
            bg_mask_b = ~fg_mask_b

        if bg_mask_b.sum() < min_keep_bg:
            lowest = torch.argmin(scores_all[b])
            bg_mask_b[lowest] = True
            fg_mask_b[lowest] = False

        idx_fg = torch.nonzero(fg_mask_b, as_tuple=False).squeeze(-1)
        idx_bg = torch.nonzero(bg_mask_b, as_tuple=False).squeeze(-1)
        tgt_fg = x[idx_fg] if idx_fg.numel() > 0 else x.new_zeros((0, C))
        tgt_bg = x[idx_bg] if idx_bg.numel() > 0 else x.new_zeros((0, C))

        tgt_fg_list.append(tgt_fg)
        tgt_bg_list.append(tgt_bg)
        idx_fg_list.append(idx_fg)
        idx_bg_list.append(idx_bg)
        len_fg_list.append(tgt_fg.size(0))
        len_bg_list.append(tgt_bg.size(0))

    max_fg = max(len_fg_list) if len_fg_list else 0
    max_bg = max(len_bg_list) if len_bg_list else 0

    def pad_group_list(group_list, max_len):
        padded = []
        for x in group_list:
            cur = x
            pad = max_len - cur.size(0)
            if pad == 0:
                padded.append(cur)
            else:
                if cur.size(0) == 0:
                    padded.append(cur.new_zeros((max_len, C)))
                else:
                    padded.append(F.pad(cur, (0, 0, 0, pad)))
        return torch.stack(padded, dim=0) if len(padded) > 0 else tgt0.new_zeros((B, 0, C))

    padded_tgt_fg = pad_group_list(tgt_fg_list, max_fg) if max_fg > 0 else tgt0.new_zeros((B, 0, C))
    padded_tgt_bg = pad_group_list(tgt_bg_list, max_bg) if max_bg > 0 else tgt0.new_zeros((B, 0, C))

    if max_fg > 0:
        lengths_fg = torch.tensor(len_fg_list, device=device)
        key_pad_fg = (torch.arange(max_fg, device=device).unsqueeze(0).expand(B, -1) >= lengths_fg.unsqueeze(1)).to(torch.bool)#掩码，T表示pad部分
        padded_tgt_fg_after = conv_mixer(padded_tgt_fg, key_padding_mask=key_pad_fg)
    else:
        padded_tgt_fg_after = padded_tgt_fg

    if max_bg > 0:
        lengths_bg = torch.tensor(len_bg_list, device=device)
        key_pad_bg = (torch.arange(max_bg, device=device).unsqueeze(0).expand(B, -1) >= lengths_bg.unsqueeze(1)).to(torch.bool)
        padded_tgt_bg_after = conv_mixer(padded_tgt_bg, key_padding_mask=key_pad_bg)
    else:
        padded_tgt_bg_after = padded_tgt_bg

    for b in range(B):
        lf = len_fg_list[b]
        lb = len_bg_list[b]
        if lf > 0:
            tgt0_new[b, idx_fg_list[b]] = padded_tgt_fg_after[b, :lf]
        if lb > 0:
            tgt0_new[b, idx_bg_list[b]] = padded_tgt_bg_after[b, :lb]

    tgt_new = torch.cat([tgt[:, :-M, :], tgt0_new], dim=1)
    return tgt_new








#基于聚类分组方法的分组注意力
def grouped_self_attn(
    tgt,                       # [B, Q, C]
    enc_topk_logits,           # [B, Q, num_classes]
    query_pos_embed,
    group_attn,                # callable: group_attn(x, key_padding_mask=...)
    num_queries=300,
    K=2,                       # 聚类簇数（可设为 2,3,4）
    kmeans_iters=10,
    min_keep=10,               # 每张图至少保留的前景数（保障）
    min_keep_bg=1,             # 每张图至少保留的背景数（保障）
    use_pos=False,             # 是否把位置拼到聚类特征
    ref_points=None,           # 如果 use_pos=True，可传 [B, num_queries, 4]
    use_sklearn=False,         # 是否使用 sklearn KMeans（需要 sklearn 安装）
):
    """
    基于聚类的分组自注意力：
    - 在 tgt 的最后 num_queries 个 queries 上做聚类（每张图单独）
    - 用 enc_topk_logits 的平均前景概率给簇打分（平均 score 更高的簇视作前景簇）
    - 对每个簇做组内 self-attn，将结果按原顺序回写回 tgt
    返回：tgt_new 与原接口兼容
    """
    B, Q, C = tgt.shape
    assert Q >= num_queries, "Q must >= num_queries"
    device = tgt.device
    M = num_queries

    tgt0 = tgt[:, -M:, :].contiguous()                     # [B, M, C]
    logits0 = enc_topk_logits[:, -M:, :].contiguous()      # [B, M, num_classes]

    # 取得前景分数（单类情况前景 index 一般为 1）
    if logits0.shape[-1] == 1:
        scores_all = torch.sigmoid(logits0.view(B, M))
    else:
        probs = torch.softmax(logits0, dim=-1)
        fg_idx = 1 if logits0.shape[-1] > 1 else 0
        scores_all = probs[..., fg_idx]                     # [B, M]

    # 为每个 batch 做聚类并收集分组结果
    tgt0_new = torch.zeros_like(tgt0)                       # 存放更新后的 M 个 query
    tgt_fg_list, tgt_bg_list = [], []
    idx_fg_list, idx_bg_list = [], []
    len_fg_list, len_bg_list = [], []

    use_sklearn_available = False
    if use_sklearn:
        try:
            from sklearn.cluster import KMeans
            use_sklearn_available = True
        except Exception:
            use_sklearn_available = False

    for b in range(B):
        x = tgt0[b]               # [M, C]
        # optionally concat position embedding (normalize spatial values)
        if use_pos and (ref_points is not None):
            pos = ref_points[b, -M:, :].to(device)   # [M,4]
            # 简单归一化/缩放 pos，避免位置主导
            pos_feat = (pos - 0.5) * 0.1             # 缩小尺度
            feats_cat = torch.cat([x, pos_feat], dim=1) if x.shape[1] == C + 0 else torch.cat([x, pos_feat], dim=1)
            feats = feats_cat.detach()
        else:
            feats = x.detach()    # detach 出计算图，不反传聚类 op

        # L2 归一化（cosine 风格更稳）
        feats_norm = F.normalize(feats, p=2, dim=1)   # [M, C']

        # 选择 K 实际值（不能大于 M）
        K_b = min(K, max(1, M))

        # 任选 KMeans 实现：sklearn(在 CPU 上) 或 torch 实现（GPU）.labels.shape=[300],centers.shape=[2,256]
        if use_sklearn_available:
            # sklearn 要把数据转 numpy 到 CPU
            feats_np = feats_norm.cpu().numpy()
            kmeans = KMeans(n_clusters=K_b, n_init=1, max_iter=kmeans_iters, random_state=b)
            labels_np = kmeans.fit_predict(feats_np)
            labels = torch.from_numpy(labels_np).to(device)
        else:
            labels, centers = kmeans_torch(feats_norm, K=K_b, num_iters=kmeans_iters, seed=b+1)

        # 对每个簇计算平均 score 作为簇的“前景度”
        cluster_scores = [] #两个簇的平均score
        for k in range(K_b):
            mask_k = (labels == k)
            if mask_k.sum() == 0:
                cluster_scores.append(torch.tensor(-1.0, device=device))  # 空簇赋 -1
            else:
                cluster_scores.append(scores_all[b][mask_k].mean().detach())
        cluster_scores = torch.stack(cluster_scores)    # [K_b]

        # 提取前景mask
        # 策略：取 mean score 最高的簇作为前景簇（你可替换为阈值或 top-n 簇）
        fg_cluster_idx = int(torch.argmax(cluster_scores).item())
        fg_mask_b = (labels == fg_cluster_idx)         # [M]
        bg_mask_b = ~fg_mask_b

        # 保证最小前景 / 最小背景个数
        if fg_mask_b.sum() < min_keep:
            _, topk_idx = torch.topk(scores_all[b], min(min_keep, M), dim=0)
            new_fg = torch.zeros_like(fg_mask_b)
            new_fg[topk_idx] = True
            fg_mask_b = new_fg
            bg_mask_b = ~fg_mask_b

        if bg_mask_b.sum() < min_keep_bg:
            lowest = torch.argmin(scores_all[b])
            bg_mask_b[lowest] = True
            fg_mask_b[lowest] = False

        # 提取 下标和对应tgt
        idx_fg = torch.nonzero(fg_mask_b, as_tuple=False).squeeze(-1)
        idx_bg = torch.nonzero(bg_mask_b, as_tuple=False).squeeze(-1)
        tgt_fg = x[idx_fg] if idx_fg.numel() > 0 else x.new_zeros((0, C))
        tgt_bg = x[idx_bg] if idx_bg.numel() > 0 else x.new_zeros((0, C))

        # 形成列表(内容查询、对应下标、对应数量)
        tgt_fg_list.append(tgt_fg)
        tgt_bg_list.append(tgt_bg)
        idx_fg_list.append(idx_fg)
        idx_bg_list.append(idx_bg)
        len_fg_list.append(tgt_fg.size(0))
        len_bg_list.append(tgt_bg.size(0))

    # padding：把每个 batch 的组补齐到最大长度，便于并行 attention
    max_fg = max(len_fg_list) if len_fg_list else 0
    max_bg = max(len_bg_list) if len_bg_list else 0

    # pad helper：安全处理 0 长度的项
    def pad_group_list(group_list, max_len):
        padded = []
        for x in group_list:
            cur = x
            pad = max_len - cur.size(0)
            if pad == 0:
                padded.append(cur)
            else:
                if cur.size(0) == 0:
                    padded.append(cur.new_zeros((max_len, C)))
                else:
                    padded.append(F.pad(cur, (0, 0, 0, pad)))  # pad height
        return torch.stack(padded, dim=0) if len(padded) > 0 else tgt0.new_zeros((B, 0, C))

    if max_fg > 0:
        padded_tgt_fg = pad_group_list(tgt_fg_list, max_fg)  # [B, max_fg, C]
    else:
        padded_tgt_fg = tgt0.new_zeros((B, 0, C))

    if max_bg > 0:
        padded_tgt_bg = pad_group_list(tgt_bg_list, max_bg)  # [B, max_bg, C]
    else:
        padded_tgt_bg = tgt0.new_zeros((B, 0, C))

    # key_padding_mask: True 表示该位置为 padding，需要被屏蔽（shape: [B, L]）
    if max_fg > 0:
        lengths_fg = torch.tensor(len_fg_list, device=device)
        key_pad_fg = (torch.arange(max_fg, device=device).unsqueeze(0).expand(B, -1) >= lengths_fg.unsqueeze(1)).to(torch.bool)#掩码，前面是F，后面（pad）是1
    else:
        key_pad_fg = None

    if max_bg > 0:
        lengths_bg = torch.tensor(len_bg_list, device=device)
        key_pad_bg = (torch.arange(max_bg, device=device).unsqueeze(0).expand(B, -1) >= lengths_bg.unsqueeze(1)).to(torch.bool)
    else:
        key_pad_bg = None

    # NaN 检查（输入层）
    assert not torch.isnan(padded_tgt_fg).any(), "NaN found in padded_tgt_fg before attention"
    assert not torch.isnan(padded_tgt_bg).any(), "NaN found in padded_tgt_bg before attention"

    # 调用 group_attn：优先用 key_padding_mask 参数（batch_first=True）
    # 若你的 group_attn 的签名不同，请相应调整
    padded_tgt_fg_after = padded_tgt_fg if max_fg == 0 else group_attn(padded_tgt_fg, key_padding_mask=key_pad_fg)
    padded_tgt_bg_after = padded_tgt_bg if max_bg == 0 else group_attn(padded_tgt_bg, key_padding_mask=key_pad_bg)

    # NaN 检查（输出层）
    assert not torch.isnan(padded_tgt_fg_after).any(), "NaN found in padded_tgt_fg after attention"
    assert not torch.isnan(padded_tgt_bg_after).any(), "NaN found in padded_tgt_bg after attention"

    # 按原索引把 group 输出写回 tgt0_new（保持原顺序）
    for b in range(B):
        lf = len_fg_list[b]
        lb = len_bg_list[b]
        if lf > 0:
            tgt0_new[b, idx_fg_list[b]] = padded_tgt_fg_after[b, :lf]
        if lb > 0:
            tgt0_new[b, idx_bg_list[b]] = padded_tgt_bg_after[b, :lb]

    # 拼回原始 tgt
    tgt_new = torch.cat([tgt[:, :-M, :], tgt0_new], dim=1)
    assert not torch.isnan(tgt_new).any(), "NaN in final tgt_new"
    return tgt_new


# def grouped_self_attn(tgt, enc_topk_logits, group_attn, num_queries, score_threshold=0.8, min_keep=10, min_keep_bg=1):
#     """
#     分组注意力（带掩码 + 顺序还原）

#     Args:
#         tgt: [B, Q, C] decoder 输入
#         enc_topk_logits: [B, Q, num_classes] encoder 筛选后的分类预测
#         group_attn: 组内自注意力模块 (支持 attn_mask)
#         num_queries: 每个 batch 的 query 数
#         score_threshold: 前景阈值
#         min_keep: 至少保留的前景 query 数

#     Return:
#         tgt_new: [B, Q, C] 更新后的 decoder 输入
#     """
#     B, Q, C = tgt.shape

#     # 1. 拿出后 num_queries 个 → decoder 专用 queries
#     tgt0 = tgt[:, -num_queries:, :]              # [B, num_queries, C]
#     logits0 = enc_topk_logits[:, -num_queries:, :]  # [B, num_queries, num_classes]

#     # 2. 计算前景分数（取前景类别的概率）
#     #   假设 num_classes=2: 第 0 维是背景, 第 1 维是前景
#     scores = torch.softmax(logits0, dim=-1)[..., 1]  # [B, num_queries]

#     # 3. 前景 mask
#     fg_mask = scores > score_threshold  # [B, num_queries]

#     # 保证至少 min_keep
#     topk_scores, topk_idx = torch.topk(scores, min_keep, dim=1)  # [B, min_keep]
#     fg_mask.scatter_(1, topk_idx, True)
#     bg_mask = ~fg_mask
#     # 保证至少 min_keep_bg 个背景
#     bg_counts = bg_mask.sum(dim=1)
#     for i in range(B):
#         if bg_counts[i] < min_keep_bg:  
#             # 找分数最低的 query 强行归为背景
#             lowest_idx = torch.argmin(scores[i])
#             bg_mask[i, lowest_idx] = True
#             fg_mask[i, lowest_idx] = False

#     # 4. tgt 分组（保持索引）
#     tgt_fg = [tgt0[i][fg_mask[i]] for i in range(B)]  # list of [k, C]
#     tgt_bg = [tgt0[i][bg_mask[i]] for i in range(B)]  # list of [num_queries-k, C]
#     idx_fg = [torch.nonzero(fg_mask[i], as_tuple=False).squeeze(-1) for i in range(B)]
#     idx_bg = [torch.nonzero(bg_mask[i], as_tuple=False).squeeze(-1) for i in range(B)]

#     # 5. 所有 batch 的 tgt 数量补齐
#     max_fg = max(x.size(0) for x in tgt_fg)
#     max_bg = max(x.size(0) for x in tgt_bg)
#     padded_tgt_fg = torch.stack([F.pad(x, (0, 0, 0, max_fg - x.size(0))) for x in tgt_fg], dim=0)
#     padded_tgt_bg = torch.stack([F.pad(x, (0, 0, 0, max_bg - x.size(0))) for x in tgt_bg], dim=0)

#     # 6. 生成 attn_mask: True 表示不能交互
#     attn_mask_fg = torch.arange(max_fg, device=tgt.device)[None, :].expand(B, -1) >= torch.tensor(
#         [x.size(0) for x in tgt_fg], device=tgt.device)[:, None]
#     attn_mask_bg = torch.arange(max_bg, device=tgt.device)[None, :].expand(B, -1) >= torch.tensor(
#         [x.size(0) for x in tgt_bg], device=tgt.device)[:, None]

#     # 7. 组内自注意力（带掩码）
#     assert not torch.isnan(padded_tgt_fg).any(), "NaN detected in fg tensor"
#     assert not torch.isnan(padded_tgt_bg).any(), "NaN detected in bg tensor"

#     padded_tgt_fg = group_attn(padded_tgt_fg, attn_mask=attn_mask_fg)  # [B, max_fg, C]
#     padded_tgt_bg = group_attn(padded_tgt_bg, attn_mask=attn_mask_bg)  # [B, max_bg, C]

#     assert not torch.isnan(padded_tgt_fg).any(), "NaN detected after fg attn"
#     assert not torch.isnan(padded_tgt_bg).any(), "NaN detected after bg attn"

#     # 8. 按原始顺序还原
#     tgt0_new = torch.zeros_like(tgt0)  # [B, num_queries, C]
#     for i in range(B):
#         tgt0_new[i, idx_fg[i]] = padded_tgt_fg[i, :idx_fg[i].size(0)]
#         tgt0_new[i, idx_bg[i]] = padded_tgt_bg[i, :idx_bg[i].size(0)]

#     # 9. 拼回 decoder 输入
#     tgt_new = torch.cat([tgt[:, :-num_queries, :], tgt0_new], dim=1)

#     return tgt_new
    




if __name__ == "__main__":
    class GroupSelfAttention(nn.Module):
        def __init__(self, hidden_dim=256, nhead=8, dropout=0.0):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x, key_padding_mask):
            # x: [B, N, C]
            attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)   #注意key_padding_mask和attn_mask的区别
            x = self.norm(x + attn_out)
            return x

    group_attn = GroupSelfAttention()
    tgt = torch.rand([8,300,256])
    enc_topk_logits = torch.rand([8,300,2])
    tgt = grouped_self_attn(tgt,enc_topk_logits, group_attn)