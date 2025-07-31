"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from ...core import register


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,            #[b,500,2]
                reference_points,   #[b,500,1,4]
                memory,     #[b,sum_hw,c]
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,   #无
                query_pos_embed=None):      #[b,500,c]作为query的位置嵌入
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))
        assert not torch.isnan(tgt).any(), f"NaN detected in tgt1"
        
        
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        assert not torch.isnan(tgt).any(), f"NaN detected in tgt2"
        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        assert not torch.isnan(tgt).any(), f"NaN detected in tgt3"
        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))

        assert not torch.isnan(tgt).any(), f"NaN detected in tgt4"
        
        return tgt


class TransformerDecoder(nn.Module):    #参考点和偏移量全是预测得到的
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, mask_dim = 256, num_heads=8):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.boltzmann_sampling: dict = {"mask_threshold": 0.5, "do_boltzmann": True, "sample_ratio": 0.1, "base_temp": 1,}
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-4)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)  
        self.num_heads = num_heads
        self.mask_factor = -10

    def forward(self,
                tgt,                #合并后的[b,500,256]        在解码器中一直是 嵌入 的身份
                ref_points_unact,   #合并后的[b,500,4]          在解码器中一直是 坐标预测（参考点） 的身份
                memory,             #[b,sum_hw,c]
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):  #无
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)       #[b,500,1,4] 这个是输入参考点（DDETR中是位置query直接预测的），不同于迭代更新的参考点（相当于query的位置编码，或者叫位置query）。在DDETR中前者是后者预测的
            query_pos_embed = query_pos_head(ref_points_detach)     #[b,500,4] -> [b,500,256]表示query的位置嵌入,
            if attn_mask is not None:
                attn_mask_bol = self.boltzmann(tgt + query_pos_embed, layer_id= i)
                attn_mask = attn_mask * attn_mask_bol * self.mask_factor    #逻辑与 可加性掩码系数
                attn_mask = attn_mask.clamp(min=-100, max=0)
            output = layer(output, ref_points_input, memory,        #解码器层  [b,500,256]
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))   #bbox预测结果漂移量+参考点 [b,500,4]

            #查询最后预测 类别和BBOX偏移量
            if self.training:
                dec_out_logits.append(score_head[i](output))    #类别预测
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)       #bbox预测
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            
            #参考点更新。上一层的预测结果作为下一层的参考点
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox    #推理就不detach

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
    #[6,b,500,4]  [6,b,500,2]
    
    def boltzmann(self, tgt_mask, layer_id=-1): #根据输入嵌入 生成注意力掩码
        # Boltzman sampling on attention mask
        threshold = self.boltzmann_sampling["mask_threshold"]  # original threshold for masked attention
        do_boltzmann = self.boltzmann_sampling["do_boltzmann"]  # whether to do Boltzman sampling
        sample_ratio = self.boltzmann_sampling["sample_ratio"]  # number of iid samples as a ratio of total number of masked tokens
        base_temp = self.boltzmann_sampling["base_temp"]  # base temperature for Boltzman sampling
              
        decoder_output = self.decoder_norm(tgt_mask)      #[b,nq,c]
        mask_embed = self.mask_embed(decoder_output)        #[b,nq,c]
        attn = torch.matmul(mask_embed, mask_embed.transpose(-1, -2)) / (mask_embed.shape[-1] ** 0.5)  #自交互 [b,nq,nq]
        
        attn = (
            attn.sigmoid() #归一化
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )       #[b, nq, nq] -> [b * n_head, nq, nq]          .detach()   

        if do_boltzmann:
            # probability of Boltzman sampling
            Temp = base_temp / (2 + layer_id)  #2  temperature decays with layer number (first layer from id -1)
            boltzmann_prob = torch.exp(attn / Temp)
            boltzmann_prob = (boltzmann_prob * (attn < threshold).float())  # 大于阈值不掩，小于阈值概率掩

            boltzmann_prob = boltzmann_prob / (boltzmann_prob.sum(dim=-1, keepdim=True) + 1e-6)
            
            assert not torch.isnan(boltzmann_prob).any(), f"NaN detected in attn_mask in layer {layer_id} in 1"
            
            # sample from Boltzman distribution n times
            n_samples = int(attn.shape[-1] * sample_ratio)  # number of iid samples on the tokens    HW/10
            masked_prob = (1 - boltzmann_prob) ** n_samples  # 对于小于阈值的部分来说，masked_prob就是最后的值
                       
            rand_tensor = torch.rand_like(boltzmann_prob)
            boltzmann_mask = 1.0 - torch.sigmoid((rand_tensor - masked_prob) * 100)
            attn_mask = (1.0 - torch.sigmoid((attn - threshold) * 100)) * boltzmann_mask  # 近似 and 操作（可导）
        else:
            attn_mask = (attn_mask < threshold).bool()
        return  attn_mask


@register()
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],       #[256,256,256]
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 version='v1'):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes  #2
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        if version == 'v1':
            self.enc_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim,)
            )
        else:
            self.enc_output = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(hidden_dim, hidden_dim)),
                ('norm', nn.LayerNorm(hidden_dim,)),
            ]))

        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()       #输入映射：1x1卷积+归一化：调整C
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),   #256 -> 256
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):   #3-3=0
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features   3个特征图做映射
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]     #[b,c,h,w]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs    展平特征图列表，尺寸列表，起始下标列表
        feat_flatten = []           #[b,hw,c]
        spatial_shapes = []         #[[h,w], [], []]
        level_start_index = [0, ]       #[0, hw0, hw0+hw1, hw0+hw1+hw2]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)        #[b,sum_hw,c]
        level_start_index.pop()                             #[0, hw0, hw0+hw1]
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:  #推理的时候？
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []    
        for lvl, (h, w) in enumerate(spatial_shapes):   #每个像素中心一个anchor
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')    #grid_y，grid_x都是[h,w],沿着y/x方向递增
            grid_xy = torch.stack([grid_x, grid_y], -1)     #合成坐标[h,w,2]
            valid_WH = torch.tensor([w, h]).to(dtype)   #[h,w]
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH   #让anchor落在网格中心并归一化   [1,h,w,2]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)    #BBOX的宽高 [b,h,w,2] 。越高层越大
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))   #[b,h,w,4] -> [b,hw,4]

        anchors = torch.concat(anchors, 1).to(device)   #[b,sum_hw,4]
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)    #[b,sum_hw,1] 检查xyhw是否都在区间(eps,1-eps)
        anchors = torch.log(anchors / (1 - anchors))    #坐标变换到sigmoid inverse
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)   #合法不变，不合法令为inf

        return anchors, valid_mask      #[b,sum_hw,4]   [b,sum_hw,1] 


    def _get_decoder_input(self,
                           memory,                      #[b, sum_hw, c] 混合编码器学到的
                           spatial_shapes,              #[[h, w],[],[]]
                           denoising_class=None,        #[b, num_denoising, 256]
                           denoising_bbox_unact=None):  #[b, num_denoising, 4]
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)  ##[b,sum_hw,4]   [b,sum_hw,1] 
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory) #线性层 [b,sum_hw,c]
        #编码器的输出预测
        enc_outputs_class = self.enc_score_head(output_memory)  #线性层预测类别
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors   #MLP预测坐标偏移    anchors相当于参考点

        #根据类别预测值topk筛选
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)     #[bs,sum_hw,2]->[bs,sum_hw]->[bs,300] 
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))   #[b,300,4]取300个作为参考点

        enc_topk_bboxes = F.sigmoid(reference_points_unact) #参考点的激活版，取300个坐标预测
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)  #[b,500,4]  合并DN坐标查询
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))     #[bs,300,2]取300个类别预测

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])) #混合编码器输出，[bs,300,c] 内容查询
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1) #合并DN类别查询

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits    #1/2取500个,3/4取300个（经过预测的）
#[b, 500, 256], [b, 500, 4], [b, 300, 4], [b, 300, 2]   前两个用于解码器，后两个用于aux_LOSS。后3个都是预测，第一个是嵌入



    def forward(self, feats, targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)    #展平特征图和2列表
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                label_noise_ratio=self.label_noise_ratio,   #0.5
                    box_noise_scale=self.box_noise_scale, ) #1.0
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)      

        # decoder
        out_bboxes, out_logits = self.decoder(                  #[6,b,500,4],[6,b,500,2]
            target,                 #合并后的类别查询
            init_ref_points_unact,  #合并后的bbox查询
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,     #bbox预测 MLP
            self.dec_score_head,    #类别预测 线性层
            self.query_pos_head,    #MLP
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)     #分成[6,b,300,4]和[6,b,200,4]
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]} #取最后解码器的输出

        if self.training and self.aux_loss: #前5层解码器输出 + DN输出
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])       #  out['aux_outputs'][0]  - [4]
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))     #out['aux_outputs'][5]  编码器的输出
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)    ##  out['dn_aux_outputs'][0]  - [5]
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
