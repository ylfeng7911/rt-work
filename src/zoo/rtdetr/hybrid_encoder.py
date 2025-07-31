"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None,groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            groups = groups,
            bias=bias,
            )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):        #从高到低
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [256,256,256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = ConvNormLayer(256, self.inter_dim, 3, 2, act='relu',groups=32)
            self.stride_level_2 = ConvNormLayer(256, self.inter_dim, 3, 2, act='relu',groups=32)
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='relu',groups=32)
        elif level==1:
            # self.compress_level_0 = ConvNormLayer(512, self.inter_dim, 1, 1, act='relu',groups=32)    不用通道调整，直接上采样
            self.stride_level_2 = ConvNormLayer(256, self.inter_dim, 3, 2, act='relu',groups=32)
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='relu',groups=32)
        elif level==2:
            # self.compress_level_0 = ConvNormLayer(512, self.inter_dim, 1, 1, act='relu',groups=32)
            self.expand = ConvNormLayer(self.inter_dim, 256, 3, 1, act='relu',groups=32)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='relu')
        self.weight_level_1 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='relu')
        self.weight_level_2 = ConvNormLayer(self.inter_dim, compress_c, 1, 1, act='relu')

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):#3个尺度(从上往下)特征图,特征融合
        if self.level==0:
            level_0_resized = x_level_0     #           [b,512,h,w]
            level_1_resized = self.stride_level_1(x_level_1)    #下采样[b,512,h,w]
            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)  #下采样1
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)            #下采样2[b,512,h,w]

        elif self.level==1:
            # level_0_compressed = self.compress_level_0(x_level_0)       #通道调节c=256
            level_0_resized =F.interpolate(x_level_0, scale_factor=2, mode='nearest')#上采样[b,256,h,w]
            level_1_resized =x_level_1              #[b,256,h,w]
            level_2_resized =self.stride_level_2(x_level_2)#下采样
        elif self.level==2:
            # level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(x_level_0, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized) #[b,8,h,w]  每幅图对应一个8维权重
        level_1_weight_v = self.weight_level_1(level_1_resized) #[b,8,h,w]
        level_2_weight_v = self.weight_level_2(level_2_resized) #[b,8,h,w]
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)       #[b,24,h,w]
        levels_weight = self.weight_levels(levels_weight_v)                                         #[b,3,h,w] 每幅图对应3个权重
        levels_weight = F.softmax(levels_weight, dim=1)         #[b,3,h,w] 

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]                       #[b,1,h,w] * [b,512,h,w]

        out = self.expand(fused_out_reduced)            #[b,256,h,w]

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
        
        


class RepVggBlock(nn.Module):       #(卷积1 + 卷积3)-激活     H/W不变  
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):       #H/W不变
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):     # v1
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
                
                
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # # top-down fpn
        # self.lateral_convs = nn.ModuleList()
        # self.fpn_blocks = nn.ModuleList()
        # for _ in range(len(in_channels) - 1, 0, -1):        #3,2
        #     self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act)) #1卷积-归一化-激活函数     C/H/W不变
        #     self.fpn_blocks.append(
        #         CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))#3卷积-归一化-激活  C减半，H/W不变

        # # bottom-up pan
        # self.downsample_convs = nn.ModuleList()
        # self.pan_blocks = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):       #0,1
        #     self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))  #C不变，H/W减半
        #     self.pan_blocks.append(
        #         CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)        #C减半,H/W不变
        #     )
            
        #ASFF
        self.ASFF_block = nn.ModuleList([ASFF(i) for i,ch in enumerate(in_channels)])
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]     #1x1卷积--> c = 256
        
        # encoder   最高级特征图 应用编码器（交互语义信息）
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):  #0,2
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1) #展平
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)       #

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()   #[b,c,h,w]

        # # broadcasting and fusion
        # # FPN：inner_outs变为：从下往上3层融合特征图E1 E2 E3(代码逻辑很绕，结合论文图来看)
        # inner_outs = [proj_feats[-1]]      
        # for idx in range(len(self.in_channels) - 1, 0, -1):     #2,1        理解：上层（论文中F5和S4）经1x1卷积+2倍插值和下层融合
        #     feat_heigh = inner_outs[0]          # S5    E2'
        #     feat_low = proj_feats[idx - 1]      # S4    S3
        #     feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)    #1x1卷积，下标是0,1,2
        #     inner_outs[0] = feat_heigh          # E3    E2
        #     upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')  #上采样
        #     inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))#concat再减半C
        #     inner_outs.insert(0, inner_out)     #[E2',E3]    

        # # PAN：outs是：E1和每一次PAN融合后的结果（尺度从下往上）
        # outs = [inner_outs[0]]
        # for idx in range(len(self.in_channels) - 1):        #0,1           理解：下层（论文中）经过下采样后和上层融合
        #     feat_low = outs[-1]         # E1 第一次PAN融合的结果 
        #     feat_height = inner_outs[idx + 1]       #E2  E3
        #     downsample_feat = self.downsample_convs[idx](feat_low)      #下采样E3
        #     out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1)) #concat再减半C
        #     outs.append(out)

        outs = [self.ASFF_block[i](proj_feats[2], proj_feats[1], proj_feats[0]) for i in range(len(self.in_channels))]
        outs.reverse()

        return outs
