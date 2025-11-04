import torch
from torch import nn
import torch.nn.functional as F


class ConvNormLayer(nn.Module):
    """Conv + BN + Activation"""
    def __init__(self, in_c, out_c, k, s, p=None, act=nn.SiLU()):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SeparableConv2d(nn.Module):
    """Depthwise + Pointwise Conv"""
    def __init__(self, in_c, out_c, act=nn.SiLU()):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = act

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPN_WeightedFusion(nn.Module):
    """可学习加权融合 (Fast Normalized Fusion)"""
    def __init__(self, num_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        # Normalize the weights
        w = F.relu(self.weights)
        weight = w / (torch.sum(w, dim=0) + 1e-4)
        fused = 0
        for i in range(len(inputs)):
            fused += weight[i] * inputs[i]
        return fused

class BiFPNLayer(nn.Module):
    """单层 BiFPN 融合逻辑"""
    def __init__(self, channels, act=nn.SiLU()):
        """
        channels: list of feature map channels, e.g. [C3, C4, C5]    --input 
        """
        super().__init__()
        self.num_levels = len(channels)         #输入尺度数

        # 横向 1×1 conv 调整通道(可省?)
        self.lateral_convs = nn.ModuleList([
            ConvNormLayer(c, channels[0], 1, 1, act=act) for c in channels
        ])

        # 自顶向下融合 (FPN)
        self.top_down_fusions = nn.ModuleList([
            BiFPN_WeightedFusion(2) for _ in range(self.num_levels - 1)
        ])
        self.top_down_convs = nn.ModuleList([
            SeparableConv2d(channels[0], channels[0], act=act) for _ in range(self.num_levels - 1)
        ])

        # 自底向上融合 (PAN)
        self.bottom_up_fusions = nn.ModuleList([
            BiFPN_WeightedFusion(2) for _ in range(self.num_levels - 1)
        ])
        self.bottom_up_convs = nn.ModuleList([
            SeparableConv2d(channels[0], channels[0], act=act) for _ in range(self.num_levels - 1)
        ])

    def forward(self, inputs):      #inputs是从大到小
        """
        inputs: list of feature maps [P3, P4, P5]
        """
        assert len(inputs) == self.num_levels
        feats = [lateral(x) for x, lateral in zip(inputs, self.lateral_convs)]

        # ----- Top-down path -----     上采样+加权融合+卷积层
        for i in range(self.num_levels - 1, 0, -1):     # 2,1
            up = F.interpolate(feats[i], size=feats[i - 1].shape[-2:], mode='nearest')
            feats[i - 1] = self.top_down_fusions[i - 1]([feats[i - 1], up])
            feats[i - 1] = self.top_down_convs[i - 1](feats[i - 1])

        # ----- Bottom-up path -----    最大值池化+加权融合+卷积层
        for i in range(0, self.num_levels - 1):         # 0,1
            down = F.max_pool2d(feats[i], 2, stride=2)
            feats[i + 1] = self.bottom_up_fusions[i]([feats[i + 1], down])
            feats[i + 1] = self.bottom_up_convs[i](feats[i + 1])

        return feats


class BiFPN(nn.Module):
    """多层堆叠的 BiFPN"""
    def __init__(self, in_channels, out_channels, num_layers=6, act=nn.SiLU()):
        super().__init__()
        self.num_layers = num_layers                    #2
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(in_channels, act=act) for _ in range(num_layers)
        ])
        # 输出层 1x1 conv 保持一致通道
        self.out_convs = nn.ModuleList([
            ConvNormLayer(256, out_channels, 1, 1, act=act)
            for _ in range(len(in_channels))
        ])

    def forward(self, inputs):
        for layer in self.bifpn_layers:
            inputs = layer(inputs)
        outs = [conv(x) for conv, x in zip(self.out_convs, inputs)]     #每个尺度特征图做1x1卷积(或许可省)
        return outs

