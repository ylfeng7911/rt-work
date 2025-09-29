import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 




class DecoPlusDecoderLayer(nn.Module):
    '''Define a layer for DECO+ Decoder'''
    def __init__(self,d_model, dropout=0.,
                 layer_scale_init_value=1e-6, normalize_before=False, qH=15, qW=20, inceptH=13, inceptW=15, branch_ratio=0.25): 

        self.qH = qH
        self.qW = qW
        self.inceptH = inceptH
        self.inceptW = inceptW

        # The SIM module   
        self.dwconv1 = InceptionDWConv2d(d_model, square_kernel_size=3, band_kernel_size_h=self.inceptH, band_kernel_size_w=self.inceptW, branch_ratio=branch_ratio)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(d_model, 4 * d_model) 
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * d_model, d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = nn.Identity()
        
        
    
    def forward(self, tgt, memory,
                query_pos_embed=None):
        
        tgt = tgt.permute(0,2,1)
        bs, n_embed, _ = tgt.shape
        tgt = tgt.reshape(bs, n_embed, self.qH, self.qW)    #[b,c,15,20]
        query_pos_embed = query_pos_embed.permute(0,2,1)
        query_pos_embed = query_pos_embed.reshape(bs, n_embed, self.qH, self.qW)    #[b,c,15,20]
        
        # SIM  把[b,300,c]的TGT分解为[b,c,h,w]做卷积
        _, _, h, w = memory.shape
        tgt2 = tgt + query_pos_embed
        tgt2 = self.dwconv1(tgt2)
        tgt2 = tgt2.permute(0, 2, 3, 1) # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)     #类似1x1卷积
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)     #类似1x1卷积
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2.permute(0,3,1,2) # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)