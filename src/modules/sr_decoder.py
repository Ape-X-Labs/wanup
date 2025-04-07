import torch
from torch import nn
import torch.nn.functional as F
from src.modules.wanvae import ResidualBlock, CausalConv3d, RMS_norm,  Resample, AttentionBlock
from einops import rearrange
from natten import  NeighborhoodAttention2D, NeighborhoodAttention3D

__all__ = [
    'SpatialNeighbourhoodAttentionBlock',
    'SpatiaTemporallNeighbourhoodAttentionBlock',
    'SRDecoder',
]

CACHE_T = 2

class SpatialNeighbourhoodAttentionBlock(nn.Module):
    """
    Spatial neighbourhood attention  with a single head.
    """

    def __init__(self, dim, kernel_size=(3, 3), dilation=(1, 1)):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.na = NeighborhoodAttention2D(
            dim,
            num_heads=1,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        x = self.norm(x).view(b * t, h, w, c)
        x = self.na(x)
        x = x.view(b, c, t, h, w)
        return x + identity
    
    
class SpatiaTemporallNeighbourhoodAttentionBlock(nn.Module):
    """
    Spatial and temporal neighbourhood attention  with a single head.
    """

    def __init__(self, dim, kernel_size=(3, 3, 3), dilation=(1, 1, 1)):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim, images=False)
        self.na = NeighborhoodAttention3D(
            dim,
            num_heads=1,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.shape
        x = self.norm(x).view(b, t, h, w, c)
        x = self.na(x)
        x = x.view(b, c, t, h, w)
        return x + identity

class SRDecoder(nn.Module):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[1.0, 2.0],
                 temporal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample
        self.dropout = dropout

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)
        
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))
        
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        upsamples = []
        
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(SpatiaTemporallNeighbourhoodAttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temporal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
                
        # Here we add super sampling
        # get dims from the last two layers
        upsamples.append(Resample(out_dim, mode='upsample2d'))
        scale *= 2.0
        for i in range(2):
            # then we add more residual blocks
            if i == 0:
                out_dim = out_dim // 2
                in_dim = out_dim
            if i == 1:
                out_dim = int(out_dim / 1.5)
                in_dim = out_dim
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(SpatiaTemporallNeighbourhoodAttentionBlock(out_dim))
                in_dim = out_dim
            if i < 1:
                upsamples.append(Resample(out_dim, mode='upsample2d_15'))
                scale *= 1.5

        self.upsamples = nn.Sequential(*upsamples)
        
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), 
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1)
        )
    
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x
        