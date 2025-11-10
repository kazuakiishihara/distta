import torch
import torch.nn as nn

from networks.transmorph.TransMorph import *

class GatingBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        # x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TransMorph_DS_MoSE(nn.Module):
    def __init__(self, config, img_size, n_experts=48, k_experts=24, n_categories=30):
        super(TransMorph_DS_MoSE, self).__init__()
        self.img_size = img_size
        self.n_experts = n_experts
        self.k_experts = k_experts
        self.n_categories = n_categories
        self.t_warmup = 100

        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(pretrain_img_size=img_size[0],
                                            patch_size=config.patch_size,
                                            in_chans=config.in_chans,
                                            embed_dim=config.embed_dim // 2, # !!! Dual-steam
                                            depths=config.depths,
                                            num_heads=config.num_heads,
                                            window_size=config.window_size,
                                            mlp_ratio=config.mlp_ratio,
                                            qkv_bias=config.qkv_bias,
                                            drop_rate=config.drop_rate,
                                            drop_path_rate=config.drop_path_rate,
                                            ape=config.ape,
                                            spe=config.spe,
                                            rpe=config.rpe,
                                            patch_norm=config.patch_norm,
                                            use_checkpoint=config.use_checkpoint,
                                            out_indices=config.out_indices,
                                            pat_merg_rf=config.pat_merg_rf,
                                            )
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        # Shape Experts: (k, c, d, h, w)
        self.shape_experts = nn.Parameter(
            torch.randn(self.n_experts, self.n_categories, self.img_size[0]//4, self.img_size[1]//4, self.img_size[2]//4) # !!!
        )
        # Gating Network
        # Args: (1, channel, d, h, w), where d, h, w = D/4, H/4, W/4
        # Return: (1, k*c, d, h, w)
        self.gating_up2 = GatingBlock(embed_dim//2, n_experts*n_categories, skip_channels=0, use_batchnorm=False)

    def apply_topk_sparsity(self, x: torch.Tensor, K_select: int) -> torch.Tensor:
        """
        Applies Top-K sparsification based on Equation (1) from the MoSE paper.

        Selects the top 'k' experts with the largest absolute values of G(e_i) 
        at each pixel location, and sets the others to 0.
        This implementation assumes that the first dimension (dim=0) of the input 'x' 
        is the expert dimension.

        Args:
            x (torch.Tensor):
                            The input tensor G(e_i). 
                            Shape: (k, c, d, h, w)
                            k: Number of experts (denoted as 'n' in the paper).
                            c: Number of categories.
                            d, h, w: Spatial dimensions (depth, height, width).
            K_select (int): The number of top experts to select (denoted as 'k' in the paper).
                            K_select must be less than or equal to k.
        Returns:
            torch.Tensor: The sparsified tensor ~G(e_i) (same shape as x).
        """
        num_experts = x.shape[0]
        if K_select > num_experts:
            raise ValueError(f"K_select ({K_select}) cannot exceed the total number of experts ({num_experts})")
        if K_select < 0:
            raise ValueError(f"K_select ({K_select}) must be non-negative")

        abs_x = torch.abs(x)
        _, topk_indices = torch.topk(abs_x, K_select, dim=0, largest=True)
        mask = torch.zeros_like(x, dtype=torch.float32)
        mask.scatter_(dim=0, index=topk_indices, value=1.0)
        g_tilde = x * mask
        return g_tilde

    def forward(self, inputs):
        source, tar = inputs
        x = torch.cat((source, tar), dim=1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats_m = self.transformer(source)
        out_feats_f = self.transformer(tar)

        if self.if_transskip:
            f1 = torch.cat((out_feats_m[-2], out_feats_f[-2]), dim=1)
            f2 = torch.cat((out_feats_m[-3], out_feats_f[-3]), dim=1)
            f3 = torch.cat((out_feats_m[-4], out_feats_f[-4]), dim=1)
        else:
            f1 = None
            f2 = None
            f3 = None

        # Decoder in Registration
        x = self.up0(torch.cat((out_feats_m[-1], out_feats_f[-1]), dim=1), f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)

        # MoSE
        gating_weights_m = self.gating_up2(out_feats_m[-4]).view(self.n_experts, self.n_categories, self.img_size[0]//4, self.img_size[1]//4, self.img_size[2]//4) # Shape: [1, k*c, d, h, w] → [k, c, d, h, w]
        gating_weights_f = self.gating_up2(out_feats_f[-4]).view(self.n_experts, self.n_categories, self.img_size[0]//4, self.img_size[1]//4, self.img_size[2]//4)
        if 1 > self.t_warmup: # !!!
            sparse_gating_weights_m = self.apply_topk_sparsity(gating_weights_m, K_select=self.k_experts)
            sparse_gating_weights_f = self.apply_topk_sparsity(gating_weights_f, K_select=self.k_experts)
        else: # Use all the expert during warm-up
            sparse_gating_weights_m = gating_weights_m
            sparse_gating_weights_f = gating_weights_f
        Im_shape_map = torch.einsum('kcxyz, kcxyz -> cxyz', sparse_gating_weights_m, self.shape_experts) # Shape: [c, d, h, w]
        If_shape_map = torch.einsum('kcxyz, kcxyz -> cxyz', sparse_gating_weights_f, self.shape_experts)
        Im_seg = torch.softmax(Im_shape_map, dim=0).unsqueeze(0) # Shape: [1, c, d, h, w]
        If_seg = torch.softmax(If_shape_map, dim=0).unsqueeze(0)
        return out, flow, (Im_seg, gating_weights_m), (If_seg, gating_weights_f)

moving, fixed = torch.randn((1, 1, 192, 224, 160)), torch.randn((1, 1, 192, 224, 160))

from networks.transmorph.TransMorph import CONFIGS as CONFIGS_TM
config = CONFIGS_TM['TransMorph_DS']
img_size = (192, 224, 160)
model = TransMorph_DS_MoSE(config, img_size)
output = model((moving, fixed))