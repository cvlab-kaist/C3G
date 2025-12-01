from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from .heads import DPTHead
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .common.gmae import Transformer, InstillTransformer
from .backbone.croco.misc import fill_default_args, freeze_all_params

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat

@dataclass
class EncoderVGGTCfg:
    name: Literal["vggt"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    num_gaussians: int
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    freeze_backbone: bool = False
    decoder_depth: int = 2
    gaussians_per_token: int = 1
    gaussian_feature_dim : int = 0
    feature_dim: int = 0        ## don't setting it manually, it will be set in main.py
    different_learnable_tokens: bool = True
    

class EncoderVGGT(Encoder[EncoderVGGTCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderVGGTCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter) if self.cfg.pose_free else GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.aggregator.patch_size
        self.raw_gs_dim = 3 + 1 + self.gaussian_adapter.d_in

        self.dpt_head = DPTHead(2048) if 'vggt' in cfg.backbone.name else None
        freeze_all_params([self.dpt_head])
        if hasattr(self.backbone, 'camera_head'):
            del self.backbone.camera_head, self.backbone.point_head, self.backbone.depth_head, self.backbone.track_head

        if cfg.freeze_backbone:
            self.backbone.set_freeze('encoder')
                    
        transformer_dim = 2048
        
        self.gaussian_tokens = nn.Parameter(torch.randn(cfg.num_gaussians, transformer_dim))
        self.anchor_positions = nn.Parameter(torch.tensor([[0,0,1]]).repeat(cfg.num_gaussians,1), requires_grad=False)      
        
        
        if self.cfg.feature_dim > 0:
            self.gmae_decoder = InstillTransformer(
                dim = transformer_dim,
                depth = cfg.decoder_depth,
                heads = 16,
                dim_head = transformer_dim//16,
                mlp_dim = transformer_dim * 2,
                cfg = cfg,
            )
            
            if self.cfg.different_learnable_tokens:
                self.gaussian_tokens_feature = nn.Parameter(torch.randn(cfg.num_gaussians, self.cfg.feature_dim))
                self.feature_gmae_to_gaussians = nn.Linear(self.cfg.feature_dim, self.cfg.gaussian_feature_dim * cfg.gaussians_per_token)
            else:
                self.feature_gmae_to_gaussians = nn.Linear(transformer_dim, self.cfg.gaussian_feature_dim * cfg.gaussians_per_token)
            
        else:
            self.gmae_decoder = Transformer(
                dim = transformer_dim,
                depth = cfg.decoder_depth,
                heads = 16,
                dim_head = transformer_dim//16,
                mlp_dim = transformer_dim * 2,
                cfg = cfg,
            )
        
        self.gmae_to_gaussians = nn.Linear(transformer_dim, self.raw_gs_dim * cfg.gaussians_per_token)



    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        context_feature: Optional[Tensor] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.freeze_backbone:
            with torch.no_grad():
                dec, shape, patch_start_idx = self.backbone(context, return_views=False)
        else:
            dec, shape, patch_start_idx = self.backbone(context, return_views=False)
            
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                res = self.dpt_head(dec, context['image'], patch_start_idx)
                vis_depth = res[0][..., -1]   # shape: (B, N, H, W) ## for visualization
            
        dec_feat = dec[-1][:, :, patch_start_idx:]
        dec_feat = rearrange(dec_feat, "b v n d -> b (v n) d")
        all_decoder_tokens = torch.cat((dec_feat, self.gaussian_tokens.unsqueeze(0).expand(b, -1, -1),), dim=1)
        
        if self.cfg.feature_dim > 0 and context_feature is not None:
            # context_feature = rearrange(context_feature, "b v c h w -> b (v h w) c")
            # context_feature = torch.cat((context_feature, self.gaussian_tokens_feature.unsqueeze(0).expand(b, -1, -1)), dim=1)
            context_feature = rearrange(context_feature, "b v c h w -> b (v h w) c")
            if self.cfg.different_learnable_tokens:
                context_feature = torch.cat((context_feature, self.gaussian_tokens_feature.unsqueeze(0).expand(b, -1, -1)), dim=1)
            else:
                context_zero_feature = torch.zeros((b, context_feature.shape[1], dec_feat.shape[2] - context_feature.shape[2]), device=device)
                context_feature = torch.cat((context_feature, context_zero_feature), dim=-1)
                
                context_feature = torch.cat((context_feature, self.gaussian_tokens.unsqueeze(0).expand(b, -1, -1)), dim=1)
                
        if self.cfg.feature_dim > 0:
            decoded_tokens, decoded_feature_token = self.gmae_decoder(all_decoder_tokens, mask=None, context_feature=context_feature)  # b n d
        else:
            decoded_tokens = self.gmae_decoder(all_decoder_tokens, mask=None)  # b n d
            
        gaussian_params = self.gmae_to_gaussians(decoded_tokens[:, -self.gaussian_tokens.shape[0]:])  # b n d(3+1+d')
        
        if self.cfg.feature_dim > 0:
            feature_gaussian_params = self.feature_gmae_to_gaussians(decoded_feature_token[:, -self.gaussian_tokens.shape[0]:])  # b n d(3+1+d')
        
        gaussian_params = rearrange(gaussian_params, "b n (gpt c) -> b (n gpt) c", gpt=self.cfg.gaussians_per_token, c=self.raw_gs_dim)
        
        pts_all = gaussian_params[:, :, :3].unsqueeze(-2) + self.anchor_positions.unsqueeze(dim=0).repeat(b,self.cfg.gaussians_per_token,1).unsqueeze(dim=2) # b n 3
        depths = pts_all[..., -1].unsqueeze(-1)

        # except_feature = (-self.cfg.gaussian_feature_dim or None) if not self.cfg.feature_dim > 0 else None
        # gaussians = gaussian_params[:, :, 3:except_feature]
        gaussians = gaussian_params[:,:,3:]
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        if self.cfg.feature_dim > 0:
            gaussian_feature = rearrange(feature_gaussian_params, "b n (gpt c) -> b (n gpt) c", gpt=self.cfg.gaussians_per_token, c=self.cfg.gaussian_feature_dim)
        else:
            gaussian_feature = None

        # Convert the features and depths into Gaussians.
        if self.cfg.pose_free:
            gaussians = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b n srf c -> b n srf () c"),
                features = gaussian_feature,
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump['depth'] = vis_depth.unsqueeze(-1).unsqueeze(-1)
            visualization_dump["scales"] = None
            visualization_dump["rotations"] = None
            visualization_dump["means"] = None
            visualization_dump['opacities'] = None

        return Gaussians(
            rearrange(
                gaussians.means,
                "b n srf spp xyz -> b (n srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b n srf spp i j -> b (n srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b n srf spp c d_sh -> b (n srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b n srf spp -> b (n srf spp)",
            ),
            gaussians.features if gaussians.features is not None else None
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
