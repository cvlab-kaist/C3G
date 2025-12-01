from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .backbone_croco import BackboneCrocoCfg
from .vggt.vggt import VGGT
from .croco.misc import fill_default_args, freeze_all_params
from ....geometry.camera_emb import get_intrinsic_embedding

inf = float('inf')

class BackboneVGGT(VGGT):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        # self.patch_embed_cls = cfg.patch_embed_cls
        super().__init__()
        self.dec_depth = self.aggregator.depth - 1      ## It is just used for DPT head.
        self.enc_embed_dim = self.aggregator.embed_dim * 2
        self.dec_embed_dim = self.aggregator.embed_dim * 2


    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        for key, value in ckpt.items():
            new_ckpt['backbone.' + key] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [],
            'encoder':  [self.aggregator],
        }
        freeze_all_params(to_be_frozen[freeze])

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # step 1: encoder input images
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images_all)

        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        

        return aggregated_tokens_list, shape, patch_start_idx

    @property
    def patch_size(self) -> int:
        return 14

    @property
    def d_out(self) -> int:
        return 2048
