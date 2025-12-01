from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable, Any
from itertools import accumulate

import moviepy as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
from torchmetrics import JaccardIndex, Accuracy
import os
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose
from ..misc.image_io import prep_image, save_image, save_video, visualize_attention_map
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from src.model.clip import clip
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .utils import  save_segmap, run_pca

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    visualize_gaussian_token: int = -1
    forward_vfm: bool = False
    labels: list[str] = field(default_factory=lambda: ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other'])
    color_hex_list: list[str] = field(default_factory=lambda: ['#000000', '#E6194B','#3CB44B','#FFE119','#4363D8','#F58231','#911EB4','#42D4F4','#808000'])

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    random_select_context_view: bool = False
    reproj_model: str = 'none' # 'vggt' or 'dino'
    feature_rendering_loss: float = 0.0


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass
    
class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        vggt = None,
        dino = None,
        lseg_feature_extractor = None,
        clip = None,
        mode: str = "train"
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.mode=mode
            
        self.vggt=vggt
        self.lseg_feature_extractor = lseg_feature_extractor

        if dino is not None:
            self.dino_model = dino['model']
            self.dino_processor = dino['processor'] if 'processor' in dino else None
            self.latent_mean = dino['latent_mean'] if dino['latent_mean'] is not None else 0
            self.latent_var = dino['latent_var'] if dino['latent_var'] is not None else 1
        else:
            self.dino_model, self.dino_processor = None, None
            
        if clip is not None:
            self.clip_model = clip['model']
        else:
            self.clip_model = None
            
        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.miou = JaccardIndex(
            task="multiclass",
            num_classes=len(self.test_cfg.labels) + 1,
            ignore_index=0,
        )
        
        self.acc = Accuracy(
            task="multiclass",
            num_classes=len(self.test_cfg.labels) + 1,
            ignore_index=0,
        )

        self.per_image_ious = []
        self.per_image_accs = []
        

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        if self.train_cfg.random_select_context_view:
            if self.global_rank == 0:
                num_ctx_views = torch.randint(2, batch['context']['extrinsics'].shape[1] + 1, size=(1,)).item()
            else:
                num_ctx_views = 0

            if dist.is_available() and dist.is_initialized():
                t = torch.tensor([num_ctx_views], dtype=torch.int64, device=self.device)
                dist.broadcast(t, src=0)
                num_ctx_views = int(t.item())
                        
            ctx = {}
            for key, value in batch['context'].items():
                if key == 'overlap':
                    ctx[key] = value
                    continue
                ctx[key] = value[:, :num_ctx_views, ...].contiguous()
            batch['context'] = ctx
        
        # Run the model.
        visualization_dump = {}
        
        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None        
        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, context_feature = context_feature)        
        
        output = self.decoder.forward(
            gaussians,
            torch.cat([batch["target"]["extrinsics"],batch["context"]["extrinsics"]],dim=1),
            torch.cat([batch["target"]["intrinsics"],batch["context"]["intrinsics"]],dim=1),
            torch.cat([batch["target"]["near"],batch["context"]["near"]],dim=1),
            torch.cat([batch["target"]["far"],batch["context"]["far"]],dim=1),
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
            global_step=self.global_step
        )

        target_gt = torch.cat([batch["target"]["image"], ((batch["context"]["image"] + 1) / 2)], dim=1)

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=torch.cat([batch["target"]["image"], ((batch["context"]["image"] + 1) / 2)], dim=1))
            self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # only for coarse gaussians
            total_loss = total_loss + loss
            
        if self.train_cfg.feature_rendering_loss > 0:
            B,CV,_,H,W = batch['context']['image'].shape
            B,TV,_,H,W = batch['target']['image'].shape
            feature = self.forward_foundation_model(torch.cat((batch["context"]["image"], batch['target']['image'] * 2 - 1), dim=1), interpolate=False)
            feature = torch.cat((feature[:,CV:], feature[:,:CV]), dim=1)        ## ordering: target -> context
            
            gaussian_feature = output.feature
            B,N,_,FH,FW = feature.shape
            
            gaussian_feature = F.interpolate(gaussian_feature.reshape(B*N,-1,H,W), size=(FH, FW), mode='bilinear', align_corners=False).reshape(B,N,-1,FH,FW)
            
            gaussian_feature = F.normalize(gaussian_feature, p=2, dim=2)
            feature = F.normalize(feature, p=2, dim=2)
            
            feature_rendering_loss = F.cosine_similarity(gaussian_feature, feature.detach(), dim=2)                
            feature_rendering_loss = (1 - feature_rendering_loss).mean()
            
            self.log("loss/feature_rendering_loss", feature_rendering_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            total_loss = total_loss + self.train_cfg.feature_rendering_loss * feature_rendering_loss
                
        self.log("loss/total", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
            and (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}",
                f"low_pass_filter = {self.decoder.low_pass_filter:.3f}",
            )
        self.log("info/global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss
    
    @torch.no_grad()
    def forward_foundation_model(self, input_image, interpolate=True, vggt_tracking=False):
        B, V, C, H, W = input_image.shape       ## [-1~1]
                
        with torch.no_grad():              
            if self.train_cfg.reproj_model == 'dinov2':
                context_feature = self.dino_model.get_intermediate_layers(input_image.reshape(B*V,C,H,W), reshape=True)[0].reshape(B,V,-1,H//14,W//14)
                
            elif 'dinov3' in self.train_cfg.reproj_model:
                context_feature = self.dino_model(**self.dino_processor((input_image.reshape(B*V,C,H,W) + 1)/2. * 255, return_tensors='pt').to(self.device))
                context_feature = rearrange(context_feature['last_hidden_state'][:,5:], 'b (h w) c -> b c h w', h=H//16, w=W//16)
                context_feature = context_feature.reshape(B,V,-1,H//16,W//16)
                
            elif self.train_cfg.reproj_model == 'lseg':
                context_feature = self.lseg_feature_extractor.extract_features(input_image.reshape(B*V,3,H,W))
                context_feature = context_feature.reshape(B,V,-1,H//2,W//2)
                
            elif 'vggt' in self.train_cfg.reproj_model:
                if self.train_cfg.reproj_model=='vggt_tracking':
                    context_feature = self.vggt(input_image)['feature']
                else:
                    aggregated_tokens_list, patch_start_idx = self.vggt(input_image, only_feature=True)
                    vggt_features = aggregated_tokens_list[-1][:,:,patch_start_idx:]
                    context_feature = rearrange(vggt_features, "b n (h w) c -> b n c h w", h=H//14, w=W//14)

            elif self.train_cfg.reproj_model == 'maskclip':
                input_image = (input_image + 1) / 2
                mean, std = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device), torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)
                input_image = (input_image - mean[None,None,:,None,None]) / std[None,None,:,None,None]
                context_feature = self.clip_model(input_image.reshape(B*V,C,H,W))
                context_feature = context_feature.reshape(B,V,-1,H,W)            

        if interpolate:
            context_feature = F.interpolate(context_feature.reshape(B*V,-1,context_feature.shape[-2],context_feature.shape[-1]), size=(H//14, W//14), mode='bilinear', align_corners=False).reshape(B,V,-1,H//14,W//14)
                
        ## B, V, C, H, W
        return context_feature
    


    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        
        if h!=224 or w!=224:
            b, cv, _, ch, cw = batch['context']['image'].shape
            batch['context']['image'] = F.interpolate(batch['context']['image'].reshape(b*cv,3,ch,cw), size=(224,224), mode='bilinear', align_corners=False).reshape(b,cv,3,224,224)
        
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        if self.test_cfg.visualize_gaussian_token>=0:
            outputs = []
            def hook_fn(module, input, output):
                outputs.append(output)
            _ = self.encoder.gmae_decoder.layers[0][0].to_qkv.register_forward_hook(hook_fn)
            _ = self.encoder.gmae_decoder.layers[1][0].to_qkv.register_forward_hook(hook_fn)
            
        # Render Gaussians.
        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None            
        
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                context_feature=context_feature
            )
        
        if self.test_cfg.visualize_gaussian_token>=0:
            num_heads = self.encoder.gmae_decoder.layers[0][0].heads
            gaussian_token_idx = self.test_cfg.visualize_gaussian_token
            
            name = get_cfg()["wandb"]["name"]
            path = self.test_cfg.output_path / name
            (scene,) = batch["scene"]
            os.makedirs(path / scene , exist_ok=True)
            
            visualize_attention_map(
                outputs[0],batch, num_heads, gaussian_token_idx, batch['context']['image'].shape[3:5], patch_size = self.encoder.patch_size, output_path= path / scene / f"{gaussian_token_idx}_layer1")
            
            visualize_attention_map(
                outputs[1], batch, num_heads, gaussian_token_idx, batch['context']['image'].shape[3:5], patch_size = self.encoder.patch_size, output_path= path / scene /  f"{gaussian_token_idx}_layer2")
            
            C0 = 0.28209479177387814
            gaussians.harmonics[:, gaussian_token_idx, :, 0] = (torch.tensor([[1,0,0]]) - 0.5) / C0
        
        if self.test_cfg.align_pose and (not self.test_cfg.forward_vfm):
            output = self.test_step_align(batch, gaussians, verbose=True)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            
            psnr = compute_psnr(rgb_gt, rgb_pred).mean()
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": psnr,
            }
            methods = ['ours']

            self.log_dict(all_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )
            
        projections = render_projections(gaussians,256,extra_label="",low_pass = self.decoder.low_pass_filter, draw_label=False)[0]
        save_image(projections[2], path / f"{scene}_projections.png")
        
        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            error_map = (rgb_gt - rgb_pred.clamp(0,1)).abs()
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*error_map), "Error Map"),
            )
            save_image(comparison, path / f"{scene}_{psnr:.3f}.png")
            
            if self.encoder.cfg.feature_dim:
                gaussian_feature = output.feature
                B,N,C,H,W = gaussian_feature.shape

                if 'dinov2' in self.train_cfg.reproj_model or 'vggt' in self.train_cfg.reproj_model:
                    B,V,C,H,W = batch['target']['image'].shape
                    target_image = F.interpolate(batch['target']['image'].reshape(B*V,C,H,W), size=(224,224), mode='bilinear', align_corners=False).reshape(B,V,C,224,224)
                else:
                    target_image = batch['target']['image']

                foundation_feature = self.forward_foundation_model((target_image * 2 - 1),interpolate=False)
                
                save_dir = path / scene / "seg"
                save_gt_dir = path / scene / "seg_gt"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_gt_dir.mkdir(parents=True, exist_ok=True)

                if 'dino' not in self.train_cfg.reproj_model and 'vggt' not in self.train_cfg.reproj_model:
                    pca_images, pca_vggt_images = [], []
                    V = gaussian_feature.shape[1]
                    
                    for i in range(foundation_feature.shape[1]):
                        pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_images.append(pca_gaussian_img.squeeze(dim=0))
                        
                        pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                    cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
                    cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 
                    
                    preds = []
                    targets = []
                    for i, (index, g_upfeat) in enumerate(zip(batch["target"]["index"][0], gaussian_feature)):
                        if self.test_cfg.forward_vfm:
                            g_upfeat = foundation_feature[i]
                            if 'lseg' in self.train_cfg.reproj_model:
                                g_upfeat = self.lseg_feature_extractor.scratch.output_conv(g_upfeat)
                        
                        if 'text' in batch['target'].keys():
                            labelset = batch['target']['text']
                            labelset = [label[0] for label in labelset]
                        else:  
                            labelset = self.test_cfg.labels
                        
                        if self.train_cfg.reproj_model == 'lseg':
                            pred = self.lseg_feature_extractor.decode_feature(g_upfeat, labelset=labelset)
                        else:
                            pred = self.clip_decode_feature(g_upfeat, labelset=labelset)
                                                    
                        pred = torch.argmax(pred, dim=1) + 1

                        target = batch["target"]["label"][0]
                        targets.append(target)
                        
                        iou_val = self.miou(pred.flatten(), target.flatten())
                        acc_val = self.acc(pred.flatten(), target.flatten())

                        self.log("test/miou", iou_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                        self.log("test/acc", acc_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

                        print(f"IoU: {iou_val.item()}, Acc: {acc_val.item()}")

                        self.per_image_ious.append(iou_val.item())
                        self.per_image_accs.append(acc_val.item())
                        
                        preds.append(pred)

                    seg_preds = []
                    seg_tgts = []
                            
                    for pred in preds:
                        for index, seg_pred in zip(batch["target"]["index"][0], pred):
                            labels = self.test_cfg.labels[:8]
                            seg_pred_vis = save_segmap(gaussian_feature, seg_pred, index, save_dir, labels, self.test_cfg.color_hex_list)
                            seg_preds.append(seg_pred_vis)
                            
                    for target in targets:
                        for index, seg_tgt in zip(batch["target"]["index"][0], target):
                            labels = self.test_cfg.labels[:8]
                            seg_tgt_vis = save_segmap(gaussian_feature, seg_tgt, index, save_gt_dir, labels, self.test_cfg.color_hex_list)
                            seg_tgts.append(seg_tgt_vis)
                            


                    
                    comparison = hcat(
                        add_label(vcat(*context_img), "Context"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                        add_label(vcat(*pca_images), "Feature (Pred)"),
                        add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                        add_label(vcat(*cocnat_pca_images), "Feature_cat (Pred)"),
                        add_label(vcat(*seg_preds), "Segmentation (Pred)"),
                        add_label(vcat(*seg_tgts), "Segmentation (GT)"),
                    )
                    save_image(comparison, path / f"{scene}_{psnr:.3f}_pca.png")
                    
                else:
                    pca_images, pca_vggt_images = [], []
                    
                    V = gaussian_feature.shape[1]
                    
                    for i in range(foundation_feature.shape[1]):
                        pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_images.append(pca_gaussian_img.squeeze(dim=0))
                        
                        pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                        
                    cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
                    cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 

                    for i in range(len(cocnat_pca_images)):
                        save_image(cocnat_pca_images[i], path / f"{scene}_cocnat_pca{i}.png")

                    comparison = hcat(
                        add_label(vcat(*context_img), "Context"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                        add_label(vcat(*pca_images), "Feature (Pred)"),
                        add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                        add_label(vcat(*cocnat_pca_images), "Feature_cat (Pred)"),
                    )
                    save_image(comparison, path / f"{scene}_{psnr:.3f}_pca.png")
                    
    @torch.no_grad()
    def clip_decode_feature(self, image_features, labelset=''):
        imshape = image_features.shape      # B C H W
        
        text = clip.tokenize(labelset)
        
        text = text.to(image_features.device)
        if 'maskclip' in self.train_cfg.reproj_model:
            text_features = self.clip_model.model.model.encode_text(text)
        else:
            text_features = self.clip_model.encode_text(text)
        image_features = image_features.permute(0,2,3,1).reshape(-1, image_features.shape[1])
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = image_features.half() @ text_features.t()
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        return out

                    
    # image-level iou and acc
    def on_test_epoch_end(self):
        mean_iou = sum(self.per_image_ious) / len(self.per_image_ious) if self.per_image_ious else 0.0
        mean_acc = sum(self.per_image_accs) / len(self.per_image_accs) if self.per_image_accs else 0.0

        print("mIoU:", mean_iou)
        print("Acc:", mean_acc)

        self.log("test/mIoU", mean_iou, prog_bar=True)
        self.log("test/Acc", mean_acc, prog_bar=True)

        # Reset lists for next epoch
        self.per_image_ious.clear()
        self.per_image_accs.clear()

    def test_step_align(self, batch, gaussians, verbose=False):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            
            if verbose:
                logger = tqdm(range(self.test_cfg.pose_align_steps))
            else:
                logger = range(self.test_cfg.pose_align_steps)
                
            prev_loss = None
            patience_counter = 0
            patience_limit = 10 
            
            with self.benchmarker.time("optimize"):
                for i in logger:
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0

                    
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=batch["target"]["image"])
                        total_loss = total_loss + loss
                        
                    if verbose:
                        logger.set_description(f"pose optim step {i}; loss = {total_loss:.6f}")

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

                    if prev_loss is not None:
                        delta = abs(total_loss.item() - prev_loss)
                        if delta < 0.00001:
                            patience_counter += 1
                            if patience_counter >= patience_limit and i >= 100:
                                break
                        else:
                            patience_counter = 0
                    prev_loss = total_loss.item()


        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        del pose_optimizer

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
                
        if self.train_cfg.random_select_context_view:
            num_ctx_views = torch.randint(2, batch['context']['extrinsics'].shape[1] + 1, size=(1,)).item()

            for key, value in batch['context'].items():
                if key == 'overlap':
                    continue
                batch['context'][key] = value[:, :num_ctx_views, ...]

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        
        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None
        
        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, context_feature = context_feature)
        
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        self.log(f"val/lpips", lpips)
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )
        
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        
        if self.train_cfg.feature_rendering_loss > 0:
            context_output = self.decoder.forward(
                gaussians,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                "depth",
            )
            
            gaussian_feature = output.feature
            B,N,C,H,W = gaussian_feature.shape
            
            context_gaussian_feature = context_output.feature
            B,CN,C,H,W = context_gaussian_feature.shape
            gaussian_feature = torch.cat((context_gaussian_feature, gaussian_feature), dim=1)
            
            foundation_feature = self.forward_foundation_model(torch.cat((batch["context"]["image"], batch['target']['image'] * 2 - 1), dim=1))
            context_foundation_features = self.forward_foundation_model(batch["context"]["image"])
            
            
            pca_images, pca_vggt_images = [], []
            for i in range(foundation_feature.shape[1]):
                pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_images.append(pca_gaussian_img.squeeze(dim=0))
                
                pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                
            cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
            cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 
            
            pca_context_images = []   
            for i in range(context_foundation_features.shape[1]):
                pca_context_img = run_pca(context_foundation_features[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_context_images.append(pca_context_img.squeeze(dim=0))
                
            context[1] = pca_context_images[0]
            context[3] = pca_context_images[1]
            
            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                add_label(vcat(*pca_images), "Feature (Prediction)"),
                add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                add_label(vcat(*cocnat_pca_images), "Feature_cat (Prediction)"),
            )
            
            self.logger.log_image(
                f"PCA",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                    low_pass = self.decoder.low_pass_filter,
                )[0]
            )
        
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        if self.encoder.cfg.feature_dim:
            context_feature = self.forward_foundation_model(batch['context']['image'])
        else:
            context_feature = None
        
        gaussians = self.encoder(batch["context"], self.global_step, context_feature = context_feature)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name or 'dpt_gs_head' in name or 'gmae' in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def tensor_mem_mb(t):
    return t.nelement() * t.element_size() / 1024**2