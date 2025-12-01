import json
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import cv2
import pandas as pd
import numpy as np

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from .cropping import crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out


def map_func(label_path, labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
    labels = [label.lower() for label in labels]

    df = pd.read_csv(label_path, sep='\t')
    id_to_nyu40class = pd.Series(df['nyu40class'].str.lower().values, index=df['id']).to_dict()

    nyu40class_to_newid = {cls: labels.index(cls) + 1 if cls in labels else labels.index('other') + 1 for cls in set(id_to_nyu40class.values())}

    id_to_newid = {id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()}

    return np.vectorize(lambda x: id_to_newid.get(x, labels.index('other') + 1) if x != 0 else 0)

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@dataclass
class ScannetCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    llff_hold: int = 8
    test_ids: list[int] = field(default_factory=lambda: [1, 4])
    num_of_inputs: int = 2
    context_eval: bool = False


@dataclass
class DatasetScannetCfgWrapper:
    scannet_eval: ScannetCfg


class DatasetScannet(IterableDataset):
    cfg: ScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: ScannetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        self.map_func = map_func(os.path.join(cfg.roots[0], 'scannetv2-labels.combined.tsv'))
        
        self.full_class = ['bag', 'bathtub', 'bed', 'blinds', 'books', 'bookshelf', 'box',
                        'cabinet', 'ceiling', 'chair', 'clothes', 'counter', 'curtain',
                        'desk', 'door', 'dresser', 'floor', 'floor mat', 'lamp', 'mirror',
                        'night stand', 'otherfurniture', 'otherprop', 'otherstructure',
                        'paper', 'person', 'picture', 'pillow', 'refridgerator', 'shelves',
                        'shower curtain', 'sink', 'sofa', 'table', 'television', 'toilet',
                        'towel', 'wall', 'whiteboard', 'window']
        
        # Collect chunks.
        with open(os.path.join(cfg.roots[0], f'selected_seqs_test.json'), 'r') as f:
            self.scenes = json.load(f)
            self.scenes = {k: sorted(v) for k, v in self.scenes.items() if len(v) > 0}
            ignored_scenes = ['scene0696_02']
            for key in ignored_scenes:
                if key in self.scenes:
                    del self.scenes[key]
                    
        self.scene_list = list(self.scenes.keys())
        self.invalidate = {scene: {} for scene in self.scene_list}
        

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        image, depthmap, intrinsics = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


    def __iter__(self):
        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.scene_list = [
                chunk
                for chunk_index, chunk in enumerate(self.scene_list)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for scene_id in self.scene_list:
            # Load the chunk.
            selected_views = [i for i in range(len(self.scenes[scene_id])) if i % self.cfg.llff_hold in self.cfg.test_ids]

            for target_view in selected_views:
                left_idxs = [max(target_view - i, 0) for i in range(1, (self.cfg.num_of_inputs + 2)//2)]
                right_idxs = [min(target_view + i, len(self.scenes[scene_id]) - 1) for i in range(1, (self.cfg.num_of_inputs + 2)//2)]
                
                

                idxs = []
                for l, r in zip(left_idxs, right_idxs):
                    idxs.extend([l, r])
                
                idxs.append(target_view)
                print(idxs)
                    
                extrinsics_list = []
                intrinsics_list = []
                images_list = []
                label_list = []
                
                for idx in idxs:
                    impath = os.path.join(self.cfg.roots[0], scene_id, 'images', f'{self.scenes[scene_id][idx]}.jpg')
                    meta_data_path = impath.replace('jpg','npz')
                    depthmap_path = impath.replace('jpg','png').replace('images','depths')
                    label_path = impath.replace('jpg','png').replace('images','labels')
                    
                    input_metadata = np.load(meta_data_path)
                    camera_pose = input_metadata['camera_pose'].astype(np.float32)
                    has_inf = np.any(np.isinf(camera_pose))
                    if has_inf:
                        print('has_inf')
                        continue
                    
                    intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
                    ## normalize it

                    
                    rgb_image = imread_cv2(impath)
                    depthmap = imread_cv2(depthmap_path, options=cv2.IMREAD_UNCHANGED)
                    maskmap = np.ones_like(depthmap) * 255
                    labelmap = imread_cv2(label_path, options=cv2.IMREAD_UNCHANGED)
                    
                    depth_mask_map = np.stack([depthmap,maskmap,labelmap], axis=-1)
                    
                    rgb_image, depth_mask_map, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depth_mask_map, intrinsics, self.cfg.input_image_shape, info=impath)

                    intrinsics[0,:] /= self.cfg.input_image_shape[0]
                    intrinsics[1,:] /= self.cfg.input_image_shape[1]

                    
                    depthmap = depth_mask_map[:, :, 0]
                    maskmap = depth_mask_map[:, :, 1]
                    labelmap = depth_mask_map[:, :, 2]
                    # map labelmap
                    labelmap = self.map_func(labelmap)
                    
                    depthmap = (depthmap.astype(np.float32) / 1000)
                    num_valid = (depthmap > 0.0).sum()
                    
                    if num_valid==0:
                        print("num_valid is 0")
                    
                    extrinsics_list.append(camera_pose)
                    intrinsics_list.append(intrinsics)
                    images_list.append(self.to_tensor(rgb_image))
                    label_list.append(torch.from_numpy(labelmap.astype(np.int64)).unsqueeze(0))
                    
                extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0).astype(np.float32))
                intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0).astype(np.float32))
                images = torch.stack(images_list, dim=0)
                labels = torch.cat(label_list, dim=0)
                    
                context_extrinsics = extrinsics[:self.cfg.num_of_inputs]
                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        print(
                            f"Skipped {scene} because of baseline out of range: "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1    
                    
                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[0:1], extrinsics)
                
                if self.cfg.context_eval:
                    for n in range(self.cfg.num_of_inputs):
                        example = {
                            "context": {
                                "extrinsics": extrinsics[:self.cfg.num_of_inputs],
                                "intrinsics": intrinsics[:self.cfg.num_of_inputs],
                                "image": images[:self.cfg.num_of_inputs],
                                "label": labels[:self.cfg.num_of_inputs],
                                "near": self.get_bound("near", len(idxs[:self.cfg.num_of_inputs])) / scale,
                                "far": self.get_bound("far", len(idxs[:self.cfg.num_of_inputs])) / scale,
                                "index": idxs[:self.cfg.num_of_inputs],
                                "overlap": 0,
                            },
                            "target": {
                                "extrinsics": extrinsics[n:n+1],
                                "intrinsics": intrinsics[n:n+1],
                                "image": images[n:n+1],
                                "label": labels[n:n+1],
                                "near": self.get_bound("near", len(idxs[n:n+1])) / scale,
                                "far": self.get_bound("far", len(idxs[n:n+1])) / scale,
                                "index": idxs[self.cfg.num_of_inputs:],
                            },
                            "scene": scene_id,
                        }

                        yield example
                else:
                    example = {
                        "context": {
                            "extrinsics": extrinsics[:self.cfg.num_of_inputs],
                            "intrinsics": intrinsics[:self.cfg.num_of_inputs],
                            "image": images[:self.cfg.num_of_inputs],
                            "label": labels[:self.cfg.num_of_inputs],
                            "near": self.get_bound("near", len(idxs[:self.cfg.num_of_inputs])) / scale,
                            "far": self.get_bound("far", len(idxs[:self.cfg.num_of_inputs])) / scale,
                            "index": idxs[:self.cfg.num_of_inputs],
                            "overlap": 0,
                        },
                        "target": {
                            "extrinsics": extrinsics[self.cfg.num_of_inputs:],
                            "intrinsics": intrinsics[self.cfg.num_of_inputs:],
                            "image": images[self.cfg.num_of_inputs:],
                            "label": labels[self.cfg.num_of_inputs:],
                            "near": self.get_bound("near", len(idxs[self.cfg.num_of_inputs:])) / scale,
                            "far": self.get_bound("far", len(idxs[self.cfg.num_of_inputs:])) / scale,
                            "index": idxs[self.cfg.num_of_inputs:],
                        },
                        "scene": scene_id,
                    }

                    yield example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
