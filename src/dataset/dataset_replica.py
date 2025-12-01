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
from collections import defaultdict

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
from .cropping import crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from .utils import read_intrinsics_binary, read_extrinsics_binary, readColmapCameras

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
class ReplicaCfg(DatasetCfgCommon):
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
    test_ids: list[int] = field(default_factory=lambda: [1])
    context_eval: bool = False


@dataclass
class DatasetReplicaCfgWrapper:
    replica: ReplicaCfg


class DatasetReplica(IterableDataset):
    cfg: ReplicaCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: ReplicaCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # self.scene_list = ['office3','office4','room0','room1']
        self.scene_list = ['office3','office4','room1']
        
        self.scenes = {
            'office3': list(range(0,80)),
            'office4': list(range(0,75)),
            'room0': list(range(0,80)),
            'room1': list(range(0,80)),
        }
        
        self.labels = {'office3': ['wall','ceiling','floor','chair','table'],
                       'office4': ['wall','ceiling','floor','chair','tv-screen','table'],
                       'room0': ['wall','ceiling','floor','sofa','table', 'blinds'],
                       'room1': ['wall','ceiling','floor','bed', 'blinds'],
                       }


        self.mapping = {
            'office3': {0:0, 8:3, 10:0, 12:1, 14:0, 15:0, 17:0, 20:4, 22:0, 29:4, 31:2, 35:5, 37:1, 40:3, 47:2, 56:0, 62:0, 76:4, 79:0, 80:5, 82:0, 83:0, 88:0, 92:0, 93:1, 95:0, 97:1},
            'office4': {0:0, 8:3, 10:0, 17:0, 20:4, 22:0, 31:2, 37:0, 40:3, 47:2, 56:5, 80:6, 87:5, 92:0, 93:1, 95:0, 97:1},
            'room0': {0:0, 3:0, 11:0, 12:6, 13:0, 18:0, 19:0, 20:4, 29:4, 31:2, 37:1, 40:3, 44:0, 47:0, 59:1, 60:0, 63:0, 64:0, 65:0, 76:4, 78:0, 79:0, 80:5, 91:0, 92:0, 93:1, 95:0, 97:6, 98:3},
            'room1': {0:0, 3:0, 7:4, 11:4, 12:5, 13:0, 18:0, 26:0, 31:2, 37:1, 40:3, 44:0, 47:0, 54:0, 56:0, 59:1, 61:4, 64:0, 79:0, 91:0, 92:0, 93:1, 95:0, 97:5, 98:3},
        }
        # Collect chunks.
        

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
            # scene_id = scene_path.name.split('/')[-1]
            selected_views = [i for i in self.scenes[scene_id] if i % self.cfg.llff_hold in self.cfg.test_ids]

            camera_pose_path = os.path.join(self.cfg.roots[0], scene_id)

            cameras_extrinsic_file = os.path.join(camera_pose_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(camera_pose_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


            cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(self.cfg.roots[0], scene_id, 'images'))
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
            
            for target_view in selected_views:
                if scene_id == 'office4' and (target_view == 33 or target_view == 36):
                    continue 
                idxs = [max(target_view - 1,0), min(target_view + 1,len(cam_infos)-1), target_view]              
                extrinsics_list = []
                intrinsics_list = []
                images_list = []
                label_list = []
                original_images_list = []
                                
                for idx in idxs:
                    rgb_image = cam_infos[idx].image
                    # label_path = impath.replace('jpg','png').replace('images','labels')
                    image_name = int(cam_infos[idx].image_name.split('_')[-1])
                    
                    camera_pose = np.zeros((4,4), dtype=np.float32)
                    camera_pose[:3,:3] = cam_infos[idx].R
                    camera_pose[:3,3] = cam_infos[idx].T
                    camera_pose[3,3] = 1.0
                    
                    intrinsics = np.zeros((3,3), dtype=np.float32)
                    focal_x = cam_infos[idx].width / (2 * np.tan(cam_infos[idx].FovX / 2))
                    focal_y = cam_infos[idx].height / (2 * np.tan(cam_infos[idx].FovY / 2))
                    intrinsics[0,0] = focal_x
                    intrinsics[1,1] = focal_y
                    intrinsics[0,2] = cam_infos[idx].width / 2  ##
                    intrinsics[1,2] = cam_infos[idx].height / 2
                    intrinsics[2,2] = 1.0
                    
                    
                    # rgb_image = imread_cv2(impath)
                    # label_path = os.path.join(self.cfg.roots[0], 'label', scene_id, f'frame_{(target_view+1):05d}.json')
                    if scene_id == 'office3':
                        label_path = os.path.join('/music-3d-shared-disk/user/KAIST/HG/replica', scene_id, 'Sequence_1/semantic_class', f'semantic_class_{image_name}.png')
                    else:
                        label_path = os.path.join('/music-3d-shared-disk/user/KAIST/HG/replica', scene_id, scene_id, 'Sequence_1/semantic_class', f'semantic_class_{image_name}.png')
                        
                    
                    labelmap = imread_cv2(label_path, options=cv2.IMREAD_UNCHANGED)
                    label_map = cv2.resize(labelmap, self.cfg.original_image_shape, interpolation=cv2.INTER_NEAREST)
                    
                    ## remap the labels following self.mapping
                    temp_label_map = label_map.copy()
                    for k, v in self.mapping[scene_id].items():
                        label_map[temp_label_map == k] = v
                    
                    for unique_label in np.unique(temp_label_map):
                        if unique_label not in self.mapping[scene_id]:
                            label_map[temp_label_map == unique_label] = 0
                    
                    ## resize image
                    original_images_list.append(self.to_tensor(rgb_image))
                    
                    rgb_image = rgb_image.resize(self.cfg.input_image_shape, Image.LANCZOS)
                    # rgb_image = cv2.resize(rgb_image, self.cfg.input_image_shape, interpolation=cv2.INTER_CUBIC)

                    intrinsics[0,:] /= cam_infos[idx].width
                    intrinsics[1,:] /= cam_infos[idx].height
                    
                    focal_medium = (intrinsics[0,0] + intrinsics[1,1]) / 2

                    intrinsics[0,0] = focal_medium*1.5
                    intrinsics[1,1] = focal_medium*1.5

                    extrinsics_list.append(camera_pose)
                    intrinsics_list.append(intrinsics)
                    images_list.append(self.to_tensor(rgb_image))
                    label_list.append(torch.from_numpy(label_map).long().unsqueeze(0))
                    
                extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0).astype(np.float32))
                intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0).astype(np.float32))
                images = torch.stack(images_list, dim=0)
                original_images = torch.stack(original_images_list, dim=0)
                labels = torch.cat(label_list, dim=0)
                    
                context_extrinsics = extrinsics[:2]
                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1    
                    
                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[0:1], extrinsics)
                    
                    
                if self.cfg.context_eval:
                    # example = {
                    #     "context": {
                    #         "extrinsics": torch.cat([extrinsics[2:3], extrinsics[0:1]], dim=0),
                    #         "intrinsics": torch.cat([intrinsics[2:3], intrinsics[0:1]], dim=0),
                    #         "image": torch.cat([images[2:3], images[0:1]], dim=0),
                    #         "original_image": torch.cat([original_images[2:3], original_images[0:1]], dim=0),
                    #         "label": torch.cat([labels[2:3], labels[0:1]], dim=0),
                    #         "near": self.get_bound("near", len(idxs[1:])) / scale,
                    #         "far": self.get_bound("far", len(idxs[1:])) / scale,
                    #         "index": idxs[1:],
                    #         'text': self.labels[scene_id],
                    #         "overlap": 0,
                    #     },
                    #     "target": {
                    #         "extrinsics": extrinsics[2:],
                    #         "intrinsics": intrinsics[2:],
                    #         "image": images[2:],
                    #         "original_image": original_images[2:],
                    #         "label": labels[2:],
                    #         "near": self.get_bound("near", len(idxs[2:])) / scale,
                    #         "far": self.get_bound("far", len(idxs[2:])) / scale,
                    #         "index": idxs[2:],
                    #         'text': self.labels[scene_id],
                    #     },
                    #     "scene": scene_id,
                    # }

                    # yield example
                    # for n in range(2):
                    #     example = {
                    #         "context": {
                    #             "extrinsics": extrinsics[:2],
                    #             "intrinsics": intrinsics[:2],
                    #             "image": images[:2],
                    #             "original_image": original_images[:2],
                    #             "label": labels[:2],
                    #             "near": self.get_bound("near", len(idxs[:2])) / scale,
                    #             "far": self.get_bound("far", len(idxs[:2])) / scale,
                    #             "index": idxs[:2],
                    #             'text': self.labels[scene_id],
                    #             "overlap": 0,
                    #         },
                    #         "target": {
                    #             "extrinsics": extrinsics[n:n+1],
                    #             "intrinsics": intrinsics[n:n+1],
                    #             "image": images[n:n+1],
                    #             "original_image": original_images[n:n+1],
                    #             "label": labels[n:n+1],
                    #             "near": self.get_bound("near", len(idxs[n:n+1])) / scale,
                    #             "far": self.get_bound("far", len(idxs[n:n+1])) / scale,
                    #             "index": idxs[n:n+1],
                    #             'text': self.labels[scene_id],
                    #         },
                    #         "scene": scene_id,
                    #     }

                    #     yield example
                    example = {
                        "context": {
                            "extrinsics": extrinsics[:2],
                            "intrinsics": intrinsics[:2],
                            "image": images[:2],
                            "original_image": original_images[:2],
                            "label": labels[:2],
                            "near": self.get_bound("near", len(idxs[:2])) / scale,
                            "far": self.get_bound("far", len(idxs[:2])) / scale,
                            "index": idxs[:2],
                            'text': self.labels[scene_id],
                            "overlap": 0,
                        },
                        "target": {
                            "extrinsics": extrinsics[0:1],
                            "intrinsics": intrinsics[0:1],
                            "image": images[0:1],
                            "original_image": original_images[0:1],
                            "label": labels[0:1],
                            "near": self.get_bound("near", len(idxs[0:1])) / scale,
                            "far": self.get_bound("far", len(idxs[0:1])) / scale,
                            "index": idxs[0:1],
                            'text': self.labels[scene_id],
                        },
                        "scene": scene_id,
                    }

                    yield example
                    
                    example = {
                        "context": {
                            "extrinsics": torch.cat([extrinsics[1:2], extrinsics[0:1]], dim=0),
                            "intrinsics": torch.cat([intrinsics[1:2], intrinsics[0:1]], dim=0),
                            "image": torch.cat([images[1:2], images[0:1]], dim=0),
                            "original_image": torch.cat([original_images[1:2], original_images[0:1]], dim=0),
                            "label": torch.cat([labels[1:2], labels[0:1]], dim=0),
                            "near": self.get_bound("near", len(idxs[:2])) / scale,
                            "far": self.get_bound("far", len(idxs[:2])) / scale,
                            "index": idxs[:2],
                            'text': self.labels[scene_id],
                            "overlap": 0,
                        },
                        "target": {
                            "extrinsics": extrinsics[1:2],
                            "intrinsics": intrinsics[1:2],
                            "image": images[1:2],
                            "original_image": original_images[1:2],
                            "label": labels[1:2],
                            "near": self.get_bound("near", len(idxs[0:1])) / scale,
                            "far": self.get_bound("far", len(idxs[0:1])) / scale,
                            "index": idxs[1:2],
                            'text': self.labels[scene_id],
                        },
                        "scene": scene_id,
                    }

                    yield example
                    
                    
                else:
                    example = {
                        "context": {
                            "extrinsics": extrinsics[:2],
                            "intrinsics": intrinsics[:2],
                            "image": images[:2],
                            "original_image": original_images[:2],
                            "label": labels[:2],
                            "near": self.get_bound("near", len(idxs[:2])) / scale,
                            "far": self.get_bound("far", len(idxs[:2])) / scale,
                            "index": idxs[:2],
                            'text': self.labels[scene_id],
                            "overlap": 0,
                        },
                        "target": {
                            "extrinsics": extrinsics[2:],
                            "intrinsics": intrinsics[2:],
                            "image": images[2:],
                            "original_image": original_images[2:],
                            "label": labels[2:],
                            "near": self.get_bound("near", len(idxs[2:])) / scale,
                            "far": self.get_bound("far", len(idxs[2:])) / scale,
                            "index": idxs[2:],
                            'text': self.labels[scene_id],
                        },
                        "scene": scene_id,
                    }

                    yield example
                
    def stack_mask(self, mask_base, mask_add):
        mask = mask_base.copy()
        mask[mask_add != 0] = 1
        return mask
                
    def polygon_to_mask(self, img_shape, points_list):
        points = np.asarray(points_list, dtype=np.int32)
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return mask

    def read_lerf_annotation(self, js_path):
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = self.polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = self.stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            # save_path = output_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            # save_path.parent.mkdir(exist_ok=True, parents=True)
            # vis_mask_save(mask, save_path)
        
        return img_ann


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




