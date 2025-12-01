import io
from pathlib import Path
from typing import Union

import numpy as np
import skvideo.io
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
import matplotlib as mpl
import matplotlib.cm as cm
import torchvision
import cv2

from .utils import inverse_normalize


FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()

def visualize_attention_map(
    qkv,
    batch,
    num_heads: int,
    gaussian_token_idx: int,
    image_size: tuple[int, int],
    patch_size,
    output_path
):
    qkv = qkv.chunk(3, dim=-1)
    query, key, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = num_heads), qkv)

    attn_score = torch.einsum('bhid,bhjd->bhij', query, key)  # (B, num_heads, N, N)
    attn_score = attn_score / (query.shape[-1] ** 0.5)
    
    PH, PW = image_size[0] // patch_size, image_size[1] // patch_size
    guidance_feature_size = PH * PW * 2
    
    attn_score = attn_score[:, :, guidance_feature_size + gaussian_token_idx, : guidance_feature_size]  # (B, num_heads, N)
    attn_score = attn_score.softmax(dim=-1).mean(dim=1)  # (B, N)
    image1_attn_score = attn_score[:,:PH*PW].reshape(1, PH, PW)
    image2_attn_score = attn_score[:,PH*PW:].reshape(1, PH, PW)
    
    
    resize = torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    image1_attn_score = resize(image1_attn_score.unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()
    image2_attn_score = resize(image2_attn_score.unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()
            
    image1_unnorm = (inverse_normalize(batch["context"]["image"][0, 0])).permute(1, 2, 0).cpu().detach().numpy()
    image1_unnorm = (image1_unnorm * 255).astype(np.uint8)
    
    image2_unnorm = (inverse_normalize(batch["context"]["image"][0, 1])).permute(1, 2, 0).cpu().detach().numpy()
    image2_unnorm = (image2_unnorm * 255).astype(np.uint8)

    normalizer = mpl.colors.Normalize(vmin=image1_attn_score.min(), vmax=image1_attn_score.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    colormapped_im = (mapper.to_rgba(image1_attn_score[:,:,0])[:, :, :3] * 255).astype(np.uint8)
    attn_map1 = cv2.addWeighted(image1_unnorm.copy(), 0.3, colormapped_im, 0.7, 0)
    attn_map1 = Image.fromarray(attn_map1)
    attn_map1.save(str(output_path) + "_image1_attn_map.png")
    
    normalizer = mpl.colors.Normalize(vmin=image2_attn_score.min(), vmax=image2_attn_score.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
    colormapped_im = (mapper.to_rgba(image2_attn_score[:,:,0])[:, :, :3] * 255).astype(np.uint8)
    attn_map2 = cv2.addWeighted(image2_unnorm.copy(), 0.3, colormapped_im, 0.7, 0)
    attn_map2 = Image.fromarray(attn_map2)
    attn_map2.save(str(output_path) + "_image2_attn_map.png")
    


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict={'-pix_fmt': 'yuv420p', '-crf': '21',
                                                 '-vf': f'setpts=1.*PTS'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()
