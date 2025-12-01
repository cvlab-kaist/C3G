import torch
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn.functional as F
import warnings

def save_segmap(gaussian_upfeature,
                seg: torch.Tensor, 
                index, 
                save_dir: Path,
                labels,
                color_hex_list):
    plt.imshow(seg.cpu().numpy(),cmap=ListedColormap(color_hex_list),vmin=0, vmax=len(labels))
    plt.axis('off')
    plt.savefig(save_dir / f"{index:0>6}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    seg = Image.open(save_dir / f"{index:0>6}.png").convert("RGB")
    seg = seg.resize((gaussian_upfeature.shape[-1], gaussian_upfeature.shape[-2]), resample=Image.NEAREST)
    seg = pil_to_tensor(seg).float() / 255.0
    
    return seg

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='firebrick', marker='o',
               s=marker_size, edgecolor='black', linewidth=2.5, alpha=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o',
               s=marker_size, edgecolor='black', linewidth=1.5, alpha=1)   


def run_pca(feature, img_size):
    # pca = PCA(n_components=3)
    try:
        B, C1, FH, FW = feature.shape
        H,W = img_size
        feature = feature.permute(0,2,3,1)
        
        feature_flat = feature.reshape(-1,C1).cpu().numpy()  # (H*W, C)
        # pca_result = pca.fit_transform(feature_flat)  # (H*W, 3)
        pca_result, _, _, _ = pca_torch(feature.reshape(-1,C1), n_components=3)  # (H*W, 3)
        
        pca_img = pca_result.reshape(B, FH, FW, -1).permute(0,3,1,2)  # (3, H//14, W//14)
        
        pca_img = F.interpolate(pca_img, size=(H, W), mode='bilinear', align_corners=False)
        pca_rgb = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min() + 1e-5)
        
    except:
        B, C1, FH, FW = feature.shape
        H,W = img_size
        pca_rgb = torch.zeros((B, 3, H, W), device=feature.device)
    return pca_rgb
    

def pca_torch(X, n_components):
    """
    X: torch.Tensor, shape (N, D), float32/float64
       GPU에서 돌리려면 X.cuda() 해서 넘기면 됨.
    n_components: int, 사용할 PCA 차원 수
    """
    # 1. 평균 빼기 (centering)
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # 2. SVD (X = U S Vh)
    # full_matrices=False 로 메모리 절약
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=".*failed to converge.*",   # convergence 관련 경고만 에러로 변환
                category=UserWarning
            )

            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    except Warning as e:
        print("❌ SVD convergence failed, aborting PCA.")
        raise RuntimeError("SVD convergence failed") from e

    # 3. 상위 n_components 고르기
    components = Vh[:n_components]          # (n_components, D)
    explained_variance = (S**2) / (X.shape[0] - 1)
    explained_variance = explained_variance[:n_components]

    # 4. 투영된 결과 (N, n_components)
    X_pca = X_centered @ components.T

    return X_pca, components, X_mean, explained_variance
