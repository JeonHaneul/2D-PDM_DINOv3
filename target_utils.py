"""Build a target reference crop+mask from a scene frame, and mask-pool DINOv3 patch features."""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backbone import PATCH_SIZE

CAM_NAMES = ["center", "top", "left", "right", "bottom"]


def load_scene_mapping(mapping_json_path: str) -> dict:
    """usd_name -> (b, g, r). Field is named 'color_rgb' in the JSON but is actually
    stored in BGR order (see similarity_map_generator.py), so it matches cv2.imread directly."""
    # scene의 seg 이미지에서 "이 색깔 = 이 물체(usd_name)"를 알려주는 색상 lookup table.
    # 필드명은 color_rgb지만 실제 저장된 값은 BGR 순서 (원본 파이프라인의 네이밍 버그를 그대로 승계).
    with open(mapping_json_path) as f:
        m = json.load(f)
    return {k: tuple(v["color_rgb"]) for k, v in m.items()}


def color_mask(bgr_img: np.ndarray, bgr_color) -> np.ndarray:
    # seg 이미지에서 특정 (b,g,r) 색상과 정확히 일치하는 픽셀만 True인 마스크를 만든다.
    # (Isaac Sim이 물체마다 고유 색을 칠해서 segmentation mask를 만드는 방식이라 단순 색상 비교로 충분함)
    b, g, r = bgr_color
    return (bgr_img[:, :, 0] == b) & (bgr_img[:, :, 1] == g) & (bgr_img[:, :, 2] == r)


def find_visible_target_frame(scene_dir: str, scene_prefix: str, target_bgr, max_envs: int = 100,
                               min_px: int = 300):
    """Scan seg frames for one where the target is clearly visible (not occluded/off-frame).
    Objects get buried under others in plenty of frames, so picking a fixed frame blindly is
    unreliable -- this returns the first frame crossing min_px, or the best one seen otherwise."""
    # scene 안에는 target이 다른 물체에 가려서 아예 안 보이는 프레임도 많다 (실제로 테스트 중
    # 첫 시도한 프레임에서 target 픽셀이 0개였음). 그래서 여러 env/cam 조합을 순서대로 훑으면서
    # target이 충분히(min_px 이상) 보이는 프레임을 찾아준다. -- 지금은 baseline_raw_dinov3.py 등
    # scene 안에서 target을 직접 잘라내야 하는 보조 스크립트에서만 쓰이고, 메인 학습 파이프라인은
    # 아래 load_target_reference()로 대체됨 (실제 target 단독 사진을 쓰기 때문에 이 탐색이 불필요).
    seg_dir = os.path.join(scene_dir, "seg")
    best_name, best_mask, best_count = None, None, -1
    for env in range(max_envs):
        for cam in CAM_NAMES:
            fname = f"{scene_prefix}_env{env:04d}_{cam}"
            path = os.path.join(seg_dir, f"{fname}.png")
            if not os.path.isfile(path):
                continue
            seg = cv2.imread(path)
            mask = color_mask(seg, target_bgr)
            count = int(mask.sum())
            if count > best_count:
                best_name, best_mask, best_count = fname, mask, count
            if count >= min_px:
                return fname, mask, count
    return best_name, best_mask, best_count


def discover_target_frame_id(target_rgb_dir: str) -> str:
    """260714_data/target/<name>/rgb/ 에는 target마다 프레임이 딱 1개(카메라 5장)만 있고,
    그 frame_id 숫자는 target마다 다르다 (예: book_1/2는 '007626', 나머지는 '000000').
    하드코딩하지 않고 실제 폴더 안 파일명에서 그대로 읽어온다."""
    files = sorted(f for f in os.listdir(target_rgb_dir) if f.endswith("_center.png"))
    if not files:
        raise FileNotFoundError(f"target rgb 프레임을 찾을 수 없음: {target_rgb_dir}")
    return files[0].split("_")[0]  # "007626_center.png" -> "007626"


def load_target_reference(target_rgb_dir: str, target_seg_dir: str, target_mapping_path: str,
                           frame_id: str = "000000", cam: str = "center"):
    """Load a real target-alone photo (same camera setup/distance as the scene, so the object's
    apparent size in-frame is physically meaningful) + its own mask, from the dedicated
    target/rgb + target/seg + target/mapping.json triplet -- NOT a crop pulled out of a scene
    frame and NOT an asset texture file. mapping.json here uses a different schema than the
    scene's per-scene mapping.json (`classes.<usd_name>.color_bgr`, correctly named this time)."""
    # target 전용 mapping.json 구조: {"target_usd_name": ..., "classes": {usd_name: {"color_bgr": [b,g,r]}}}
    # scene의 mapping.json과 필드 구조가 다르므로 따로 파싱한다.
    with open(target_mapping_path) as f:
        m = json.load(f)
    target_usd_name = m["target_usd_name"]
    bgr_color = tuple(m["classes"][target_usd_name]["color_bgr"])

    # 같은 frame_id/cam 조합으로 target 단독 RGB 사진과 그 seg 마스크를 같이 불러온다.
    # scene과 동일한 카메라 세팅(거리/화각)으로 찍힌 사진이라 물체 크기가 물리적으로 의미 있음.
    rgb = cv2.imread(os.path.join(target_rgb_dir, f"{frame_id}_{cam}.png"))
    seg = cv2.imread(os.path.join(target_seg_dir, f"{frame_id}_{cam}.png"))
    if rgb is None or seg is None:
        raise FileNotFoundError(f"missing target frame {frame_id}_{cam} under {target_rgb_dir} / {target_seg_dir}")
    mask = color_mask(seg, bgr_color)
    return rgb, mask, target_usd_name


def extract_target_geometry(mask: np.ndarray, silhouette_size: int = 8) -> np.ndarray:
    """Convert a target mask into the fixed geometry vector used by Occlusion FiLM.

    The vector contains normalized mask area, bounding-box height/width, log aspect ratio,
    and a coarse soft silhouette.  It is deterministic and therefore works for a new target
    without learning target-specific parameters.
    """
    if mask.ndim != 2:
        raise ValueError(f"target mask must be 2-D, got shape {mask.shape}")
    if silhouette_size <= 0:
        raise ValueError("silhouette_size must be positive")

    mask_bool = np.asarray(mask, dtype=bool)
    height, width = mask_bool.shape
    ys, xs = np.where(mask_bool)
    output_dim = 4 + silhouette_size * silhouette_size
    if len(ys) == 0:
        return np.zeros(output_dim, dtype=np.float32)

    area_ratio = float(np.count_nonzero(mask_bool)) / float(mask_bool.size)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bbox_h = int(y1 - y0 + 1)
    bbox_w = int(x1 - x0 + 1)
    bbox_h_ratio = bbox_h / height
    bbox_w_ratio = bbox_w / width
    log_aspect = float(np.log(bbox_w / bbox_h))
    crop = mask_bool[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    silhouette = cv2.resize(
        crop,
        (silhouette_size, silhouette_size),
        interpolation=cv2.INTER_AREA,
    )
    return np.concatenate(
        [
            np.asarray(
                [area_ratio, bbox_h_ratio, bbox_w_ratio, log_aspect],
                dtype=np.float32,
            ),
            silhouette.reshape(-1).astype(np.float32),
        ]
    )


def crop_with_mask(rgb_img: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.25):
    """Tight bbox around `mask`, padded by pad_ratio, clipped to image bounds."""
    # target 단독 사진은 640x480 프레임 안에 물체가 아주 작게(수십 px) 찍혀있어서, 그대로
    # DINOv3에 넣으면 대부분의 patch가 배경(검은색)이라 정보가 희석된다. 그래서 마스크의
    # bounding box를 25% 여백을 두고 타이트하게 잘라내서 물체가 프레임을 꽉 채우게 만든다.
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    pad_y, pad_x = int(h * pad_ratio), int(w * pad_ratio)
    H, W = rgb_img.shape[:2]
    y0, y1 = max(0, y0 - pad_y), min(H - 1, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(W - 1, x1 + pad_x)
    return rgb_img[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1], (y0, y1, x0, x1)


def prepare_target_input(crop_bgr: np.ndarray, crop_mask: np.ndarray, size: int = 224):
    """Resize crop + mask to a canonical square (multiple of patch size) for the target encoder pass."""
    # DINOv3에 넣기 좋은 정사각형(224=16의 배수)으로 리사이즈. RGB는 부드럽게 보간(INTER_LINEAR),
    # 마스크는 이진값이 뭉개지면 안 되므로 최근접 보간(INTER_NEAREST)을 따로 사용.
    resized_bgr = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return resized_bgr, resized_mask


# DINOv3(및 대부분의 ImageNet 사전학습 모델)가 학습 때 쓴 정규화 통계값.
# 입력 이미지를 이 값으로 정규화해야 backbone이 원래 학습된 분포와 맞는 입력을 받게 됨.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bgr_to_chw(bgr_img: np.ndarray) -> torch.Tensor:
    """HWC BGR uint8 -> (3,H,W) float RGB CPU tensor, ImageNet-normalized. No batch dim, no
    device move -- meant for use inside a Dataset.__getitem__ where the DataLoader's default
    collate stacks these into a batch."""
    # cv2는 BGR로 읽으므로 RGB로 변환 -> [0,1] 스케일 -> ImageNet 정규화 -> (C,H,W) 축 순서로 변경.
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


def bgr_to_tensor(bgr_img: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """HWC BGR uint8 -> (1,3,H,W) float RGB tensor, ImageNet-normalized (DINOv3 convention)."""
    # 배치 차원(1)을 추가하고 GPU로 올린 버전 -- 단발성 추론(테스트 스크립트)에서 주로 사용.
    return bgr_to_chw(bgr_img).unsqueeze(0).to(device)


def masked_pool(patch_grid: torch.Tensor, pixel_mask: np.ndarray, min_coverage: float = 1e-6):
    """patch_grid: (C,H',W') single-sample. pixel_mask: (H,W) bool at the crop's pixel resolution
    (H,W must equal H'*patch_size, W'*patch_size). Returns L2-normalized pooled vector (C,) and
    the per-patch weight map (H',W') used, for debugging/visualization."""
    # === Target을 "patch 여러 개"에서 "벡터 하나"로 요약하는 핵심 함수 ===
    # DINOv3는 target 이미지도 scene처럼 patch 단위 grid로 특징을 뽑아주는데, target은 물체
    # "하나"를 대표하는 단일 벡터가 필요하므로 여기서 patch들을 평균 내어 하나로 합친다.
    C, Hp, Wp = patch_grid.shape

    # 픽셀 단위 마스크(H,W)를 patch 해상도(H',W')로 다운샘플링.
    # avg_pool2d를 쓰는 이유: 이진(0/1) 마스크를 16x16 커널로 평균 내면 각 patch가
    # "물체 픽셀을 몇 %나 포함하는지"가 자동으로 나온다 (예: 완전히 덮이면 1.0, 절반 걸치면 0.5).
    # 이걸 그대로 가중치로 쓰면 물체 경계에 걸친 patch도 비율만큼 부드럽게(soft) 반영되어,
    # 이진 threshold(포함/제외)로 자르는 것보다 경계 정보 손실이 적다.
    m = torch.from_numpy(pixel_mask.astype(np.float32))[None, None].to(patch_grid.device)
    weight = F.avg_pool2d(m, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)[0, 0]  # (Hp,Wp), soft coverage ratio

    if weight.sum() < min_coverage:
        # 물체가 너무 작아서(또는 얇아서) 다운샘플링 후 마스크가 완전히 사라지는 극단적 경우 ->
        # 전체 patch를 균등 가중치로 풀링 (에러 대신 안전한 fallback)
        weight = torch.ones_like(weight)
    weight = weight / weight.sum()  # 가중치 합이 1이 되도록 정규화 (weighted average를 위해)

    # 가중 평균: 물체를 많이 포함한 patch일수록 최종 벡터에 더 크게 기여
    pooled = (patch_grid * weight[None]).sum(dim=(1, 2))
    # L2 정규화 -- 이후 cosine similarity 계산 시 벡터 크기(scale)에 영향받지 않도록,
    # 그리고 학습 시 head에 들어가는 입력 스케일을 일정하게 유지하기 위함.
    pooled = F.normalize(pooled, dim=0)
    return pooled, weight
