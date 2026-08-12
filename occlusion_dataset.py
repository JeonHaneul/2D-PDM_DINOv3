"""Occlusion 스트림 학습용 Dataset. train_similarity_v2.py의 MultiTargetSceneDataset과 같은
디스크 스트리밍 패턴(RAM 캐싱 없음)을 따르되, 데이터 소스가 다르다:

  - scene_rgb/scene_depth : scene_generator/output/<clutter_target>/scene/{rgb,depth}/
    (260714_data/scene이 아님 -- generate_occlusion_gt_batched_v2.py가 GT를 계산할 때 실제로
    읽은 clutter 소스가 여기이므로, GT와 scene 입력이 반드시 이 경로 기준으로 짝이 맞아야 함)
  - corrected probability GT : 2D-PDM_DINOv3/occlusion_gt_output_batched_v2/<target>/scale_<scale>/
    <scene_key>_<cam>/map_corrected.npy (legacy map은 비교/문서화용으로만 남기고 학습엔 안 씀)
  - target 참조 사진(target_vecs 계산용) : 260714_data/target/<target>/ (target_paths()로 조회,
    scene 소스와는 무관하게 항상 여기서 가져옴 -- target 단독 사진은 캡처 배치와 무관한 고정 자산)
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")
CLUTTER_ROOT = os.path.join(SRC_DIR, "scene_generator", "output")
GT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occlusion_gt_output_batched_v2")

CAMS = ["center", "top", "left", "right", "bottom"]

# 고정(scene별 min-max 아님) depth normalization 범위. empty_scene(배경 벽) depth 실측이
# 5개 카메라 전부 2.70~3.40 사이(260714_data/target/empty_scene/depth/*.npy로 확인)라
# 여유를 두고 [2.5, 3.5]로 고정 -- drawer 카메라 세팅이 바뀌지 않는 한 재측정 불필요.
DEPTH_MIN, DEPTH_MAX = 2.5, 3.5


def normalize_depth(depth_raw: np.ndarray) -> np.ndarray:
    """depth_raw(0=무효 반환, 그 외=실측값) -> (2,H,W) float32: 채널0=고정범위 정규화 depth
    (무효 픽셀은 0), 채널1=valid mask(1=유효, 0=무효). scene별 min-max normalization을 쓰지
    않는 이유: 카메라-drawer 거리가 고정 세팅이라 scene마다 다시 정규화할 이유가 없고,
    scene별 정규화를 쓰면 오히려 "이 scene의 최댓값"이라는 scene-dependent 정보가 새어들어감."""
    valid = depth_raw > 0
    norm = np.clip((depth_raw - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN), 0.0, 1.0)
    norm = np.where(valid, norm, 0.0).astype(np.float32)
    return np.stack([norm, valid.astype(np.float32)], axis=0)


def load_coverage_mask(target_name: str, scale: float, cam: str) -> np.ndarray:
    """N_all_corrected(target mesh가 44,100-pose grid 전체에서 이 픽셀을 몇 번이나 '도달
    가능'했는지 카운트, generate_occlusion_gt_batched_v2.py의 _coverage/에 저장됨) > 0 인
    픽셀만 학습에 쓴다. N_all_corrected==0인 픽셀은 "이 target이 어떤 pose에서도 이 위치에
    존재할 수 없음"을 뜻하므로 corrected GT가 항상 정확히 0이지만(코드 invariant로 보장됨),
    이건 "여기 있었으면 가려졌을 것"이 아니라 "애초에 여기 있을 수 없음"이라 의미가 다르다.
    coverage가 없는 픽셀을 일반 negative로 학습시키면 두 의미를 섞어버리게 되므로 loss/metric
    계산에서 제외해야 한다 -- scene과 무관하게 (target,scale,cam)에만 의존하므로 scene마다
    다시 읽지 않고 호출부에서 캐싱해 재사용하면 된다."""
    path = os.path.join(GT_ROOT, target_name, f"scale_{scale}", "_coverage", f"N_all_corrected_{cam}.npy")
    return (np.load(path) > 0).astype(np.float32)


class OcclusionDataset(Dataset):
    """target_scale_scenes: [(target_name, clutter_target, scale, [scene_key, ...]), ...]
    scene_key 리스트를 미리 걸러서(scene-held-out split) 넘기면 그 부분집합만 인덱싱된다.
    target_name 자체를 아예 통째로 넣거나 빼면 target-held-out split이 된다(별도 유틸 불필요,
    호출부에서 리스트 필터링만 하면 됨)."""

    def __init__(self, target_scale_scenes: list):
        self.samples = []  # (target_name, clutter_target, scale, scene_key, cam)
        self._coverage_cache = {}  # (target_name, scale, cam) -> (H,W) float32, scene 무관이라 1회만 로드
        skipped = 0
        for target_name, clutter_target, scale, scene_keys in target_scale_scenes:
            gt_root = os.path.join(GT_ROOT, target_name, f"scale_{scale}")
            for cam in CAMS:
                key = (target_name, scale, cam)
                if key not in self._coverage_cache:
                    self._coverage_cache[key] = load_coverage_mask(target_name, scale, cam)
            for sk in scene_keys:
                for cam in CAMS:
                    gt_path = os.path.join(gt_root, f"{sk}_{cam}", "map_corrected.npy")
                    rgb_path = os.path.join(CLUTTER_ROOT, clutter_target, "scene", "rgb", f"{sk}_{cam}.png")
                    depth_path = os.path.join(CLUTTER_ROOT, clutter_target, "scene", "depth", f"{sk}_{cam}.npy")
                    if os.path.isfile(gt_path) and os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                        self.samples.append((target_name, clutter_target, scale, sk, cam))
                    else:
                        skipped += 1
        self.skipped = skipped  # 호출부가 원하면 "누락된 조합 N개" 경고에 사용

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx):
        target_name, clutter_target, scale, sk, cam = self.samples[idx]
        rgb = cv2.imread(os.path.join(CLUTTER_ROOT, clutter_target, "scene", "rgb", f"{sk}_{cam}.png"))
        depth_raw = np.load(os.path.join(CLUTTER_ROOT, clutter_target, "scene", "depth", f"{sk}_{cam}.npy")).squeeze()
        gt = np.load(os.path.join(GT_ROOT, target_name, f"scale_{scale}", f"{sk}_{cam}", "map_corrected.npy"))
        coverage = self._coverage_cache[(target_name, scale, cam)]
        return rgb, depth_raw, gt.astype(np.float32), coverage

    def __getitem__(self, idx):
        rgb, depth_raw, gt, coverage = self.load_sample(idx)
        target_name, _clutter_target, _scale, sk, cam = self.samples[idx]
        depth_2ch = normalize_depth(depth_raw)
        return (rgb, torch.from_numpy(depth_2ch), torch.from_numpy(gt), torch.from_numpy(coverage),
                target_name, cam, sk)


def build_category_balanced_sampler(dataset: OcclusionDataset, category_of: dict) -> WeightedRandomSampler:
    """category별 target/scene 수가 다르면(book 3개 vs toy 2개 등) 그냥 균등 shuffle을 쓰는 경우
    sample 수가 많은 category가 더 자주 뽑혀서, held-out 성능 차이가 "카테고리 자체의 어려움"이
    아니라 "학습 노출량 차이"로 설명될 위험이 있다. sample i의 weight를
    1/(n_categories * count(category(i)))로 주면 category별 총 확률질량이 1/n_categories로
    같아진다(각 category 안에서는 여전히 모든 sample이 균등)."""
    categories = sorted(set(category_of.values()))
    n_cat = len(categories)
    cat_counts = {c: 0 for c in categories}
    sample_cats = []
    for target_name, _clutter, _scale, _sk, _cam in dataset.samples:
        c = category_of[target_name]
        cat_counts[c] += 1
        sample_cats.append(c)
    weights = [1.0 / (n_cat * cat_counts[c]) for c in sample_cats]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
