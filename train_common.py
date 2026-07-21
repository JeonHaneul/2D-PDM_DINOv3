import os
import random

import cv2
import numpy as np
import torch

from target_utils import bgr_to_tensor, crop_with_mask, discover_target_frame_id, load_target_reference, masked_pool, prepare_target_input

DATA_ROOT_260714 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "260714_data")


def discover_scene_ids(scene_dir: str) -> list:
    """SCENE_DIR/seg/ 안의 'sceneNNNNN_mapping.json' 파일들에서 실제 존재하는 scene 번호를
    전부 찾아 정렬된 리스트로 반환. 하드코딩 없이 target 폴더 상태를 그대로 반영한다."""
    seg_dir = os.path.join(scene_dir, "seg")
    ids = set()
    for f in os.listdir(seg_dir):
        if f.startswith("scene") and f.endswith("_mapping.json"):
            ids.add(int(f[len("scene"):len("scene") + 5]))
    return sorted(ids)


def split_scene_ids(scene_ids: list, train_ratio: float, seed: int | None) -> tuple:
    """scene 번호 리스트를 그룹째로 셔플한 뒤 train_ratio 비율로 나눔 (개별 이미지가 아니라
    scene 단위로 나눠야 같은 배치의 카메라 5장이 train/val에 걸쳐 섞이는 leakage를 막을 수 있음).
    seed에 정수를 주면 항상 같은 분할(고정), None을 주면 실행할 때마다 다른 무작위 분할이 된다
    -- 이 경우에도 나중에 재현할 수 있도록 실제 사용된 시드값을 반환한다."""
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)  # OS 엔트로피 기반 -- 매 실행 진짜 무작위
    shuffled = list(scene_ids)
    random.Random(seed).shuffle(shuffled)
    n_train = round(len(shuffled) * train_ratio)
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:])
    return train_ids, val_ids, seed


def encode_target(backbone, device, target_rgb_dir, target_seg_dir, target_mapping_path,
                   target_frame_id, cam):
    """target 사진 한 장(frame_id + cam 하나로 특정됨)을 crop+마스크풀링해서 DINO layer별
    벡터 리스트로 인코딩. 결정적(같은 인자면 항상 같은 결과) -- 여러 target x cam 조합을
    미리 전부 인코딩해서 캐싱해둘 때(precompute_target_vec_cache) 쓰인다."""
    tgt_rgb, tgt_mask, _ = load_target_reference(target_rgb_dir, target_seg_dir, target_mapping_path,
                                                  target_frame_id, cam)
    crop_rgb, crop_mask, _ = crop_with_mask(tgt_rgb, tgt_mask, pad_ratio=0.25)
    target_input_bgr, target_input_mask = prepare_target_input(crop_rgb, crop_mask, size=224)
    target_tensor = bgr_to_tensor(target_input_bgr, device=device)
    target_feats = backbone(target_tensor)  # DINOv3(frozen)로 target 인코딩 -- 여기는 학습 대상 아님
    return [masked_pool(patch[0], target_input_mask)[0] for patch, _cls in target_feats]  # [(C,), ...] per layer


def sample_target_vecs(backbone, device, target_rgb_dir, target_seg_dir, target_mapping_path,
                        target_frame_id, target_cams):
    """Random cam angle each call -- cheap viewpoint augmentation.
    260714_data는 target당 프레임이 딱 1개(frame_id 고정)라서 예전(260708_data, 프레임 44100개
    -> 크기 0.7~1.3배 augmentation)과 달리 랜덤 frame_id를 고를 수 없다. 대신 그 1개 프레임의
    카메라 5장(center/top/left/right/bottom) 중에서만 무작위로 골라 약한 augmentation 효과를 준다."""
    cam = random.choice(target_cams)
    return encode_target(backbone, device, target_rgb_dir, target_seg_dir, target_mapping_path,
                          target_frame_id, cam)


def target_paths(name: str) -> dict:
    """target 이름 하나에 대한 모든 관련 경로를 한번에 계산. v1(디스크 스트리밍)과 v2(VRAM
    preload) 둘 다 여러 target을 섞어 학습하므로 이 경로 계산 로직을 공유한다."""
    scene_dir = os.path.join(DATA_ROOT_260714, "scene", name)
    target_dir = os.path.join(DATA_ROOT_260714, "target", name)
    return {
        "scene_dir": scene_dir,
        "gt_dir": os.path.join(DATA_ROOT_260714, "GT_data", name),
        "target_rgb_dir": os.path.join(target_dir, "rgb"),
        "target_seg_dir": os.path.join(target_dir, "seg"),
        "target_mapping_path": os.path.join(target_dir, "mapping.json"),
    }


def precompute_target_vec_cache(backbone, targets: list, device: str, target_cams: list) -> tuple:
    """모든 target x 카메라 조합을 미리 인코딩해서 {(target_name, cam): [layer별 (C,) 벡터]}
    캐시로 만든다. 이후 학습 루프는 이 캐시에서 조회만 하므로 DINOv3 target 인코딩이 학습 중엔
    한 번도 다시 일어나지 않는다. target/mapping.json이 없는 target은 건너뛰고 알려준다."""
    cache = {}
    usable, skipped = [], []
    for name in targets:
        paths = target_paths(name)
        if not os.path.isfile(paths["target_mapping_path"]):
            skipped.append(name)
            continue
        frame_id = discover_target_frame_id(paths["target_rgb_dir"])
        for cam in target_cams:
            cache[(name, cam)] = encode_target(
                backbone, device, paths["target_rgb_dir"], paths["target_seg_dir"],
                paths["target_mapping_path"], frame_id, cam
            )
        usable.append(name)
    return cache, usable, skipped


def gather_target_vecs(target_vec_cache: dict, target_names: list, num_layers: int, device: str,
                        target_cams: list, cam: str | None = None) -> list:
    """배치 안 각 샘플이 가리키는 target에 맞는 벡터를 캐시에서 조회해서 (B,C) 텐서로 쌓는다.
    cam=None(학습용): 카메라를 샘플마다 무작위로 골라서 가벼운 viewpoint augmentation.
    cam='center' 등 특정 값(검증용): 그 카메라로 고정 -- 검증 때 무작위로 고르면 val_loss가 모델
    성능과 무관하게 "이번 epoch엔 어떤 각도가 뽑혔나"로 흔들려서 EarlyStopping이 오작동한다."""
    if cam is not None:
        per_sample = [target_vec_cache[(name, cam)] for name in target_names]
    else:
        per_sample = [target_vec_cache[(name, random.choice(target_cams))] for name in target_names]
    return [torch.stack([vecs[li] for vecs in per_sample], dim=0).to(device) for li in range(num_layers)]


class EarlyStopping:
    """val_loss가 이전 최고 기록보다 최소 min_delta_pct(비율)만큼 줄어들 때만 "개선"으로 인정.
    260707_code/FCN_train/train_250506.py의 EarlyStopping은 절대값 min_delta(예: 0.001)를
    썼는데, 여기 similarity 모델의 MSE loss 자체가 0.0001~0.001 스케일이라 절대값 threshold는
    무의미하다(loss 전체 범위보다 threshold가 더 큼) -- 그래서 상대적(%) 기준으로 구현."""

    def __init__(self, patience: int = 10, min_delta_pct: float = 0.01, verbose: bool = True):
        self.patience = patience
        self.min_delta_pct = min_delta_pct  # 0.01 = 1% 이상 줄어들어야 개선으로 인정
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """이번 val_loss가 새로운 최고 기록이면 True 반환 (best 체크포인트 저장 트리거용)."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return True

        threshold = self.best_loss * (1 - self.min_delta_pct)
        if val_loss < threshold:
            if self.verbose:
                pct = (self.best_loss - val_loss) / self.best_loss * 100
                print(f"    val_loss 개선: {self.best_loss:.6f} -> {val_loss:.6f} ({pct:.2f}% 감소)")
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.verbose:
            print(f"    EarlyStopping: {self.counter}/{self.patience} epoch 연속 "
                  f"{self.min_delta_pct * 100:.1f}% 이상 개선 없음 (best={self.best_loss:.6f})")
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def batch_accuracy_counts(pred01: torch.Tensor, gt01: torch.Tensor,
                           pos_thresh_255: float = 25.5, diff_thresh_255: float = 13.0) -> tuple:
    """260707_code/FCN_train/train_250506.py의 label_accuracy_values()를 그대로 이식했었는데,
    원래 diff_thresh=51(0~1 스케일로 ±0.2)은 우리 모델의 실제 달성 가능한 정밀도(val_mse가
    0.0001~0.001까지 내려감 -> RMSE로 0.01~0.03)에 비해 너무 헐렁해서, 학습이 어느 정도만
    잘 돼도 거의 모든 픽셀이 threshold 안에 들어와 accuracy가 99~100%로 바로 포화되고 더 이상
    모델 품질을 구분 못 하는 문제가 있었다(실측: RMSE 0.02짜리 예측도 100% 나옴). 그래서
    diff_thresh_255=13(±0.05)로 좁혀서 실제로 모델 간 차이가 변별력 있게 나오도록 했다.
    pred01/gt01은 [0,1] 범위 텐서(아무 shape나 가능, 그냥 전부 펼쳐서 픽셀 단위로 계산) -- 0~255
    스케일로 되돌려서 threshold를 적용한다(FCN 코드와 스케일만 맞춘 것, 값 자체는 재보정함).
    한 배치의 raw count(true_pos 등)만 반환 -- epoch 전체 정확도는 이 count들을 누적한 뒤
    accuracy_scores_from_counts()로 한 번에 계산해야 한다 (배치마다 accuracy를 구해서 단순
    평균내면 배치별로 분모가 달라 틀린 값이 나옴)."""
    lt = gt01.detach() * 255.0
    lp = pred01.detach() * 255.0
    close = (lp - lt).abs() < diff_thresh_255
    is_pos = lt > pos_thresh_255
    true_pos = (close & is_pos).sum().item()
    true_neg = (close & ~is_pos).sum().item()
    num_true_pos = is_pos.sum().item()
    num_union_pos = (is_pos | (lp > pos_thresh_255)).sum().item()
    total = lt.numel()
    return true_pos, true_neg, num_true_pos, num_union_pos, total


def accuracy_scores_from_counts(true_pos: float, true_neg: float, num_true_pos: float,
                                 num_union_pos: float, total: float) -> tuple:
    """batch_accuracy_counts()로 여러 배치에 걸쳐 누적한 raw count로부터 최종 지표 계산.
    acc: 전체 픽셀 중 (임계값 내로) 맞춘 비율
    bal_acc: 양성/음성 클래스 각각의 recall을 평균 (클래스 불균형에 덜 민감 -- GT의 대부분이
             배경(0)인 similarity map 특성상 그냥 acc만 보면 "다 0이라고만 해도 높게" 나올 수 있어서
             bal_acc가 더 의미 있는 지표)
    iou: true_pos / (positive로 예측했거나 실제 positive인 픽셀의 합집합)"""
    num_true_neg = total - num_true_pos
    acc = (true_pos + true_neg) / total if total > 0 else 0.0
    pos_recall = true_pos / num_true_pos if num_true_pos > 0 else 0.0
    neg_recall = true_neg / num_true_neg if num_true_neg > 0 else 0.0
    bal_acc = 0.5 * (pos_recall + neg_recall)
    iou = true_pos / num_union_pos if num_union_pos > 0 else 0.0
    return acc, bal_acc, iou


def append_log(log_path: str, text: str):
    with open(log_path, "a") as f:
        f.write(text + "\n")


def build_qualitative_panel(target_usd_name: str, crop_rgb: np.ndarray, query_rgb: np.ndarray,
                             gt_map: np.ndarray, pred_full: np.ndarray, val_frame_name: str,
                             extra_label: str = "") -> np.ndarray:
    """target crop | val scene | GT(grayscale) | 예측(grayscale) 4-패널 비교 이미지를 만든다.
    학습 스크립트가 주기적으로(또는 마지막에) 같은 val 샘플에 대해 호출해서 진행 상황을 시각적으로
    추적할 수 있게 한다."""
    H, W = query_rgb.shape[:2]

    def to_gray_bgr(m01):
        u8 = (np.clip(m01, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    def label(img, text, bar_h=36):
        bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (8, bar_h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([bar, img], axis=0)

    target_vis = cv2.resize(crop_rgb, (H, H))
    return np.concatenate(
        [
            label(target_vis, f"TARGET ({target_usd_name})"),
            label(query_rgb, f"VAL scene ({val_frame_name}, unseen)"),
            label(to_gray_bgr(gt_map), "GT similarity (grayscale)"),
            label(to_gray_bgr(pred_full), f"PREDICTION {extra_label}".strip()),
        ],
        axis=1,
    )
