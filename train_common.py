import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from paths_config import GT_DIR, SCENE_DIR, TARGET_DIR
from target_utils import bgr_to_tensor, crop_with_mask, discover_target_frame_id, load_target_reference, masked_pool, prepare_target_input


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
    scene_dir = os.path.join(SCENE_DIR, name)
    target_dir = os.path.join(TARGET_DIR, name)
    return {
        "scene_dir": scene_dir,
        "gt_dir": os.path.join(GT_DIR, name),
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


# === 카테고리 zero-shot 분류 (DINOv3 CLS 토큰 기반 최근접 프로토타입) ===
# 문제의식: 지금까지 target 벡터(encode_target/masked_pool)는 patch grid를 풀링한 "공간적 매칭용"
# 벡터라 scene 안에서 target과 비슷한 위치를 찾는 데는 쓰이지만, "이 target이 애초에 toy인지
# book인지"를 판단하는 의미론적 신호는 아니다. GT를 만들 때 카테고리는 asset 폴더 구조(사람이
# 미리 정해둔 값)에서만 나오고, 학습된 모델 자체는 새로운(asset 폴더에 없는) 물체가 오면 그 물체의
# 카테고리를 스스로 추론할 방법이 없다. 아래는 그 문제를 DINOv3만으로(추가 대규모 모델 없이)
# 푸는 가장 가벼운 방법: CLS 토큰(패치 grid가 아니라 이미지 전체를 요약하는 벡터, 물체의 "종류"를
# patch보다 더 잘 담고 있음)을 카테고리별로 평균 내서 프로토타입을 만들고, 새 물체는 그 프로토타입
# 중 가장 가까운 것으로 분류한다 (few-shot/prototypical network의 zero-shot 버전 -- 새 물체
# 자체에 대한 학습은 전혀 필요 없고, 인코딩 1번 + 코사인 유사도 비교만 하면 됨).

def encode_target_cls(backbone, device, target_rgb_dir, target_seg_dir, target_mapping_path,
                       target_frame_id, cam, layer_idx: int = -1) -> torch.Tensor:
    """target 사진 한 장을 crop해서 DINOv3에 넣고, patch grid가 아니라 CLS 토큰을 뽑아 L2
    정규화해서 반환. layer_idx=-1(요청한 layer들 중 가장 깊은 층)을 쓰는 이유: 얕은 층은
    색/질감 같은 저수준 특징이 강하고, 깊은 층일수록 더 추상화된(의미론적) 정보를 담기 때문에
    "이 물체가 무슨 종류인가" 판단에는 깊은 층 쪽이 유리하다."""
    tgt_rgb, tgt_mask, _ = load_target_reference(target_rgb_dir, target_seg_dir, target_mapping_path,
                                                  target_frame_id, cam)
    crop_rgb, crop_mask, _ = crop_with_mask(tgt_rgb, tgt_mask, pad_ratio=0.25)
    target_input_bgr, _target_input_mask = prepare_target_input(crop_rgb, crop_mask, size=224)
    target_tensor = bgr_to_tensor(target_input_bgr, device=device)
    target_feats = backbone(target_tensor)  # [(patch, cls), ...] per requested layer
    _patch, cls = target_feats[layer_idx]
    return F.normalize(cls[0], dim=0)  # (C,)


def precompute_target_cls_cache(backbone, targets: list, device: str, target_cams: list) -> tuple:
    """target마다 카메라 5장의 CLS 벡터를 평균 내서 target 하나당 벡터 하나로 캐싱.
    (patch 벡터 캐시(precompute_target_vec_cache)와는 별개 용도 -- 이건 카테고리 분류용, 그건
    scene 안에서의 공간적 매칭용.) 카메라를 평균 내는 이유: "이 물체가 어떤 종류인가"는 보는
    각도와 무관해야 하는 성질이라, 각도별 CLS 임베딩 편차를 평균으로 지운다."""
    cache = {}
    usable, skipped = [], []
    for name in targets:
        paths = target_paths(name)
        if not os.path.isfile(paths["target_mapping_path"]):
            skipped.append(name)
            continue
        frame_id = discover_target_frame_id(paths["target_rgb_dir"])
        vecs = [
            encode_target_cls(backbone, device, paths["target_rgb_dir"], paths["target_seg_dir"],
                               paths["target_mapping_path"], frame_id, cam)
            for cam in target_cams
        ]
        cache[name] = F.normalize(torch.stack(vecs).mean(dim=0), dim=0)
        usable.append(name)
    return cache, usable, skipped


def build_category_prototypes(target_cls_cache: dict, target_categories: dict) -> dict:
    """target별 CLS 벡터를 카테고리별로 묶어서 평균 낸 뒤 재정규화 -> {category: 프로토타입 벡터}.
    target_cls_cache: {target_name: (C,) 벡터} (precompute_target_cls_cache의 반환값 형식)
    target_categories: {target_name: category_str}"""
    by_cat = {}
    for name, vec in target_cls_cache.items():
        cat = target_categories[name]
        by_cat.setdefault(cat, []).append(vec)
    return {cat: F.normalize(torch.stack(vecs).mean(dim=0), dim=0) for cat, vecs in by_cat.items()}


def classify_category(query_vec: torch.Tensor, prototypes: dict) -> list:
    """query_vec과 각 카테고리 프로토타입의 코사인 유사도를 계산해 (category, score) 내림차순
    리스트로 반환. 둘 다 L2 정규화된 벡터이므로 코사인 유사도 = 내적."""
    scores = [(cat, (query_vec * proto).sum().item()) for cat, proto in prototypes.items()]
    return sorted(scores, key=lambda kv: kv[1], reverse=True)


# === 카테고리 확률을 SimilarityMapModel의 입력 feature로 반영 ===
# 여기까지는 카테고리 분류가 순수 "보조 신호"(추론 결과를 얼마나 믿을지 판단하는 용도)였는데,
# 이제 그 확률 분포 자체를 모델 입력에 concat해서 pixel 단위 예측에 직접 반영한다 -- 그러면
# 2D-PDM 결과 이미지에서 "이 target은 book 카테고리다"라는 정보가 하이라이트되는 영역 자체에
# 반영된다. (SimilarityMapModel.forward의 category_probs 인자, similarity_model.py 참고)
CATEGORY_ORDER = ["book", "toy", "fruit", "packaged_food"]  # 모델 입력 벡터의 채널 순서 고정용


def category_probs_from_scores(ranked: list, category_order: list = CATEGORY_ORDER,
                                temperature: float = 0.1) -> torch.Tensor:
    """classify_category()가 반환한 (category, cos유사도) 목록을 고정된 순서의 확률 분포로 변환.
    cos유사도 범위(대략 0.1~0.7)가 좁아서 그냥 softmax하면 거의 균등분포가 되므로, temperature로
    나눠서 확률 분포가 뚜렷하게 갈리게 만든다 (작을수록 더 확신에 찬 분포)."""
    score_by_cat = dict(ranked)
    logits = torch.tensor([score_by_cat[c] for c in category_order], dtype=torch.float32)
    return F.softmax(logits / temperature, dim=0)


def precompute_target_category_probs(target_cls_cache: dict, target_categories: dict,
                                      category_order: list = CATEGORY_ORDER,
                                      temperature: float = 0.1) -> dict:
    """target마다 leave-one-out 프로토타입(자기 자신은 제외한 나머지 target들로 만든 프로토타입)
    으로 카테고리 확률 분포를 계산해 {target_name: (K,) 확률벡터} 로 반환.

    자기 자신을 포함해서 프로토타입을 만들면 카테고리 확률이 항상 정답에 100% 확신을 갖고
    나오게 되어(카닝) 모델이 "카테고리 신호는 항상 완벽하다"고 잘못 학습한다. 실제 배포 때는
    지금까지 한 번도 못 본 새 물체가 들어오므로, 학습 때도 "이 target을 프로토타입 계산에서
    아예 빼고 나머지로만 분류했을 때 나올 법한, 가끔 애매하거나 틀리기도 하는" 확률 분포를
    그대로 학습 신호로 써야 head가 그런 불확실성에도 안정적으로 대응하는 법을 배운다.

    단, 어떤 카테고리에 instance가 단 하나뿐이면(예: 작은 subset으로 스모크 테스트할 때)
    그 하나를 빼는 순간 카테고리 자체가 프로토타입에서 통째로 사라져 분류가 불가능해지므로,
    그 경우에 한해 예외적으로 자기 자신을 포함해서 계산한다(경고 출력). 실제 15-target 전체
    사용 시(카테고리당 3~4개)는 이 예외가 발생하지 않는다."""
    cat_counts = {}
    for name in target_cls_cache:
        cat_counts[target_categories[name]] = cat_counts.get(target_categories[name], 0) + 1

    probs = {}
    for name in target_cls_cache:
        cat = target_categories[name]
        if cat_counts[cat] >= 2:
            remaining_cache = {n: v for n, v in target_cls_cache.items() if n != name}
        else:
            print(f"    [WARN] '{name}' ({cat}) 카테고리에 다른 instance가 없어 leave-one-out 불가 "
                  f"-- 자기 자신을 포함해서 계산")
            remaining_cache = dict(target_cls_cache)
        remaining_categories = {n: target_categories[n] for n in remaining_cache}
        prototypes = build_category_prototypes(remaining_cache, remaining_categories)
        ranked = classify_category(target_cls_cache[name], prototypes)
        probs[name] = category_probs_from_scores(ranked, category_order, temperature)
    return probs


def gather_category_probs(target_category_probs: dict, target_names: list, device: str) -> torch.Tensor:
    """배치 안 각 샘플의 target 이름에 맞는 카테고리 확률 벡터를 캐시에서 조회해 (B,K)로 쌓는다.
    gather_target_vecs와 짝을 이루는 함수 -- 이쪽은 카메라 각도와 무관(target 하나당 값 하나)."""
    return torch.stack([target_category_probs[name] for name in target_names], dim=0).to(device)


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


if __name__ == "__main__":
    # 이 파일을 직접 실행하면: DINOv3 CLS-프로토타입 카테고리 분류기가 "완전히 처음 보는 물체"에
    # 대해서도 동작하는지 leave-one-out으로 검증한다. target 15개 중 하나(held_out)를 프로토타입
    # 계산에서 완전히 빼고, 나머지 14개로만 카테고리 프로토타입을 만든 뒤, held_out을 분류해본다
    # -- "학습 데이터에 전혀 없던 새 물체가 온 상황"을 그대로 흉내낸 것이라 이게 곧 zero-shot
    # 일반화 성능의 실측치가 된다.
    from backbone import DINOv3Backbone
    from gt_similarity import discover_assets, resolve_target_from_data_folder
    from paths_config import ASSET_DIR

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LAYERS = (2, 5, 8, 11)
    TARGET_CAMS = ["center", "top", "left", "right", "bottom"]
    TARGETS = [
        "book_1", "book_2", "book_3", "book_4",
        "fruit_1", "fruit_2", "fruit_3", "fruit_4",
        "packaged_food_2", "packaged_food_3", "packaged_food_4",
        "toy_1", "toy_2", "toy_3", "toy_4",
    ]

    print(f"loading DINOv3 backbone (device={DEVICE})...")
    backbone = DINOv3Backbone(variant="vitb16", layers=LAYERS, device=DEVICE)
    usd_to_category = discover_assets(ASSET_DIR)
    target_categories = {}
    for name in TARGETS:
        usd_name, _ = resolve_target_from_data_folder(ASSET_DIR, name)
        target_categories[name] = usd_to_category[usd_name]

    print("encoding CLS vectors for all targets (5 cams averaged each)...")
    cls_cache, usable, skipped = precompute_target_cls_cache(backbone, TARGETS, DEVICE, TARGET_CAMS)
    if skipped:
        print(f"    [WARN] skipped (no mapping.json): {skipped}")

    print(f"\nleave-one-out zero-shot category classification ({len(usable)} targets):")
    correct = 0
    for held_out in usable:
        remaining_cache = {n: v for n, v in cls_cache.items() if n != held_out}
        remaining_categories = {n: target_categories[n] for n in remaining_cache}
        prototypes = build_category_prototypes(remaining_cache, remaining_categories)
        ranked = classify_category(cls_cache[held_out], prototypes)
        pred_cat, pred_score = ranked[0]
        true_cat = target_categories[held_out]
        ok = pred_cat == true_cat
        correct += int(ok)
        all_scores = "  ".join(f"{c}:{s:.3f}" for c, s in ranked)
        print(f"    [{'OK   ' if ok else 'WRONG'}] {held_out:20s} true={true_cat:15s} "
              f"pred={pred_cat:15s} ({all_scores})")
    print(f"\nleave-one-out accuracy: {correct}/{len(usable)} = {correct / len(usable) * 100:.1f}%")
