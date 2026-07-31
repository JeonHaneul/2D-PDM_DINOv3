import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, SiglipModel

from backbone import DINOv3Backbone, PATCH_SIZE
from gt_similarity import build_color_to_score, discover_assets, load_scene_mapping, render_gt_map, resolve_target_from_data_folder
from paths_config import ASSET_DIR
from similarity_model import SimilarityMapModel
from target_utils import bgr_to_chw, bgr_to_tensor, crop_with_mask, discover_target_frame_id, load_target_reference
from train_common import (
    EarlyStopping, accuracy_scores_from_counts, append_log, batch_accuracy_counts,
    build_qualitative_panel, discover_scene_ids, gather_target_vecs, precompute_target_vec_cache,
    split_scene_ids, target_paths,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 260714_data에 데이터가 준비된 15개 target 전체 (packaged_food_1은 scene 데이터 자체가 없어서 제외)
TARGETS = [
    "book_1", "book_2", "book_3", "book_4",
    "fruit_1", "fruit_2", "fruit_3", "fruit_4",
    "packaged_food_1", "packaged_food_2", "packaged_food_3", "packaged_food_4",
    "toy_1", "toy_2", "toy_3", "toy_4",
]
TARGET_CAMS = ["center", "top", "left", "right", "bottom"]

TRAIN_RATIO = 0.8   # target마다 scene 그룹을 셔플한 뒤 이 비율로 train/val 분할
SPLIT_SEED = None      # 정수 -> 항상 같은 분할(재현 가능). None -> 매 실행 랜덤(사용된 시드는 로그 출력)
CAMS = ["center", "top", "left", "right", "bottom"]  # scene 쪽 카메라 5개 전부 사용
ENV_STRIDE = 10         # 300개 env 중 몇 개마다 하나씩 쓸지. 디스크에서 그때그때 읽는 방식이라
                       # 메모리 상한과는 무관 -- v1과 동일한 트레이드오프.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = (2, 5, 8, 11)
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

SAVE_INTERVAL = 2
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA_PCT = 0.05

# --- SigLIP semantic encoder 설정 (2D_PDM-TH와 동일 체크포인트) ---
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
SIGLIP_DIM = 1152
SIGLIP_IMG_SZ = 384
_SIG_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SIG_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class MultiTargetSceneDataset(Dataset):
    """v1(train_similarity.py)과 완전히 동일 -- 디스크 스트리밍, RAM 캐싱 없음."""
    def __init__(self, target_scene_ids: dict, usd_to_category: dict, target_usd_names: dict):
        self.samples = []  # [(target_name, fname, prefix), ...]
        self.mapping_cache = {}  # (target_name, prefix) -> {usd_name: bgr색상}
        self.paths = {name: target_paths(name) for name in target_scene_ids}
        for name, scene_ids in target_scene_ids.items():
            paths = self.paths[name]
            for sid in scene_ids:
                prefix = f"scene{sid:05d}"
                self.mapping_cache[(name, prefix)] = load_scene_mapping(
                    os.path.join(paths["scene_dir"], "seg", f"{prefix}_mapping.json")
                )
                for env in range(0, 300, ENV_STRIDE):
                    for cam in CAMS:
                        fname = f"{prefix}_env{env:04d}_{cam}"
                        if os.path.isfile(os.path.join(paths["scene_dir"], "rgb", f"{fname}.png")):
                            self.samples.append((name, fname, prefix))
        self.usd_to_category = usd_to_category
        self.target_usd_names = target_usd_names

    def load_sample(self, idx):
        target_name, fname, prefix = self.samples[idx]
        paths = self.paths[target_name]
        rgb = cv2.imread(os.path.join(paths["scene_dir"], "rgb", f"{fname}.png"))

        precomputed_path = os.path.join(paths["gt_dir"], f"{fname}.png")
        if os.path.isfile(precomputed_path):
            gt_u8 = cv2.imread(precomputed_path, cv2.IMREAD_UNCHANGED)
            gt = gt_u8.astype(np.float32) / 255.0
        else:
            seg = cv2.imread(os.path.join(paths["scene_dir"], "seg", f"{fname}.png"))
            color_to_score = build_color_to_score(
                self.mapping_cache[(target_name, prefix)], self.target_usd_names[target_name], self.usd_to_category
            )
            gt = render_gt_map(seg, color_to_score)

        return rgb, gt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, gt = self.load_sample(idx)
        target_name = self.samples[idx][0]
        return bgr_to_chw(rgb), torch.from_numpy(gt.astype(np.float32)), target_name


def bgr_to_siglip_tensor(bgr: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """BGR uint8 -> (1,3,384,384) SigLIP 정규화(mean=0.5,std=0.5) 텐서."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIGLIP_IMG_SZ, SIGLIP_IMG_SZ), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return ((t - _SIG_MEAN) / _SIG_STD).to(device)


class SemanticProjection(nn.Module):
    """SigLIP semantic 임베딩(1152,)을 DINOv3 레이어별 차원(768)으로 투영하는 학습 가능한 어댑터.
    유일하게 gradient가 흐르는 semantic 관련 파라미터. 레이어마다 독립된 Linear를 쓰는 이유는 DINOv3 얕은 층/깊은 층이 담는 정보
    성격이 달라서(형태 vs 의미), 같은 SigLIP 신호도 레이어마다 다르게 변형해서 주입하는 게유리하기 때문."""
    def __init__(self, siglip_dim: int, embed_dim: int, num_layers: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(siglip_dim, embed_dim) for _ in range(num_layers)])

    def forward(self, semantic_raw: torch.Tensor) -> list:
        """semantic_raw: (B, siglip_dim) -> [(B, embed_dim), ...] per layer"""
        return [p(semantic_raw) for p in self.proj]

TARGET_LABELS = {
    "book_1": "a hardcover book",
    "book_2": "a hardcover book",
    "book_3": "a hardcover book",
    "book_4": "a hardcover book",
    "fruit_1": "an apple",
    "fruit_2": "an avocado",
    "fruit_3": "a lime",
    "fruit_4": "an orange",
    "packaged_food_1": "a can of tomato soup",
    "packaged_food_2": "a can of potted meat",
    "packaged_food_3": "a bottle of mustard",
    "packaged_food_4": "a box of pudding",
    "packaged_food_5": "a bag of instant coffee",
    "toy_1": "a toy truck",
    "toy_2": "a wooden ball toy",
    "toy_3": "a shield-shaped game controller toy",
    "toy_4": "a rubik's cube toy",
}


def target_to_prompt(target_name: str, category: str) -> str:
    """target -> "a photo of {구체적 이름}, a type of {category}" (TH의 PROMPT_TEMPLATE과 동일 형식).
    TARGET_LABELS에 없는 target(라벨 미작성)은 카테고리 수준으로 자동 폴백."""
    specific = TARGET_LABELS.get(target_name, f"a {category.replace('_', ' ')}")
    return f"a photo of {specific}, a type of {category.replace('_', ' ')}"


@torch.no_grad()
def encode_text_siglip(siglip_model, tokenizer, prompt: str, device: str) -> torch.Tensor:
    """텍스트(카테고리 프롬프트) -> L2정규화 (1152,) SigLIP 텍스트 임베딩."""
    tokens = tokenizer([prompt], padding="max_length", max_length=64, truncation=True,
                        return_tensors="pt").to(device)
    out = siglip_model.text_model(**tokens)
    return F.normalize(out.pooler_output[0], dim=0)  # (1152,)


@torch.no_grad()
def encode_target_semantic(siglip_model, device: str, target_rgb_dir: str, target_seg_dir: str,
                            target_mapping_path: str, target_frame_id: str, cam: str,
                            text_embed: torch.Tensor = None) -> torch.Tensor:
    """target crop 한 장을 SigLIP에 넣어 L2정규화된 (1152,) semantic 임베딩을 뽑는다.
    마스크는 우리 segmentation 기반(crop_with_mask)을 재사용
    text_embed가 주어지면 이미지 임베딩과 평균 fusion(TH와 동일, 같은 SigLIP 공간이라 평균이
    수학적으로 유효) -- 없으면 image-only."""
    tgt_rgb, tgt_mask, _ = load_target_reference(target_rgb_dir, target_seg_dir, target_mapping_path,
                                                  target_frame_id, cam)
    crop_rgb, _crop_mask, _ = crop_with_mask(tgt_rgb, tgt_mask, pad_ratio=0.25)
    sig_tensor = bgr_to_siglip_tensor(crop_rgb, device=device)
    out = siglip_model.vision_model(pixel_values=sig_tensor)
    img_embed = F.normalize(out.pooler_output[0], dim=0)  # (1152,)
    if text_embed is None:
        return img_embed
    return F.normalize((img_embed + text_embed) / 2.0, dim=0)


def precompute_target_semantic_cache(siglip_model, tokenizer, targets: list, device: str,
                                      target_categories: dict) -> dict:
    """target마다 semantic 임베딩 하나(center 카메라 기준 대표 시점 1장 + 인스턴스별 텍스트)를
    미리 계산해 캐싱. DINOv3 appearance(gather_target_vecs)와 달리 카메라별 augmentation이
    필요 없다고 보고("개념"은 보는 각도와 무관해야 함) center 한 장만 사용 -- 2D_PDM-TH도 target
    이미지 1장 기준. 텍스트는 프롬프트 문자열 기준으로 캐싱(TARGET_LABELS에 없어 폴백된 target들은
    같은 카테고리 프롬프트를 공유하게 됨)."""
    text_cache = {}
    cache = {}
    for name in targets:
        paths = target_paths(name)
        frame_id = discover_target_frame_id(paths["target_rgb_dir"])
        prompt = target_to_prompt(name, target_categories[name])
        if prompt not in text_cache:
            text_cache[prompt] = encode_text_siglip(siglip_model, tokenizer, prompt, device)
        cache[name] = encode_target_semantic(
            siglip_model, device, paths["target_rgb_dir"], paths["target_seg_dir"],
            paths["target_mapping_path"], frame_id, "center", text_embed=text_cache[prompt]
        )
    return cache


def gather_target_semantic(semantic_cache: dict, target_names: list, device: str) -> torch.Tensor:
    return torch.stack([semantic_cache[name] for name in target_names], dim=0).to(device)


def run_epoch(backbone, model, semantic_proj, loader, target_vec_cache, semantic_cache, optim=None, desc=""):
    train_mode = optim is not None
    model.train(train_mode)
    semantic_proj.train(train_mode)
    total_loss, total_n = 0.0, 0
    acc_counts = [0.0, 0.0, 0.0, 0.0, 0.0]

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for rgb_batch, gt_batch, target_names in pbar:
        rgb_batch = rgb_batch.to(DEVICE)
        gt_batch = gt_batch.to(DEVICE)
        B = rgb_batch.shape[0]
        names = list(target_names)

        scene_feats = backbone(rgb_batch)  # frozen, no_grad internally
        gt_patch = F.avg_pool2d(gt_batch[:, None], kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

        # semantic은 카메라 각도와 무관(target 하나당 값 하나)이므로 cam 루프 밖에서 한 번만 투영
        semantic_raw = gather_target_semantic(semantic_cache, names, DEVICE)
        sem_proj = semantic_proj(semantic_raw)  # [(B,768), ...] per layer, gradient 흐름

        if train_mode:
            appearance = gather_target_vecs(target_vec_cache, names, len(LAYERS), DEVICE, TARGET_CAMS, cam=None)
            target_vecs = [a + s for a, s in zip(appearance, sem_proj)]  # additive fusion (TH와 동일)
            out = model(scene_feats, target_vecs)
            pred_patch = out["prob_patch_res"]
            loss = F.mse_loss(pred_patch, gt_patch)
            optim.zero_grad()
            loss.backward()
            optim.step()

            counts = batch_accuracy_counts(pred_patch, gt_patch)
            for i in range(5):
                acc_counts[i] += counts[i]
            total_loss += loss.item() * B
            total_n += B
            pbar.set_postfix(mse=f"{loss.item():.5f}")
        else:
            with torch.no_grad():
                cam_losses = []
                for cam in TARGET_CAMS:
                    appearance = gather_target_vecs(target_vec_cache, names, len(LAYERS), DEVICE, TARGET_CAMS, cam=cam)
                    target_vecs = [a + s for a, s in zip(appearance, sem_proj)]
                    out = model(scene_feats, target_vecs)
                    pred_patch = out["prob_patch_res"]
                    loss = F.mse_loss(pred_patch, gt_patch)
                    cam_losses.append(loss.item())

                    counts = batch_accuracy_counts(pred_patch, gt_patch)
                    for i in range(5):
                        acc_counts[i] += counts[i]
                    total_loss += loss.item() * B
                    total_n += B
            pbar.set_postfix(mse=f"{sum(cam_losses) / len(cam_losses):.5f}")

    acc, bal_acc, iou = accuracy_scores_from_counts(*acc_counts)
    return total_loss / total_n, acc, bal_acc, iou


def make_panel(backbone, model, semantic_proj, val_ds, target_vec_cache, semantic_cache,
               sample_idx=None, extra_label=""):
    if sample_idx is None:
        sample_idx = random.randrange(len(val_ds))
    target_name, fname, _prefix = val_ds.samples[sample_idx]
    query_rgb, gt_map = val_ds.load_sample(sample_idx)
    H, W = query_rgb.shape[:2]

    paths = target_paths(target_name)
    frame_id = discover_target_frame_id(paths["target_rgb_dir"])
    tgt_rgb, tgt_mask, _ = load_target_reference(paths["target_rgb_dir"], paths["target_seg_dir"],
                                                   paths["target_mapping_path"], frame_id, "center")
    crop_rgb, _crop_mask, _ = crop_with_mask(tgt_rgb, tgt_mask, pad_ratio=0.25)

    appearance = [target_vec_cache[(target_name, "center")][li].unsqueeze(0) for li in range(len(LAYERS))]
    semantic_raw = semantic_cache[target_name].unsqueeze(0).to(DEVICE)

    was_training_m, was_training_s = model.training, semantic_proj.training
    model.eval()
    semantic_proj.eval()
    with torch.no_grad():
        sem_proj = semantic_proj(semantic_raw)
        target_vecs = [a + s for a, s in zip(appearance, sem_proj)]
        scene_feats = backbone(bgr_to_tensor(query_rgb, device=DEVICE))
        out = model(scene_feats, target_vecs, out_size=(H, W))
    model.train(was_training_m)
    semantic_proj.train(was_training_s)
    pred_full = out["prob_full_res"][0, 0].cpu().numpy()

    return build_qualitative_panel(f"{target_name}", crop_rgb, query_rgb, gt_map, pred_full, fname, extra_label)


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, f"multi_target_{run_id}_siglip")
    os.makedirs(run_dir, exist_ok=True)
    print(f"이번 실행 결과물 저장 위치: {run_dir}")

    print(f"loading DINOv3 backbone (vitb16, layers={LAYERS}, device={DEVICE})")
    backbone = DINOv3Backbone(variant="vitb16", layers=LAYERS, device=DEVICE)
    usd_to_category = discover_assets(ASSET_DIR)

    target_usd_names = {}
    for name in TARGETS:
        usd_name, _ = resolve_target_from_data_folder(ASSET_DIR, name)
        target_usd_names[name] = usd_name

    print(f"precomputing target appearance vectors for {len(TARGETS)} targets x {len(TARGET_CAMS)} cams...")
    target_vec_cache, usable_targets, skipped_targets = precompute_target_vec_cache(backbone, TARGETS, DEVICE, TARGET_CAMS)
    if skipped_targets:
        print(f"    [WARN] target/mapping.json 없어서 제외됨: {skipped_targets}")
    print(f"    usable targets ({len(usable_targets)}): {usable_targets}")

    print(f"loading SigLIP so400m ({SIGLIP_MODEL_ID}, frozen) ...")
    siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model.eval()
    for p in siglip_model.parameters():
        p.requires_grad_(False)
    siglip_model.to(DEVICE)
    siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)

    target_categories = {name: usd_to_category[target_usd_names[name]] for name in usable_targets}
    print(f"precomputing target semantic embeddings ({len(usable_targets)} targets, image+text fusion)...")
    semantic_cache = precompute_target_semantic_cache(siglip_model, siglip_tokenizer, usable_targets, DEVICE,
                                                       target_categories)

    # target별로 scene을 discover하고 leakage 없이 분할한 뒤, train/val 각각 전부 합침
    # 버그 수정: 예전엔 split_scene_ids(..., SPLIT_SEED)를 target마다 반복 호출했는데, SPLIT_SEED=None이면
    # 매 호출마다 각자 새로운 랜덤 시드를 뽑아서(target 15개면 15개의 서로 다른 시드) 실제로는 target별로
    # 전부 다른 시드로 분할됐다. 그런데 로그에는 마지막 target의 시드 하나만 찍혀서, 그 값을 SPLIT_SEED에
    # 넣어도 마지막 target만 재현되고 나머지 target은 원래 실행과 다른 분할이 나온다. 이제 시드를
    # 루프 시작 전에 딱 한 번만 확정해서 모든 target에 동일하게 넘긴다 -- 이러면 이 하나의 값만으로
    # 전체 target의 분할이 정확히 재현된다(target마다 scene_ids 목록 자체는 다르므로 같은 시드를
    # 써도 각 target은 각자 다른 셔플 결과를 얻는다).
    resolved_split_seed = SPLIT_SEED if SPLIT_SEED is not None else random.SystemRandom().randint(0, 2**31 - 1)
    train_scene_ids, val_scene_ids = {}, {}
    for name in usable_targets:
        paths = target_paths(name)
        ids = discover_scene_ids(paths["scene_dir"])
        tr, va, _ = split_scene_ids(ids, TRAIN_RATIO, resolved_split_seed)
        train_scene_ids[name] = tr
        val_scene_ids[name] = va
    print(f"scene split done (split_seed={resolved_split_seed}"
          f"{' -- SPLIT_SEED=None이라 매 실행 랜덤, 재현하려면 이 값을 SPLIT_SEED에 넣으세요' if SPLIT_SEED is None else ''})")
    for name in usable_targets:
        print(f"    {name}: train={train_scene_ids[name]}  val={val_scene_ids[name]}")

    train_ds = MultiTargetSceneDataset(train_scene_ids, usd_to_category, target_usd_names)
    val_ds = MultiTargetSceneDataset(val_scene_ids, usd_to_category, target_usd_names)
    print(f"train samples={len(train_ds)}  val samples={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = SimilarityMapModel(embed_dim=backbone.embed_dim, num_layers=len(LAYERS)).to(DEVICE)
    semantic_proj = SemanticProjection(SIGLIP_DIM, backbone.embed_dim, len(LAYERS)).to(DEVICE)
    optim = torch.optim.AdamW(list(model.parameters()) + list(semantic_proj.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    log_path = os.path.join(run_dir, "train_log.txt")
    open(log_path, "w").close()
    best_ckpt_path = os.path.join(run_dir, "similarity_head_best.pt")
    last_ckpt_path = os.path.join(run_dir, "similarity_head_last.pt")
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta_pct=EARLY_STOP_MIN_DELTA_PCT)

    def save_ckpt(path):
        # DINOv3/SigLIP은 frozen이라 저장 안 함 -- 학습 대상(model, semantic_proj)만 저장
        torch.save({"model_state": model.state_dict(), "semantic_proj_state": semantic_proj.state_dict()}, path)

    history = []
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_bal_acc, train_iou = run_epoch(
            backbone, model, semantic_proj, train_loader, target_vec_cache, semantic_cache, optim,
            desc=f"epoch {epoch + 1}/{EPOCHS} [train]")
        val_loss, val_acc, val_bal_acc, val_iou = run_epoch(
            backbone, model, semantic_proj, val_loader, target_vec_cache, semantic_cache, optim=None,
            desc=f"epoch {epoch + 1}/{EPOCHS} [val]")
        scheduler.step()
        history.append((epoch + 1, train_loss, val_loss))
        log_line = (f"epoch {epoch + 1:2d}/{EPOCHS}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}\n"
                    f"    train_acc={train_acc:.4f} train_bal_acc={train_bal_acc:.4f} train_iou={train_iou:.4f}\n"
                    f"    val_acc={val_acc:.4f}   val_bal_acc={val_bal_acc:.4f}   val_iou={val_iou:.4f}")
        print(log_line)
        append_log(log_path, log_line)

        if early_stopper(val_loss):
            save_ckpt(best_ckpt_path)
            print(f"    -> best model 저장: {best_ckpt_path}")
            append_log(log_path, f"    -> new best (val_mse={val_loss:.5f}), saved {best_ckpt_path}")

        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            save_ckpt(last_ckpt_path)
            panel = make_panel(backbone, model, semantic_proj, val_ds, target_vec_cache, semantic_cache,
                                extra_label=f"(epoch {epoch + 1}/{EPOCHS})")
            panel_path = os.path.join(run_dir, f"trained_result_panel_epoch{epoch + 1:03d}.png")
            cv2.imwrite(panel_path, panel)
            print(f"    saved: {last_ckpt_path}, {panel_path}")

        if early_stopper.early_stop:
            msg = f"EarlyStopping 발동 (patience={EARLY_STOP_PATIENCE}) -- epoch {epoch + 1}에서 학습 중단"
            print(msg)
            append_log(log_path, msg)
            break

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs_, tr, va = zip(*history)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_, tr, label="train MSE")
        plt.plot(epochs_, va, label="val MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE (patch-res, [0,1] targets)")
        plt.title(f"SimilarityMapModel training v2/SigLIP (multi-target, {len(usable_targets)} targets)")
        plt.legend()
        plt.tight_layout()
        loss_plot_path = os.path.join(run_dir, "train_loss_curve.png")
        plt.savefig(loss_plot_path, dpi=120)
        print(f"saved: {loss_plot_path}")
    except ImportError:
        print("matplotlib not available, skipping loss curve plot")

    print(f"done. best checkpoint: {best_ckpt_path} (val_mse={early_stopper.best_loss:.5f})")


if __name__ == "__main__":
    main()
