import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backbone import DINOv3Backbone, PATCH_SIZE
from gt_similarity import build_color_to_score, discover_assets, load_scene_mapping, render_gt_map, resolve_target_from_data_folder
from similarity_model import SimilarityMapModel
from target_utils import bgr_to_chw, bgr_to_tensor, crop_with_mask, discover_target_frame_id, load_target_reference
from train_common import (
    EarlyStopping, accuracy_scores_from_counts, append_log, batch_accuracy_counts,
    build_qualitative_panel, discover_scene_ids, gather_target_vecs, precompute_target_vec_cache,
    split_scene_ids, target_paths,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(ROOT, "asset")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 260714_data에 데이터가 준비된 15개 target 전체 (packaged_food_1은 scene 데이터 자체가 없어서 제외)
TARGETS = [
    "book_1", "book_2", "book_3", "book_4",
    "fruit_1", "fruit_2", "fruit_3", "fruit_4",
    "packaged_food_2", "packaged_food_3", "packaged_food_4",
    "toy_1", "toy_2", "toy_3", "toy_4",
]
TARGET_CAMS = ["center", "top", "left", "right", "bottom"]

TRAIN_RATIO = 0.8   # target마다 scene 그룹을 셔플한 뒤 이 비율로 train/val 분할
SPLIT_SEED = None      # 정수 -> 항상 같은 분할(재현 가능). None -> 매 실행 랜덤(사용된 시드는 로그 출력)
CAMS = ["center", "top", "left", "right", "bottom"]  # scene 쪽 카메라 5개 전부 사용
ENV_STRIDE = 1         # 300개 env 중 몇 개마다 하나씩 쓸지. 디스크에서 그때그때 읽는 방식이라
                       # 메모리 상한과는 무관 -- "epoch 하나 도는 데 걸리는 시간 vs 데이터 다양성"
                       # 트레이드오프일 뿐. 1이면 전체(15target*10scene*300env*5cam) 다 씀.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = (2, 5, 8, 11)
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

SAVE_INTERVAL = 2
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA_PCT = 0.05


class MultiTargetSceneDataset(Dataset):
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
        """디스크에서 RGB(BGR uint8 ndarray)와 GT(float32 [0,1] ndarray)를 읽어서 반환.
        __getitem__과 make_panel이 공통으로 사용."""
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
            gt = render_gt_map(seg, color_to_score)  # (H,W) float32 in [0,1]

        return rgb, gt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, gt = self.load_sample(idx)
        target_name = self.samples[idx][0]
        return bgr_to_chw(rgb), torch.from_numpy(gt.astype(np.float32)), target_name


def run_epoch(backbone, model, loader, target_vec_cache, optim=None, desc=""):
    train_mode = optim is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    acc_counts = [0.0, 0.0, 0.0, 0.0, 0.0]

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for rgb_batch, gt_batch, target_names in pbar:
        rgb_batch = rgb_batch.to(DEVICE)
        gt_batch = gt_batch.to(DEVICE)
        B = rgb_batch.shape[0]
        names = list(target_names)

        scene_feats = backbone(rgb_batch)  # frozen, no_grad internally -- target과 무관하므로 1번만
        gt_patch = F.avg_pool2d(gt_batch[:, None], kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

        if train_mode:
            target_vecs = gather_target_vecs(target_vec_cache, names, len(LAYERS), DEVICE, TARGET_CAMS, cam=None)
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
                    target_vecs = gather_target_vecs(target_vec_cache, names, len(LAYERS), DEVICE, TARGET_CAMS, cam=cam)
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


def make_panel(backbone, model, val_ds, target_vec_cache, sample_idx=None, extra_label=""):
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

    target_vecs = [target_vec_cache[(target_name, "center")][li].unsqueeze(0) for li in range(len(LAYERS))]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        scene_feats = backbone(bgr_to_tensor(query_rgb, device=DEVICE))
        out = model(scene_feats, target_vecs, out_size=(H, W))
    model.train(was_training)
    pred_full = out["prob_full_res"][0, 0].cpu().numpy()

    return build_qualitative_panel(f"{target_name}", crop_rgb, query_rgb, gt_map, pred_full, fname, extra_label)


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, f"multi_target_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"이번 실행 결과물 저장 위치: {run_dir}")

    print(f"loading DINOv3 backbone (vitb16, layers={LAYERS}, device={DEVICE})")
    backbone = DINOv3Backbone(variant="vitb16", layers=LAYERS, device=DEVICE)
    usd_to_category = discover_assets(ASSET_DIR)

    target_usd_names = {}
    for name in TARGETS:
        usd_name, _ = resolve_target_from_data_folder(ASSET_DIR, name)
        target_usd_names[name] = usd_name

    print(f"precomputing target vectors for {len(TARGETS)} targets x {len(TARGET_CAMS)} cams...")
    target_vec_cache, usable_targets, skipped_targets = precompute_target_vec_cache(backbone, TARGETS, DEVICE, TARGET_CAMS)
    if skipped_targets:
        print(f"    [WARN] target/mapping.json 없어서 제외됨: {skipped_targets}")
    print(f"    usable targets ({len(usable_targets)}): {usable_targets}")

    # target별로 scene을 discover하고 leakage 없이 분할한 뒤, train/val 각각 전부 합침
    train_scene_ids, val_scene_ids = {}, {}
    used_seed = None
    for name in usable_targets:
        paths = target_paths(name)
        ids = discover_scene_ids(paths["scene_dir"])
        tr, va, used_seed = split_scene_ids(ids, TRAIN_RATIO, SPLIT_SEED)
        train_scene_ids[name] = tr
        val_scene_ids[name] = va
    print(f"scene split done (split_seed={used_seed}"
          f"{' -- SPLIT_SEED=None이라 매 실행 랜덤, 재현하려면 이 값을 SPLIT_SEED에 넣으세요' if SPLIT_SEED is None else ''})")
    for name in usable_targets:
        print(f"    {name}: train={train_scene_ids[name]}  val={val_scene_ids[name]}")

    train_ds = MultiTargetSceneDataset(train_scene_ids, usd_to_category, target_usd_names)
    val_ds = MultiTargetSceneDataset(val_scene_ids, usd_to_category, target_usd_names)
    print(f"train samples={len(train_ds)}  val samples={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = SimilarityMapModel(embed_dim=backbone.embed_dim, num_layers=len(LAYERS)).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    log_path = os.path.join(run_dir, "train_log.txt")
    open(log_path, "w").close()
    best_ckpt_path = os.path.join(run_dir, "similarity_head_best.pt")
    last_ckpt_path = os.path.join(run_dir, "similarity_head_last.pt")
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta_pct=EARLY_STOP_MIN_DELTA_PCT)

    history = []
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_bal_acc, train_iou = run_epoch(
            backbone, model, train_loader, target_vec_cache, optim, desc=f"epoch {epoch + 1}/{EPOCHS} [train]")
        val_loss, val_acc, val_bal_acc, val_iou = run_epoch(
            backbone, model, val_loader, target_vec_cache, optim=None, desc=f"epoch {epoch + 1}/{EPOCHS} [val]")
        scheduler.step()
        history.append((epoch + 1, train_loss, val_loss))
        log_line = (f"epoch {epoch + 1:2d}/{EPOCHS}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}\n"
                    f"    train_acc={train_acc:.4f} train_bal_acc={train_bal_acc:.4f} train_iou={train_iou:.4f}\n"
                    f"    val_acc={val_acc:.4f}   val_bal_acc={val_bal_acc:.4f}   val_iou={val_iou:.4f}")
        print(log_line)
        append_log(log_path, log_line)

        if early_stopper(val_loss):
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    -> best model 저장: {best_ckpt_path}")
            append_log(log_path, f"    -> new best (val_mse={val_loss:.5f}), saved {best_ckpt_path}")

        if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), last_ckpt_path)
            panel = make_panel(backbone, model, val_ds, target_vec_cache,
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
        plt.title(f"SimilarityMapModel training v1 (multi-target, {len(usable_targets)} targets)")
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
