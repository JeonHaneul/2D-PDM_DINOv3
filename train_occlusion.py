"""Occlusion 스트림 학습 파이프라인, Stage 2 (4-model ablation, 1-seed smoke test).

=== 이전 단계 요약 ===
Stage 1(target_agnostic_rgbd vs full, 2-model, 120/30 split)에서 확인된 것:
  - target/scene pool을 4개 target 공유로 고쳐서 confound 제거
  - coverage-aware GT downsampling, best-checkpoint(진짜 argmin) 저장, 4x4 wrong-target
    confusion matrix, invariant 검증까지 전부 코드로 확인 완료 -- Stage 1 자체는 통과.
  - 다만 Stage 1 재실행 간 성능 변동(IoU 0.497 -> 0.359)을 "checkpoint 버그 탓"으로 잘못
    설명했었음 -- 실제로는 checkpoint 버그를 고치면 이전 실행 성능은 오히려 올라갔어야 함
    (저장 epoch 21: IoU 0.497 -> 진짜 최소 epoch 24: IoU 0.516). 즉 두 실행의 차이는
    checkpoint 수정 때문이 아니라 explicit seeding이 바뀌면서 생긴 순수한 run-to-run
    variance였음 -- 이게 바로 1-seed 결과를 못 믿는 이유이고, 3-seed 반복이 필요한 근거.

=== 이번(Stage 2) 변경 ===
  1. target_agnostic_rgbd/full 2-model -> 4-model ablation으로 확장
     (target 정보를 appearance/geometry 두 경로로 분리해서 독립적으로 끄고 켬):
         Variant              | appearance(DINO) | geometry(FiLM)
         target_agnostic_rgbd |        X          |       X
         appearance_only      |        O          |       X
         geometry_only        |        X          |       O
         full                 |        O          |       O
  2. validation(early stopping/checkpoint 선택)과 test(4x4 confusion matrix, 최종 모델 비교)를
     분리 -- 이전엔 val set이 두 역할을 동시에 해서 "val에서 잘 고른 checkpoint가 val 자체에서
     평가받는" 낙관 편향이 있었음. 150개 공유 scene을 100/20/30(train/val/test)으로 고정.
  3. "wrong target 최악"을 loss 기준 하나로만 뽑아서 IoU 기준 최악과 다른 조합을 가리킬 수
     있었음(실측: loss 최악=shift 1, IoU 최악=shift 3) -- worst-by-loss/worst-by-IoU/평균을
     전부 분리해서 기록.
  4. SEEDS 리스트로 감싸서 나중에 [0,1,2] 3-seed 확장이 SEEDS만 바꾸면 되게 구조화.

=== seed 0 결과 검토 후(evaluate_stage2_diagnostics.py) 추가 수정 ===
  5. pooled test IoU만 보고 "appearance_only(0.461)가 full(0.422)보다 낫다"고 했던 판단은
     성급했음 -- 150개(30 scene x 5 cam)를 독립 표본처럼 pooling하면 같은 scene의 5개 카메라가
     서로 강하게 상관돼 있다는 사실이 무시됨. scene 단위(카메라 5장 평균)로 다시 묶어 paired
     비교하면 full-appearance_only 평균 차이가 +0.0003(std=0.053, n=30)으로 사실상 0 --
     이번 1-seed 결과로는 세 conditioned variant 사이 우열을 말할 근거가 없음(target_agnostic과의
     차이는 확실함). main()에 scene 단위 paired 비교를 기본 내장.
  6. confusion matrix를 full만이 아니라 appearance_only/geometry_only도 계산하도록 확장 --
     toy_3<->packaged_food_2 혼동이 geometry_only(84% 유지)에서 appearance_only(59% 유지)보다
     훨씬 강하게 나타남, 즉 이 혼동은 주로 geometry-FiLM 경로에서 온다는 신호.
  7. per-sample(scene, camera, variant, shift 단위) 결과를 CSV로 저장 -- 이후 scene 단위
     재집계·paired 비교에 필요.
  8. seed 0은 재학습하지 않고 기존 checkpoint 재사용(evaluate_stage2_diagnostics.py), 이 파일은
     SEEDS=[1,2]만 새로 학습해서 3-seed를 채움. 이 시점부터 이 TEST set(30 scene)은 이미 구조
     비교에 여러 번 쓰였으므로 "완전히 손 안 댄 최종 test"가 아니라 Stage 2 exploratory로
     취급함 -- 코드/구조는 여기서 고정하고 seed 1,2 결과를 보고 추가로 구조를 바꾸지 않는다.

여전히 seen-target(4개 전부 학습에 포함)·scale=1.0 범위 안. target-held-out/unseen-scale은
이 4-model ablation(3-seed)이 끝난 뒤 별도 단계."""
import csv
import hashlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import DINOv3Backbone, PATCH_SIZE
from occlusion_dataset import CAMS, GT_ROOT, OcclusionDataset
from occlusion_model import GEOMETRY_DIM, OcclusionMapModel, encode_target_occlusion_frame
from target_utils import bgr_to_chw, discover_target_frame_id
from train_common import (
    EarlyStopping, accuracy_scores_from_counts, append_log, batch_accuracy_counts, target_paths,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

TARGETS = ["book_1", "fruit_1", "toy_3", "packaged_food_2"]
CLUTTER_TARGET = "packaged_food_1"
SCALE = 1.0
N_TRAIN_SCENES, N_VAL_SCENES, N_TEST_SCENES = 100, 20, 30
SPLIT_SEED = 0  # scene split은 seed와 무관하게 고정(재현성) -- 모델별 반복은 MODEL_SEED로
SEEDS = [1, 2]  # seed 0은 occlusion_stage2_20260810_170053에 이미 완료됨(재학습 안 함) -- 이 실행은
                # seed 1,2만 추가해서 3-seed를 채운다. seed 0 checkpoint는 evaluate_stage2_diagnostics.py로
                # 이미 3-variant confusion matrix + per-sample CSV까지 뽑아둠(재사용).
DIAGNOSE_VARIANTS = ["appearance_only", "geometry_only", "full"]  # target_agnostic은 target을
                # 안 보므로 confusion matrix가 무의미해서 제외

VARIANT_CONFIGS = {
    # (zero_appearance, zero_geometry)
    "target_agnostic_rgbd": (True, True),
    "appearance_only": (False, True),
    "geometry_only": (True, False),
    "full": (False, False),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = (2, 5, 8, 11)
BATCH_SIZE = 16
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA_PCT = 0.01
LR = 1e-3
FOREGROUND_LOSS_WEIGHT = 3.0
EPS = 1e-6


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================== manifest ==============================

def build_shared_scene_manifest_3way(targets: list, clutter_target: str, scale: float) -> dict:
    """4개 target 전부에 corrected GT가 존재하는 scene_key의 교집합을 100/20/30(train/val/test)
    으로 고정 분할해서 저장. val은 early stopping/checkpoint 선택에만, test는 4x4 confusion
    matrix와 최종 모델 비교에만(한 번만) 쓴다 -- Stage 1의 120/30 manifest와는 별도 파일."""
    manifest_path = os.path.join(
        GT_ROOT, f"shared_manifest_3way_{'_'.join(targets)}_scale{scale}.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)

    per_target_scenes = {}
    for t in targets:
        gt_root = os.path.join(GT_ROOT, t, f"scale_{scale}")
        keys = set()
        for d in os.listdir(gt_root):
            full = os.path.join(gt_root, d)
            if not os.path.isdir(full) or not d.startswith("scene") or d.startswith("_"):
                continue
            sk = d.rsplit("_", 1)[0]
            if os.path.isfile(os.path.join(full, "map_corrected.npy")):
                keys.add(sk)
        cam_complete = {sk for sk in keys if all(
            os.path.isdir(os.path.join(gt_root, f"{sk}_{c}")) for c in CAMS)}
        per_target_scenes[t] = cam_complete

    shared = sorted(set.intersection(*per_target_scenes.values()))
    assert len(shared) == N_TRAIN_SCENES + N_VAL_SCENES + N_TEST_SCENES, (
        f"공유 scene 수({len(shared)})가 {N_TRAIN_SCENES}+{N_VAL_SCENES}+{N_TEST_SCENES}와 다름")
    shuffled = list(shared)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    train_scenes = sorted(shuffled[:N_TRAIN_SCENES])
    val_scenes = sorted(shuffled[N_TRAIN_SCENES:N_TRAIN_SCENES + N_VAL_SCENES])
    test_scenes = sorted(shuffled[N_TRAIN_SCENES + N_VAL_SCENES:])

    manifest = {
        "targets": targets, "clutter_target": clutter_target, "scale": scale,
        "per_target_scene_counts": {t: len(v) for t, v in per_target_scenes.items()},
        "n_shared_scenes": len(shared), "shared_scenes": shared,
        "split_seed": SPLIT_SEED,
        "train_scenes": train_scenes, "val_scenes": val_scenes, "test_scenes": test_scenes,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ============================== target cache ==============================

def precompute_occlusion_target_cache(backbone, targets: list, device: str) -> dict:
    cache = {}
    for name in targets:
        paths = target_paths(name)
        frame_id = discover_target_frame_id(paths["target_rgb_dir"])
        for cam in CAMS:
            vecs, geom = encode_target_occlusion_frame(
                backbone, device, paths["target_rgb_dir"], paths["target_seg_dir"],
                paths["target_mapping_path"], frame_id, cam,
            )
            cache[(name, cam)] = (vecs, torch.from_numpy(geom).to(device))
    return cache


def gather_target_inputs(cache: dict, names: list, cams: list, num_layers: int, device: str,
                          zero_appearance: bool = False, zero_geometry: bool = False) -> tuple:
    """zero_appearance: DINO target_vecs(matching interaction용)를 0으로 고정.
    zero_geometry: target_geometry(FiLM 입력)를 0으로 고정. 둘을 독립적으로 제어해서
    appearance-only/geometry-only/target_agnostic/full 4개 조합을 전부 만들 수 있다."""
    if zero_appearance:
        C = cache[(names[0], cams[0])][0][0].shape[0]
        target_vecs = [torch.zeros(len(names), C, device=device) for _ in range(num_layers)]
    else:
        target_vecs = [
            torch.stack([cache[(n, c)][0][li] for n, c in zip(names, cams)]).to(device)
            for li in range(num_layers)
        ]
    if zero_geometry:
        geometry = torch.zeros(len(names), GEOMETRY_DIM, device=device)
    else:
        geometry = torch.stack([cache[(n, c)][1] for n, c in zip(names, cams)]).to(device)
    return target_vecs, geometry


# ============================== loss / pooling ==============================

def coverage_aware_patch_pool(x: torch.Tensor, coverage: torch.Tensor) -> tuple:
    num = F.avg_pool2d((x * coverage)[:, None], kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    den = F.avg_pool2d(coverage[:, None], kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    return num / (den + EPS), den


def occlusion_loss(pred_patch: torch.Tensor, gt_patch: torch.Tensor, coverage_patch: torch.Tensor) -> torch.Tensor:
    bce_map = F.binary_cross_entropy(pred_patch.clamp(1e-6, 1 - 1e-6), gt_patch, reduction="none")
    l1_map = F.smooth_l1_loss(pred_patch, gt_patch, reduction="none")
    weight = 1.0 + FOREGROUND_LOSS_WEIGHT * gt_patch
    combined = bce_map + weight * l1_map
    denom = coverage_patch.sum().clamp(min=1.0)
    return (combined * coverage_patch).sum() / denom


# ============================== epoch loop ==============================

def run_epoch(backbone, model, loader, target_cache, optim=None, desc="",
              zero_appearance=False, zero_geometry=False, target_shift: int = 0,
              per_pair_stats: dict = None, per_sample_rows: list = None,
              row_tags: dict = None):
    """반환하는 acc/bal_acc/iou는 **sample(scene x camera) 단위로 각각 IoU를 구한 뒤 평균**한
    값이다(픽셀/union count로 pooling한 값이 아님) -- pooled 집계는 union이 큰(=target 발자국이
    큰) sample에 자동으로 더 큰 가중치를 줘서, 실제로는 sample 단위 평균에서 순위가 뒤집히는
    경우가 있었다(실측: seed 2에서 pooled로는 appearance_only(0.427) > geometry_only(0.386)
    였는데 sample 단위 평균으론 geometry_only(0.388) > appearance_only(0.261)로 부호가 반대).
    pooled 값은 pooled_iou로 별도 반환(참고용)."""
    train_mode = optim is not None
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    acc_counts = [0.0, 0.0, 0.0, 0.0, 0.0]  # pooled(참고용)
    per_sample_metrics = []  # [(acc, bal_acc, iou), ...] -- sample당 1개, 최종 반환값의 기준

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for rgb_batch, depth_batch, gt_batch, coverage_batch, names, cams, _sks in pbar:
        rgb_tensor = torch.stack([bgr_to_chw(r.numpy()) for r in rgb_batch]).to(DEVICE)
        depth_batch = depth_batch.to(DEVICE)
        gt_batch = gt_batch.to(DEVICE)
        coverage_batch = coverage_batch.to(DEVICE)
        B = rgb_tensor.shape[0]

        orig_names = list(names)
        if target_shift != 0:
            assigned_names = [TARGETS[(TARGETS.index(n) + target_shift) % len(TARGETS)] for n in orig_names]
            assert all(a != o for a, o in zip(assigned_names, orig_names)), "wrong-target 적용률 100% 위반"
        else:
            assigned_names = orig_names

        scene_feats = backbone(rgb_tensor)
        gt_patch, coverage_patch = coverage_aware_patch_pool(gt_batch, coverage_batch)
        target_vecs, target_geometry = gather_target_inputs(
            target_cache, assigned_names, list(cams), len(LAYERS), DEVICE,
            zero_appearance=zero_appearance, zero_geometry=zero_geometry)

        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            out = model(scene_feats, depth_batch, target_vecs, target_geometry)
            pred_patch = out["prob_patch_res"]
            loss = occlusion_loss(pred_patch, gt_patch, coverage_patch)
            if train_mode:
                optim.zero_grad()
                loss.backward()
                optim.step()

        cov_mask = coverage_patch > 0.5
        counts = batch_accuracy_counts(pred_patch[cov_mask], gt_patch[cov_mask])
        for i in range(5):
            acc_counts[i] += counts[i]
        total_loss += loss.item() * B
        total_n += B
        pbar.set_postfix(loss=f"{loss.item():.5f}")

        # sample(scene x camera) 단위 IoU -- 항상 계산(반환값의 기준이 되므로 per_pair_stats/
        # per_sample_rows 유무와 무관하게 매번 필요).
        for i, (t_true, t_assigned, sk) in enumerate(zip(orig_names, assigned_names, _sks)):
            m = cov_mask[i]
            pt_counts = batch_accuracy_counts(pred_patch[i:i + 1][m[None]], gt_patch[i:i + 1][m[None]])
            p_acc, p_bal, p_iou = accuracy_scores_from_counts(*pt_counts)
            per_sample_metrics.append((p_acc, p_bal, p_iou))

            if per_pair_stats is not None:
                s = per_pair_stats.setdefault((t_true, t_assigned), [0.0] * 5 + [0])
                for k in range(5):
                    s[k] += pt_counts[k]
                s[5] += 1
            if per_sample_rows is not None:
                p_loss = occlusion_loss(pred_patch[i:i + 1], gt_patch[i:i + 1], coverage_patch[i:i + 1]).item()
                row = dict(row_tags or {})
                row.update({"scene_key": sk, "camera": cams[i], "true_target": t_true,
                            "assigned_target": t_assigned, "loss": p_loss, "iou": p_iou})
                per_sample_rows.append(row)

    n = len(per_sample_metrics)
    acc = sum(m[0] for m in per_sample_metrics) / n
    bal_acc = sum(m[1] for m in per_sample_metrics) / n
    iou = sum(m[2] for m in per_sample_metrics) / n
    pooled_iou = accuracy_scores_from_counts(*acc_counts)[2]
    return total_loss / total_n, acc, bal_acc, iou, pooled_iou


# ============================== invariant checks ==============================

def verify_coverage_zero_no_contribution() -> bool:
    pred = torch.rand(2, 1, 4, 4)
    gt = torch.rand(2, 1, 4, 4)
    coverage = torch.zeros(2, 1, 4, 4)
    coverage[:, :, :2, :] = 1.0
    loss_full = occlusion_loss(pred, gt, coverage)
    pred2 = pred.clone()
    pred2[:, :, 2:, :] = torch.rand(2, 1, 2, 4) * 100
    loss_changed = occlusion_loss(pred2, gt, coverage)
    return abs(loss_full.item() - loss_changed.item()) < 1e-6


def verify_coverage_dilution_fix() -> bool:
    gt = torch.zeros(1, PATCH_SIZE, PATCH_SIZE)
    coverage = torch.zeros(1, PATCH_SIZE, PATCH_SIZE)
    half = PATCH_SIZE // 2
    gt[:, :half, :] = 1.0
    coverage[:, :half, :] = 1.0
    gt_patch, coverage_patch = coverage_aware_patch_pool(gt, coverage)
    return abs(gt_patch.item() - 1.0) < 1e-4 and abs(coverage_patch.item() - 0.5) < 1e-4


def check_invariants(manifest: dict, train_ds, val_ds, test_ds, coverage_zero_contributes: bool,
                      coverage_dilution_fixed: bool, checkpoint_reports: dict, log_path: str,
                      shift0_mismatch: list = None):
    checks = [
        ("train scene 수 == 100", len(manifest["train_scenes"]) == N_TRAIN_SCENES),
        ("val scene 수 == 20", len(manifest["val_scenes"]) == N_VAL_SCENES),
        ("test scene 수 == 30", len(manifest["test_scenes"]) == N_TEST_SCENES),
        ("train sample 수 == 2000", len(train_ds) == N_TRAIN_SCENES * len(TARGETS) * len(CAMS)),
        ("val sample 수 == 400", len(val_ds) == N_VAL_SCENES * len(TARGETS) * len(CAMS)),
        ("test sample 수 == 600", len(test_ds) == N_TEST_SCENES * len(TARGETS) * len(CAMS)),
        ("train/val scene 교집합 == 0",
         len(set(manifest["train_scenes"]) & set(manifest["val_scenes"])) == 0),
        ("train/test scene 교집합 == 0",
         len(set(manifest["train_scenes"]) & set(manifest["test_scenes"])) == 0),
        ("val/test scene 교집합 == 0",
         len(set(manifest["val_scenes"]) & set(manifest["test_scenes"])) == 0),
        ("coverage=0 patch가 loss에 기여하지 않음", coverage_zero_contributes),
        ("coverage 경계 GT 희석 수정 확인", coverage_dilution_fixed),
        ("wrong-target 적용률 == 100% (run_epoch assert 통과)", True),
    ]
    for v, rep in checkpoint_reports.items():
        checks.append((f"[{v}] 저장된 checkpoint epoch({rep['saved_epoch']}) == 실제 argmin epoch({rep['true_argmin_epoch']})",
                        rep["saved_epoch"] == rep["true_argmin_epoch"]))
        checks.append((f"[{v}] 저장된 checkpoint val_loss == 관측된 최솟값",
                        abs(rep["saved_val_loss"] - rep["true_min_val_loss"]) < 1e-9))
    # shift=0(target 안 바꿈)은 confusion matrix 루프 진입 전의 "기본 test 평가"와 완전히
    # 동일해야 함 -- ablation flag(zero_appearance/zero_geometry)를 안 넘기는 버그가 있으면
    # 이 invariant가 즉시 잡아낸다(실제로 이 버그가 있었고, 이 체크가 없어서 못 잡았었음).
    for variant_name, ok, main_loss, shift0_loss, main_iou, shift0_iou in (shift0_mismatch or []):
        checks.append((f"[{variant_name}] shift=0 loss/iou == 기본 test 평가 "
                        f"({main_loss:.5f}/{main_iou:.3f} vs {shift0_loss:.5f}/{shift0_iou:.3f})", ok))

    append_log(log_path, "\n=== invariant 검증 ===")
    all_pass = True
    for name, ok in checks:
        append_log(log_path, f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok
    append_log(log_path, f"=> {'전부 통과' if all_pass else '일부 실패 -- seed 확장하면 안 됨'}")
    return all_pass


# ============================== confusion matrix reporting ==============================

def report_confusion_matrix(shift_summaries: dict, pair_stats: dict, log_path: str, tag: str):
    append_log(log_path, f"\n=== {tag}: shift별 요약 (loss/IoU) ===")
    for shift, (loss, iou) in shift_summaries.items():
        tag2 = "correct(shift=0)" if shift == 0 else f"wrong(shift={shift})"
        append_log(log_path, f"  {tag2}: loss={loss:.5f} iou={iou:.3f}")

    correct_loss, correct_iou = shift_summaries[0]
    wrong_shifts = list(range(1, len(TARGETS)))
    worst_loss_shift = max(wrong_shifts, key=lambda s: shift_summaries[s][0])
    worst_iou_shift = min(wrong_shifts, key=lambda s: shift_summaries[s][1])
    avg_wrong_iou = sum(shift_summaries[s][1] for s in wrong_shifts) / len(wrong_shifts)
    avg_wrong_loss = sum(shift_summaries[s][0] for s in wrong_shifts) / len(wrong_shifts)
    append_log(log_path, f"  correct target        : loss={correct_loss:.5f} iou={correct_iou:.3f}")
    append_log(log_path, f"  wrong target 평균      : loss={avg_wrong_loss:.5f} iou={avg_wrong_iou:.3f}")
    append_log(log_path, f"  wrong target 최악(loss): loss={shift_summaries[worst_loss_shift][0]:.5f} "
                          f"(shift={worst_loss_shift})")
    append_log(log_path, f"  wrong target 최악(IoU) : iou={shift_summaries[worst_iou_shift][1]:.3f} "
                          f"(shift={worst_iou_shift})")

    append_log(log_path, f"\n=== {tag}: 4x4 confusion matrix (true_target -> assigned_target, IoU) ===")
    header = "true\\assigned".ljust(18) + "".join(t[:10].ljust(14) for t in TARGETS)
    append_log(log_path, header)
    for t_true in TARGETS:
        row = t_true.ljust(18)
        for t_assigned in TARGETS:
            s = pair_stats.get((t_true, t_assigned))
            if s:
                _a, _b, iou = accuracy_scores_from_counts(*s[:5])
                row += f"{iou:.3f}(n={int(s[5])})".ljust(14)
            else:
                row += "-".ljust(14)
        append_log(log_path, row)
    return {"correct": (correct_loss, correct_iou), "avg_wrong": (avg_wrong_loss, avg_wrong_iou),
            "worst_loss_shift": worst_loss_shift, "worst_iou_shift": worst_iou_shift}


# ============================== main ==============================

def train_one_variant(variant_name, zero_appearance, zero_geometry, backbone, target_cache,
                       train_loader, val_loader, run_dir, log_path, model_seed):
    append_log(log_path, f"\n=== 모델 학습: {variant_name} (appearance={'X' if zero_appearance else 'O'}, "
                          f"geometry={'X' if zero_geometry else 'O'}, seed={model_seed}) ===")
    set_seed(model_seed)
    model = OcclusionMapModel(dino_embed_dim=backbone.embed_dim, num_layers=len(LAYERS)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta_pct=EARLY_STOP_MIN_DELTA_PCT, verbose=False)
    best_path = os.path.join(run_dir, f"model_{variant_name}_seed{model_seed}_best.pth")
    best_epoch, last_epoch_run = -1, -1
    best_loss_so_far = float("inf")
    val_loss_history = []

    for epoch in range(MAX_EPOCHS):
        train_loss, _a, _b, train_iou, _tp = run_epoch(
            backbone, model, train_loader, target_cache, optim=optim,
            desc=f"{variant_name} train {epoch}", zero_appearance=zero_appearance, zero_geometry=zero_geometry)
        val_loss, val_acc, val_bal, val_iou, _vp = run_epoch(
            backbone, model, val_loader, target_cache, optim=None,
            desc=f"{variant_name} val {epoch}", zero_appearance=zero_appearance, zero_geometry=zero_geometry)
        last_epoch_run = epoch
        val_loss_history.append(val_loss)

        is_true_best = val_loss < best_loss_so_far
        if is_true_best:
            best_loss_so_far = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)
            best_epoch = epoch
        stopper(val_loss)

        append_log(log_path,
                    f"[{variant_name}] epoch {epoch}: train_loss={train_loss:.5f} iou={train_iou:.3f} | "
                    f"val_loss={val_loss:.5f} acc={val_acc:.3f} bal_acc={val_bal:.3f} iou={val_iou:.3f}"
                    f"{'  <- best 저장' if is_true_best else ''}")
        if stopper.early_stop:
            append_log(log_path, f"[{variant_name}] early stop @ epoch {epoch} (best epoch {best_epoch})")
            break

    true_argmin_epoch = int(min(range(len(val_loss_history)), key=lambda i: val_loss_history[i]))
    report = {"best_epoch": best_epoch, "last_epoch_run": last_epoch_run,
              "saved_epoch": best_epoch, "saved_val_loss": best_loss_so_far,
              "true_argmin_epoch": true_argmin_epoch, "true_min_val_loss": min(val_loss_history)}
    return best_path, report


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, f"occlusion_stage2_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")

    manifest = build_shared_scene_manifest_3way(TARGETS, CLUTTER_TARGET, SCALE)
    manifest_hash = hashlib.md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:8]
    append_log(log_path, f"shared manifest(3way): {manifest['n_shared_scenes']}개 scene, "
                          f"split_seed={manifest['split_seed']}, hash={manifest_hash}")
    append_log(log_path, f"train={len(manifest['train_scenes'])} val={len(manifest['val_scenes'])} "
                          f"test={len(manifest['test_scenes'])}")

    train_ds = OcclusionDataset([(t, CLUTTER_TARGET, SCALE, manifest["train_scenes"]) for t in TARGETS])
    val_ds = OcclusionDataset([(t, CLUTTER_TARGET, SCALE, manifest["val_scenes"]) for t in TARGETS])
    test_ds = OcclusionDataset([(t, CLUTTER_TARGET, SCALE, manifest["test_scenes"]) for t in TARGETS])
    append_log(log_path, f"train samples={len(train_ds)} (skipped {train_ds.skipped}), "
                          f"val samples={len(val_ds)} (skipped {val_ds.skipped}), "
                          f"test samples={len(test_ds)} (skipped {test_ds.skipped})")

    coverage_ok = verify_coverage_zero_no_contribution()
    dilution_ok = verify_coverage_dilution_fix()
    append_log(log_path, f"coverage=0 무기여 자체 검증: {'PASS' if coverage_ok else 'FAIL'}")
    append_log(log_path, f"coverage 경계 희석 방지 자체 검증: {'PASS' if dilution_ok else 'FAIL'}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    backbone = DINOv3Backbone(variant="vitb16", layers=LAYERS, device=DEVICE)
    target_cache = precompute_occlusion_target_cache(backbone, TARGETS, DEVICE)

    all_seed_results = {}
    final_all_pass = True
    for seed in SEEDS:
        append_log(log_path, f"\n\n########## SEED {seed} ##########")
        ckpt_paths, checkpoint_reports = {}, {}
        for variant_name, (zero_app, zero_geo) in VARIANT_CONFIGS.items():
            path, report = train_one_variant(
                variant_name, zero_app, zero_geo, backbone, target_cache,
                train_loader, val_loader, run_dir, log_path, model_seed=seed)
            ckpt_paths[variant_name] = path
            checkpoint_reports[variant_name] = report

        # test set은 여기서 한 번만 사용(4x4 confusion matrix + 최종 비교). val은 checkpoint
        # 선택에만 썼으므로 이 숫자엔 val 기반 모델 선택의 낙관 편향이 섞이지 않는다.
        append_log(log_path, f"\n=== seed {seed}: test set 최종 평가(각 variant의 best checkpoint) ===")
        variant_test_results = {}
        for variant_name, (zero_app, zero_geo) in VARIANT_CONFIGS.items():
            model = OcclusionMapModel(dino_embed_dim=backbone.embed_dim, num_layers=len(LAYERS)).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_paths[variant_name])["model_state"])
            loss, acc, bal, iou, pooled_iou = run_epoch(
                backbone, model, test_loader, target_cache, optim=None,
                desc=f"{variant_name} test", zero_appearance=zero_app, zero_geometry=zero_geo)
            append_log(log_path, f"[{variant_name}] test: loss={loss:.5f} iou(sample평균)={iou:.3f} "
                                  f"iou(pooled,참고용)={pooled_iou:.3f} "
                                  f"(best_epoch={checkpoint_reports[variant_name]['best_epoch']})")
            variant_test_results[variant_name] = {"loss": loss, "iou": iou, "pooled_iou": pooled_iou, "model": model}

        # 3개 target-conditioned variant 전부 confusion matrix + per-sample 기록(이전엔 full만
        # 계산해서 toy_3<->packaged_food_2 혼동이 appearance/geometry 중 어디서 오는지 몰랐음).
        # 반드시 variant 자신의 zero_appearance/zero_geometry를 넘겨야 함 -- 이걸 빠뜨리면
        # appearance_only/geometry_only checkpoint가 학습 때와 다른(둘 다 켜진) 입력을 받게 되는
        # 실제 버그가 있었음(seed2 appearance_only: 기본 test IoU=0.427 vs 버그 있던 shift=0
        # IoU=0.282로 확인됨 -- shift=0은 target을 안 바꾸므로 두 값이 같아야 정상).
        shift0_mismatch = []
        all_rows = []
        cm_summaries = {}
        for variant_name in DIAGNOSE_VARIANTS:
            zero_app, zero_geo = VARIANT_CONFIGS[variant_name]
            model = variant_test_results[variant_name]["model"]
            pair_stats = {}
            shift_summaries = {}
            for shift in range(len(TARGETS)):
                rows = []
                loss, acc, bal, iou, pooled_iou = run_epoch(
                    backbone, model, test_loader, target_cache, optim=None,
                    desc=f"{variant_name} test shift={shift}", target_shift=shift,
                    zero_appearance=zero_app, zero_geometry=zero_geo,
                    per_pair_stats=pair_stats, per_sample_rows=rows,
                    row_tags={"seed": seed, "variant": variant_name, "shift": shift})
                shift_summaries[shift] = (loss, iou)
                all_rows.extend(rows)
            cm_summaries[variant_name] = report_confusion_matrix(
                shift_summaries, pair_stats, log_path, tag=f"seed {seed}, {variant_name}, TEST set")

            # invariant: shift=0(target 안 바꿈)은 위의 "test set 최종 평가"와 완전히 동일해야 함.
            main_loss = variant_test_results[variant_name]["loss"]
            main_iou = variant_test_results[variant_name]["iou"]
            shift0_loss, shift0_iou = shift_summaries[0]
            ok = abs(main_loss - shift0_loss) < 1e-4 and abs(main_iou - shift0_iou) < 1e-4
            shift0_mismatch.append((variant_name, ok, main_loss, shift0_loss, main_iou, shift0_iou))
            if not ok:
                append_log(log_path, f"  [FAIL] [{variant_name}] shift=0 != 기본 test: "
                                      f"loss {main_loss:.5f} vs {shift0_loss:.5f}, iou {main_iou:.3f} vs {shift0_iou:.3f}")

        csv_path = os.path.join(run_dir, f"per_sample_results_seed{seed}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        append_log(log_path, f"\nsample 단위 결과 저장: {csv_path} ({len(all_rows)}행)")

        # scene 단위(카메라 5장 평균) paired 비교. 반드시 (variant, true_target, scene_key)로
        # 묶어야 한다 -- 이전엔 (variant, scene_key)로만 묶어서 같은 scene_key를 공유하는 4개
        # target x 5 camera = 20개 sample이 "카메라 5장 평균"이라는 이름으로 섞여 있었음(scene_key가
        # 4개 target 전부에 공통이라 생기는 문제). target별로 먼저 집계한 뒤 target별/전체를 따로 본다.
        append_log(log_path, f"\n=== seed {seed}: scene 단위(target별 카메라 5장 평균) paired 비교, correct target(shift=0) ===")
        scene_means = {}
        for row in all_rows:
            if row["shift"] != 0:
                continue
            scene_means.setdefault((row["variant"], row["true_target"], row["scene_key"]), []).append(row["iou"])
        scene_means = {k: sum(v) / len(v) for k, v in scene_means.items()}
        scenes = sorted(manifest["test_scenes"])
        for a, b in [("full", "appearance_only"), ("full", "geometry_only"), ("appearance_only", "geometry_only")]:
            all_diffs = []
            for t in TARGETS:
                diffs = [scene_means[(a, t, sk)] - scene_means[(b, t, sk)] for sk in scenes]
                all_diffs.extend(diffs)
                mean_diff = sum(diffs) / len(diffs)
                std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
                n_a, n_b = sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)
                append_log(log_path, f"  [{t}] {a} - {b}: 평균={mean_diff:+.4f} std={std_diff:.4f} "
                                      f"(n={len(diffs)}) | {a} 우세 {n_a} / {b} 우세 {n_b} / 동률 {len(diffs)-n_a-n_b}")
            om = sum(all_diffs) / len(all_diffs)
            osd = (sum((d - om) ** 2 for d in all_diffs) / max(1, len(all_diffs) - 1)) ** 0.5
            append_log(log_path, f"  [전체(4 target x 30 scene=120)] {a} - {b}: 평균={om:+.4f} std={osd:.4f} (n={len(all_diffs)})")

        all_pass = check_invariants(manifest, train_ds, val_ds, test_ds,
                                      coverage_zero_contributes=coverage_ok,
                                      coverage_dilution_fixed=dilution_ok,
                                      checkpoint_reports=checkpoint_reports, log_path=log_path,
                                      shift0_mismatch=shift0_mismatch)
        final_all_pass = final_all_pass and all_pass

        append_log(log_path, f"\n=== seed {seed} 요약 (test set) ===")
        for v in VARIANT_CONFIGS:
            append_log(log_path, f"  {v.ljust(22)}: loss={variant_test_results[v]['loss']:.5f} "
                                  f"iou={variant_test_results[v]['iou']:.3f}")
        append_log(log_path, f"  invariant 통과 여부: {all_pass}")

        all_seed_results[seed] = {v: variant_test_results[v]["iou"] for v in VARIANT_CONFIGS}
        all_seed_results[seed]["confusion_matrix_summaries"] = cm_summaries
        all_seed_results[seed]["scene_means"] = scene_means

    append_log(log_path, f"\n\n=== 전체 요약 (seeds={SEEDS}) ===")
    for seed, res in all_seed_results.items():
        append_log(log_path, f"  seed {seed}: " + ", ".join(
            f"{v}={res[v]:.3f}" for v in VARIANT_CONFIGS))
    append_log(log_path, f"random split_seed={SPLIT_SEED}, manifest_hash={manifest_hash}, run_dir={run_dir}")
    append_log(log_path, f"invariant 전체(모든 seed) 통과 여부: {final_all_pass}")

    print(f"\n로그: {log_path}")
    return final_all_pass


if __name__ == "__main__":
    main()
