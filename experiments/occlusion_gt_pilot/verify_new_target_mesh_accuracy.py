"""신규 target mesh-vs-real depth/seg 정량 검증, v2 -- single-pose-across-cameras 방법론으로 수정.

v1(camera마다 독립적으로 pose를 재탐색)의 문제: 물리적으로 5개 카메라는 같은 target을 같은
pose에서 찍은 것이어야 하는데, camera별로 독립 탐색하면 camera extrinsic 오차를 "이 카메라에
맞는 다른 pose"로 보상해버려도 IoU가 높게 나올 수 있다 -- 실제로 toy_2(구형 물체라 yaw가
거의 안 보임)에서 camera마다 다른 yaw(270/300/30도)가 나와 이 위험이 실재함을 확인했음.

v2: center camera에서만 pose를 탐색해서 고정한 뒤, 나머지 4개 카메라는 그 pose 그대로
평가한다(재탐색 없음) -- 이래야 "5개 카메라 캘리브레이션이 서로 일관되게 정확한가"를
실제로 검증하게 된다. 모든 target(신규 10개 전부)에 동일하게 적용."""
import json
import os
import sys

import cv2
import numpy as np
import torch

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")
DINOV3_DIR = os.path.join(SRC_DIR, "2D-PDM_DINOv3")
sys.path.insert(0, DINOV3_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_utils import extract_world_mesh
from depth_rasterizer import load_camera_metadata
from depth_rasterizer_gpu import render_depth_gpu_batch, upload_mesh, yaw_translate_matrices
from generate_occlusion_gt_batched_v2 import BASE_Z_TABLE, X_MIN, X_MAX, XY_STEP, YAW_STEP_DEG, Z_OFFSET, Z_LEVELS
from target_catalog_manifest import usd_path

PILOT_ROOT = os.path.join(SRC_DIR, "scene_generator", "output_occlusion_pilot")
TARGET_ROOT = os.path.join(SRC_DIR, "260714_data", "target")
CAMS = ["center", "left", "right", "top", "bottom"]
SEARCH_CAM = "center"  # pose는 이 카메라에서만 찾고, 나머지는 이 pose를 그대로 재사용(재탐색 없음)
BATCH_SIZE = 512

TARGETS = {t: usd_path(t) for t in
           ["book_2", "book_3", "book_4", "fruit_2", "fruit_3", "fruit_4",
            "toy_2", "toy_4", "packaged_food_3", "packaged_food_4"]}


def build_pose_grid(base_z):
    x_values = np.round(np.arange(X_MIN, X_MAX + XY_STEP * 0.5, XY_STEP), 4)
    y_values = x_values.copy()
    yaw_values = np.arange(0, 360, YAW_STEP_DEG)
    z_values = [base_z + i * Z_OFFSET for i in range(Z_LEVELS)]
    xs, ys, zs, yaws = [], [], [], []
    for x in x_values:
        for y in y_values:
            for yaw in yaw_values:
                for z in z_values:
                    xs.append(x); ys.append(y); zs.append(z); yaws.append(yaw)
    return np.array(xs), np.array(ys), np.array(zs), np.array(yaws)


def color_mask(seg_bgr, color_bgr):
    return (seg_bgr[:, :, 0] == color_bgr[0]) & (seg_bgr[:, :, 1] == color_bgr[1]) & (seg_bgr[:, :, 2] == color_bgr[2])


def search_best_pose(cam, real_mask_t, real_depth_t, verts_t, faces_t, xs, ys, zs, yaws):
    n_poses = len(xs)
    best_score, best_idx = float("inf"), -1
    for i0 in range(0, n_poses, BATCH_SIZE):
        i1 = min(i0 + BATCH_SIZE, n_poses)
        transforms = yaw_translate_matrices(xs[i0:i1], ys[i0:i1], zs[i0:i1], yaws[i0:i1])
        depth_batch = render_depth_gpu_batch(verts_t, faces_t, transforms, cam["position"], cam["R"], cam["intrinsics"])
        pred_mask = depth_batch > 0
        diff = (depth_batch - real_depth_t[None]).abs()
        masked_mae = torch.where(real_mask_t[None], diff, torch.zeros_like(diff)).sum(dim=(-2, -1)) / real_mask_t.sum().clamp(min=1)
        overlap = (pred_mask & real_mask_t[None]).sum(dim=(-2, -1)).float()
        valid = overlap > (real_mask_t.sum().float() * 0.5)
        score = torch.where(valid, masked_mae, torch.full_like(masked_mae, 1e9))
        batch_best = int(torch.argmin(score).item())
        if score[batch_best].item() < best_score:
            best_score, best_idx = score[batch_best].item(), i0 + batch_best
    return best_idx


def evaluate_at_pose(cam, pose, verts_t, faces_t, real_mask_t, real_depth_t):
    transforms = yaw_translate_matrices(np.array([pose["x"]]), np.array([pose["y"]]),
                                          np.array([pose["z"]]), np.array([pose["yaw"]]))
    depth_pred = render_depth_gpu_batch(verts_t, faces_t, transforms, cam["position"], cam["R"], cam["intrinsics"])[0]
    pred_mask = depth_pred > 0
    inter = (pred_mask & real_mask_t).sum().item()
    union = (pred_mask | real_mask_t).sum().item()
    iou = inter / union if union > 0 else 0.0
    common = pred_mask & real_mask_t
    if common.sum().item() > 0:
        resid = depth_pred[common] - real_depth_t[common]
        mae_mm = resid.abs().mean().item() * 1000
        rmse_mm = (resid ** 2).mean().sqrt().item() * 1000
        bias_mm = resid.mean().item() * 1000
    else:
        mae_mm = rmse_mm = bias_mm = float("nan")
    return {"iou": iou, "mae_mm": mae_mm, "rmse_mm": rmse_mm, "bias_mm": bias_mm}


def main():
    results = {}
    for target, usd in TARGETS.items():
        print(f"\n=== {target} ===")
        base_z = BASE_Z_TABLE[target]
        mesh = extract_world_mesh(usd)
        verts_t, faces_t = upload_mesh(mesh["vertices"], mesh["faces"])
        xs, ys, zs, yaws = build_pose_grid(base_z)

        pilot_dir = os.path.join(PILOT_ROOT, target)
        if not os.path.isdir(pilot_dir):
            pilot_dir = os.path.join(PILOT_ROOT, "packaged_food_1")
        cam_meta = load_camera_metadata(os.path.join(pilot_dir, "camera_metadata.json"))

        with open(os.path.join(TARGET_ROOT, target, "mapping.json")) as f:
            mapping = json.load(f)
        usd_name = mapping["target_usd_name"]
        color_bgr = tuple(mapping["classes"][usd_name]["color_bgr"])
        rgb_dir = os.path.join(TARGET_ROOT, target, "rgb")
        frame_id = sorted(f for f in os.listdir(rgb_dir) if f.endswith("_center.png"))[0].split("_")[0]

        def load_real(cam_name):
            depth = np.load(os.path.join(TARGET_ROOT, target, "depth", f"{frame_id}_{cam_name}.npy")).squeeze()
            seg = cv2.imread(os.path.join(TARGET_ROOT, target, "seg", f"{frame_id}_{cam_name}.png"))
            mask = color_mask(seg, color_bgr)
            return torch.tensor(mask, device="cuda"), torch.tensor(depth.astype(np.float32), device="cuda")

        search_mask_t, search_depth_t = load_real(SEARCH_CAM)
        if search_mask_t.sum().item() == 0:
            print(f"  [SKIP] {SEARCH_CAM} 카메라에 target이 안 보여서 pose 탐색 불가")
            continue
        best_idx = search_best_pose(cam_meta["cameras"][SEARCH_CAM], search_mask_t, search_depth_t,
                                      verts_t, faces_t, xs, ys, zs, yaws)
        if best_idx < 0:
            print(f"  [FAIL] {SEARCH_CAM}에서 겹치는 pose를 못 찾음")
            continue
        fixed_pose = {"x": float(xs[best_idx]), "y": float(ys[best_idx]),
                      "z": float(zs[best_idx]), "yaw": float(yaws[best_idx])}
        print(f"  단일 고정 pose({SEARCH_CAM}에서 탐색) = {fixed_pose} -- 5개 카메라 전부 이 pose로만 평가(재탐색 없음)")

        cam_results = {}
        for cam_name in CAMS:
            real_mask_t, real_depth_t = load_real(cam_name)
            if real_mask_t.sum().item() == 0:
                print(f"  [{cam_name}] 실측 mask가 비어있음 -- 스킵")
                continue
            metrics = evaluate_at_pose(cam_meta["cameras"][cam_name], fixed_pose, verts_t, faces_t, real_mask_t, real_depth_t)
            metrics["pose"] = fixed_pose
            cam_results[cam_name] = metrics
            print(f"  [{cam_name}] (고정 pose) IoU={metrics['iou']:.4f} depth_MAE={metrics['mae_mm']:.2f}mm "
                  f"RMSE={metrics['rmse_mm']:.2f}mm bias={metrics['bias_mm']:+.2f}mm")

        # invariant: 5개 카메라 전부 동일한 pose를 썼는지(구조상 보장되지만 명시적으로 확인)
        assert all(v["pose"] == fixed_pose for v in cam_results.values()), "카메라마다 다른 pose가 쓰임 -- 버그"
        results[target] = cam_results

    print("\n\n=== 요약 (단일 고정 pose, 5개 카메라 재탐색 없음) ===")
    for target, cam_results in results.items():
        ious = [v["iou"] for v in cam_results.values()]
        maes = [v["mae_mm"] for v in cam_results.values() if not np.isnan(v["mae_mm"])]
        if ious:
            print(f"  {target}: IoU range={min(ious):.3f}~{max(ious):.3f}  depth_MAE range={min(maes):.2f}~{max(maes):.2f}mm "
                  f"({len(ious)}/{len(CAMS)} camera)")
        else:
            print(f"  {target}: 검증 가능한 카메라 없음 -- FAIL")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_target_mesh_verification_v2_singlepose.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
