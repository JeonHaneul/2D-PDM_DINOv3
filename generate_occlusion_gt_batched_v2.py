"""generate_occlusion_gt_batched.py(V1, pose만 GPU batch, scene/camera는 Python 순차,
44,100pose x 20scene x 5camera=83초, GPU peak 0.87GB -- reference로 보존)의 scene까지
벡터화한 V2. camera는 이번에도 순차 유지(rasterizer는 이미 검증된 V1 그대로 재사용 --
정확도 위험을 새로 만들지 않기 위해 scene 비교 부분만 바꿈).

핵심 변화:
  - scene depth를 dict가 아니라 camera별 (S,H,W) GPU 텐서로 미리 적재
  - scene을 scene_batch_size(기본 8)개씩 묶어 broadcasting으로 한 번에 비교:
      occluded[B,Sc,H,W] = (scene_chunk[None] < depth_batch[:,None]) & ... & footprint[:,None]
  - N_occ를 dict-of-scene이 아니라 (S,H,W) GPU 텐서로 유지, accepted.T @ footprint.reshape(B,-1)
    matmul로 scene 전체를 한 번에 누적
  - ratio histogram도 GPU 텐서 카운터로 누적하고 체크포인트/종료 시점에만 .item()으로 내림
    (V1의 "매 scene마다 .cpu().numpy() 강제 동기화" 병목 제거)

정확성은 validate_v2_generator.py로 V1과 대조 검증(1,000 pose x 20 scene x 5 camera에서
N_all/N_occ/map/histogram/threshold 판정 일치)."""
import argparse
import hashlib
import json
import os
import re
import time

import cv2
import numpy as np
import torch

from mesh_cache import get_simplified_mesh, mesh_content_hash, _file_hash
from mesh_utils import extract_world_mesh
from depth_rasterizer import load_camera_metadata
from depth_rasterizer_gpu import render_depth_gpu_batch, render_depth_gpu, upload_mesh, yaw_translate_matrices

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")
DRAWER_USD = os.path.join(SRC_DIR, "asset", "drawer.usd")
PILOT_ROOT = os.path.join(SRC_DIR, "scene_generator", "output_occlusion_pilot")
CLUTTER_ROOT = os.path.join(SRC_DIR, "scene_generator", "output")
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occlusion_gt_output_batched_v2")

X_MIN, X_MAX, XY_STEP = -0.17, 0.17, 0.01
YAW_STEP_DEG = 30
Z_OFFSET, Z_LEVELS = 0.03, 3
OCCLUSION_THRESHOLD = 0.7
SIMPLIFY_TRIGGER_FACES = 50000
SIMPLIFIED_FACES = 10000
EPS = 1e-6
RATIOS = ["legacy", "corrected"]
# vectorized_object_occlusion.py(원본 legacy generator) L68 주석에 기록된 실측값 -- 16개 target
# 전부 포함. 이게 1순위 소스이고, 여기 없는(향후 추가될) target에 한해서만 아래 compute_base_z()
# 자동 산출을 fallback으로 쓴다(경고 로그 남김).
BASE_Z_TABLE = {
    "book_1": 0.01, "book_2": 0.01, "book_3": 0.01, "book_4": 0.01,
    "fruit_1": 0.01, "fruit_2": 0.01, "fruit_3": 0.01, "fruit_4": 0.05,
    "toy_1": 0.041, "toy_2": 0.01, "toy_3": 0.01, "toy_4": 0.01,
    "packaged_food_1": 0.043, "packaged_food_2": 0.04, "packaged_food_3": 0.04, "packaged_food_4": 0.028,
}
BASE_Z_CLEARANCE = 0.01  # BASE_Z_TABLE에 없는 target에 한해서만 쓰는 자동 산출 여유 간격


def compute_base_z(bbox_min_z: float, clearance: float = BASE_Z_CLEARANCE) -> float:
    """BASE_Z_TABLE에 없는(향후 추가될) target용 fallback. mesh 바닥(bbox_min_z, extract_world_mesh의
    로컬 원점 기준)이 서랍 바닥 위 clearance만큼 뜨도록 base_z를 계산. place_mesh()가 mesh 좌표에
    (x,y,z)를 그대로 더하는 방식이라, 실제 world 바닥 높이는 z(=base_z) + bbox_min_z가 된다 --
    이게 정확히 clearance가 되게 만드는 값. BASE_Z_TABLE의 실측값과는 최대 ~1mm 오차가 있었음
    (예: packaged_food_3 실측 0.04 vs 이 공식 0.0391) -- 실측값이 있으면 항상 그걸 우선한다."""
    return -bbox_min_z + clearance
FOOTPRINT_RULE_VERSION = "v1_legacy=fullmask_corrected=fullmask&valid_pos"
GENERATOR_VERSION = "v2_scene_vectorized"


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def build_pose_grid(base_z, max_poses=None, batch_size=1):
    x_values = np.round(np.arange(X_MIN, X_MAX + XY_STEP * 0.5, XY_STEP), 4)
    y_values = x_values.copy()
    yaw_values = np.arange(0, 360, YAW_STEP_DEG)
    z_values = [base_z + i * Z_OFFSET for i in range(Z_LEVELS)]
    poses = [{"x": float(x), "y": float(y), "yaw_deg": float(yaw), "z": float(z)}
              for x in x_values for y in y_values for yaw in yaw_values for z in z_values]
    if max_poses is not None:
        n_target = min(len(poses), ((max_poses + batch_size - 1) // batch_size) * batch_size)
        poses = poses[:n_target]
    return poses


def scale_mesh_bottom_center(verts, scale):
    if scale == 1.0:
        return verts
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    anchor = np.array([(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, bmin[2]])
    return anchor + scale * (verts - anchor)


def render_drawer_empty_depth_gpu(cam_meta, device="cuda"):
    drawer = extract_world_mesh(DRAWER_USD)
    out = {}
    for cam_name, cam in cam_meta["cameras"].items():
        d = render_depth_gpu(drawer["vertices"], drawer["faces"], cam_pos=cam["position"],
                              R=cam["R"], intr=cam["intrinsics"])
        out[cam_name] = torch.tensor(d, dtype=torch.float32, device=device)
    return out


def list_clutter_scenes_stacked(clutter_target, cam_names, max_scenes=None, device="cuda"):
    """scene depth를 camera별 (S,H,W) 텐서로 미리 쌓아서 반환. scene_keys 순서가 텐서의
    scene axis 순서와 정확히 대응됨(N_occ 저장 시 이 순서를 그대로 씀)."""
    scene_depth_dir = os.path.join(CLUTTER_ROOT, clutter_target, "scene", "depth")
    pattern = re.compile(r"scene(\d+)_env(\d+)_(\w+)\.npy")
    combos = sorted(set((m.group(1), m.group(2)) for f in os.listdir(scene_depth_dir)
                         if (m := pattern.match(f))))
    if max_scenes is not None and max_scenes < len(combos):
        # 정렬된 목록의 앞쪽 N개만 자르면(캡처 순서에 어떤 체계적 편향이 있을 경우) 대표성이
        # 떨어질 수 있어서, np.linspace로 전체 범위에 고르게 퍼진 인덱스를 뽑는다.
        idx = np.linspace(0, len(combos) - 1, max_scenes).astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    scene_keys = []
    per_cam_arrays = {cam: [] for cam in cam_names}
    for scene_idx, env_idx in combos:
        ok = True
        tmp = {}
        for cam in cam_names:
            p = os.path.join(scene_depth_dir, f"scene{scene_idx}_env{env_idx}_{cam}.npy")
            if not os.path.exists(p):
                ok = False
                break
            tmp[cam] = np.load(p).squeeze()
        if ok:
            scene_keys.append(f"scene{scene_idx}_env{env_idx}")
            for cam in cam_names:
                per_cam_arrays[cam].append(tmp[cam])
    scene_depth_stack = {cam: torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32, device=device)
                          for cam, arrs in per_cam_arrays.items()}  # (S,H,W) per camera
    return scene_keys, scene_depth_stack


def _camera_params_hash(cam_meta):
    payload = json.dumps({
        name: {"position": list(np.round(c["position"], 8)), "R": np.round(c["R"], 8).tolist(),
               "intrinsics": {k: (round(v, 6) if isinstance(v, (int, float)) else v)
                              for k, v in c["intrinsics"].items()}}
        for name, c in cam_meta["cameras"].items()
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def compute_config_fingerprint(args, usd_hash, mesh_hash, scene_keys, scene_depth_content_hash,
                                 cam_names, cam_meta, base_z, drawer_hash):
    payload = json.dumps({
        "target": args.target, "usd_hash": usd_hash, "mesh_hash": mesh_hash,
        "scale": args.scale, "clutter_target": args.clutter_target, "scene_keys": scene_keys,
        "scene_depth_content_hash": scene_depth_content_hash,
        "cam_names": cam_names, "camera_params_hash": _camera_params_hash(cam_meta),
        "batch_size": args.batch_size, "scene_batch_size": args.scene_batch_size, "base_z": base_z,
        "pose_grid": {"X_MIN": X_MIN, "X_MAX": X_MAX, "XY_STEP": XY_STEP,
                      "YAW_STEP_DEG": YAW_STEP_DEG, "Z_OFFSET": Z_OFFSET, "Z_LEVELS": Z_LEVELS},
        "occlusion_threshold": OCCLUSION_THRESHOLD, "drawer_hash": drawer_hash,
        "footprint_rule_version": FOOTPRINT_RULE_VERSION, "generator_version": GENERATOR_VERSION,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--clutter_target", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max_poses", type=int, default=None)
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--scene_batch_size", type=int, default=8)
    ap.add_argument("--checkpoint_every_batches", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base_z", type=float, default=None,
                     help="지정하면 BASE_Z_TABLE/자동 산출을 무시하고 이 값을 그대로 씀 -- "
                          "테스트에서 V1/V2에 동일 base_z를 명시적으로 강제할 때 사용.")
    args = ap.parse_args()

    usd_path = args.usd if os.path.isabs(args.usd) else os.path.join(SRC_DIR, args.usd)
    clutter_target = args.clutter_target or args.target
    usd_hash = _file_hash(usd_path)

    out_dir = os.path.join(OUT_ROOT, args.target, f"scale_{args.scale}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.npz")
    ckpt_meta_path = os.path.join(out_dir, "checkpoint_meta.json")

    pilot_dir = os.path.join(PILOT_ROOT, args.target)
    if not os.path.isdir(pilot_dir):
        pilot_dir = os.path.join(PILOT_ROOT, clutter_target)
    cam_meta = load_camera_metadata(os.path.join(pilot_dir, "camera_metadata.json"))
    cam_names = list(cam_meta["cameras"].keys())
    H, W = cam_meta["resolution"][1], cam_meta["resolution"][0]
    dev = torch.device("cuda")

    print(f"[1/6] {args.target} mesh 로드/캐시 (scale={args.scale})...")
    mesh = extract_world_mesh(usd_path)
    n_orig_faces = len(mesh["faces"])
    if n_orig_faces > SIMPLIFY_TRIGGER_FACES:
        v_use, f_use, was_cached = get_simplified_mesh(usd_path, args.target, SIMPLIFIED_FACES,
                                                          mesh["vertices"], mesh["faces"])
        mesh_faces_used = "simplified"
    else:
        v_use, f_use = mesh["vertices"], mesh["faces"]
        mesh_faces_used = "original"
    mesh_hash = mesh_content_hash(v_use, f_use)
    print(f"  {mesh_faces_used}: {len(f_use)} faces, hash={mesh_hash[:8]}")
    v_scaled = scale_mesh_bottom_center(v_use, args.scale)
    verts_t, faces_t = upload_mesh(v_scaled, f_use, device="cuda")

    print(f"[2/6] drawer depth (GPU)...")
    empty_depth_gpu = render_drawer_empty_depth_gpu(cam_meta)

    print(f"[3/6] clutter scene 로드 (GPU stacked, {clutter_target})...")
    scene_keys, scene_depth_stack = list_clutter_scenes_stacked(clutter_target, cam_names, args.max_scenes)
    S = len(scene_keys)
    print(f"  {S}개 scene x {len(cam_names)} camera (scene_batch_size={args.scene_batch_size})")

    # preflight: 렌더링 시작 전에 out_dir에 이번 run이 안 만들 leftover가 있는지 먼저 확인해서
    # 계산을 다 끝낸 뒤에야 실패하는 낭비를 막는다(아래 [6/6]에서 동일 검사를 완료 후에도 한 번
    # 더 함 -- 혹시 이 사이에 다른 프로세스가 out_dir을 건드릴 가능성까지 잡기 위해 둘 다 유지).
    _existing = {d for d in os.listdir(out_dir)
                 if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("scene")}
    _expected = {f"{sk}_{cam}" for sk in scene_keys for cam in cam_names}
    _unexpected_pre = _existing - _expected
    assert not _unexpected_pre, (
        f"[preflight] out_dir({out_dir})에 이번 run이 만들지 않을 leftover scene 디렉토리 "
        f"{len(_unexpected_pre)}개가 이미 있음(예: {sorted(_unexpected_pre)[:3]}) -- 렌더링 시작 전 중단. "
        f"이전 실행(다른 --max_scenes 등)의 결과로 보임, 확인 후 별도 폴더로 옮기고 재실행할 것.")
    scene_content_hash = hashlib.md5(b"".join(
        scene_depth_stack[cam].cpu().numpy().tobytes() for cam in cam_names)).hexdigest()

    if args.base_z is not None:
        base_z = args.base_z
    elif args.target in BASE_Z_TABLE:
        base_z = BASE_Z_TABLE[args.target]
    else:
        # BASE_Z_TABLE(16개 target 실측값)에 없는, 향후 추가될 target만 자동 산출로 fallback.
        base_z = compute_base_z(mesh["bbox_min"][2])
        print(f"  [base_z][WARN] '{args.target}'가 BASE_Z_TABLE에 없어 자동 산출: "
              f"bbox_min_z={mesh['bbox_min'][2]:.4f} -> base_z={base_z:.4f} "
              f"(BASE_Z_CLEARANCE={BASE_Z_CLEARANCE}) -- 반드시 실측값으로 검증/교체할 것")
    poses = build_pose_grid(base_z, args.max_poses, args.batch_size)
    n_batches = (len(poses) + args.batch_size - 1) // args.batch_size
    print(f"[4/6] pose grid: {len(poses)}개 -> {n_batches} batch")

    drawer_hash = _file_hash(DRAWER_USD)
    fingerprint = compute_config_fingerprint(args, usd_hash, mesh_hash, scene_keys, scene_content_hash,
                                              cam_names, cam_meta, base_z, drawer_hash)

    N_all = {r: {cam: torch.zeros((H, W), dtype=torch.float32, device=dev) for cam in cam_names} for r in RATIOS}
    N_occ = {r: {cam: torch.zeros((S, H, W), dtype=torch.float32, device=dev) for cam in cam_names} for r in RATIOS}
    hist_counters = {b: torch.zeros((), dtype=torch.int64, device=dev)
                      for b in ["lt_0.69", "0.69_0.70", "0.70_0.71", "ge_0.71"]}
    start_batch_idx = 0

    if args.resume and os.path.exists(ckpt_path) and os.path.exists(ckpt_meta_path):
        with open(ckpt_meta_path) as f:
            ckpt_meta = json.load(f)
        if ckpt_meta.get("fingerprint") != fingerprint:
            raise RuntimeError(f"checkpoint fingerprint 불일치! 실행 설정이 바뀌었을 가능성 -- 이어받지 않음.")
        start_batch_idx = ckpt_meta["batch_idx"]
        for b_name in hist_counters:
            hist_counters[b_name] = torch.tensor(ckpt_meta["ratio_hist"][b_name], device=dev)
        data = np.load(ckpt_path)
        for r in RATIOS:
            for cam in cam_names:
                N_all[r][cam] = torch.tensor(data[f"N_all_{r}_{cam}"], device=dev)
                N_occ[r][cam] = torch.tensor(data[f"N_occ_{r}_{cam}"], device=dev)
        print(f"[체크포인트 재개] batch_idx={start_batch_idx}/{n_batches}")

    print(f"[5/6] 배치 렌더링+scene-벡터화 누적 시작...")
    t0 = time.time()
    gpu_mem_peak = 0
    for b in range(start_batch_idx, n_batches):
        batch_poses = poses[b * args.batch_size: (b + 1) * args.batch_size]
        xs = np.array([p["x"] for p in batch_poses])
        ys = np.array([p["y"] for p in batch_poses])
        zs = np.array([p["z"] for p in batch_poses])
        yaws = np.array([p["yaw_deg"] for p in batch_poses])
        transforms = yaw_translate_matrices(xs, ys, zs, yaws)
        Bp = len(batch_poses)

        for cam_name, cam in cam_meta["cameras"].items():
            depth_batch = render_depth_gpu_batch(verts_t, faces_t, transforms, cam["position"],
                                                   cam["R"], cam["intrinsics"])  # (Bp,H,W)
            full_mask = depth_batch > 0
            ed = empty_depth_gpu[cam_name]
            valid_pos = (ed == 0) | (ed >= depth_batch)
            footprints = {"legacy": full_mask, "corrected": full_mask & valid_pos}
            scene_stack_cam = scene_depth_stack[cam_name]  # (S,H,W)

            for r, fp in footprints.items():
                n_px = fp.sum(dim=(-2, -1)).float()  # (Bp,)
                pose_valid = n_px > 0
                if not pose_valid.any():
                    continue
                N_all[r][cam_name] += fp.float().sum(dim=0)
                fp_flat = fp.reshape(Bp, -1).float()  # (Bp, H*W)

                for s0 in range(0, S, args.scene_batch_size):
                    s1 = min(s0 + args.scene_batch_size, S)
                    scene_chunk = scene_stack_cam[s0:s1]  # (Sc,H,W)
                    occluded = ((scene_chunk[None] < depth_batch[:, None]) &
                                (scene_chunk[None] != 0) & fp[:, None])  # (Bp,Sc,H,W)
                    occ_count = occluded.sum(dim=(-2, -1)).float()  # (Bp,Sc)
                    ratio = occ_count / n_px[:, None].clamp(min=1)  # (Bp,Sc)

                    if r == "corrected":
                        rv = ratio[pose_valid]  # (n_valid,Sc)
                        hist_counters["lt_0.69"] += (rv < 0.69).sum()
                        hist_counters["0.69_0.70"] += ((rv >= 0.69) & (rv < 0.70)).sum()
                        hist_counters["0.70_0.71"] += ((rv >= 0.70) & (rv < 0.71)).sum()
                        hist_counters["ge_0.71"] += (rv >= 0.71).sum()

                    accepted = (ratio >= OCCLUSION_THRESHOLD) & pose_valid[:, None]  # (Bp,Sc)
                    # accepted.T @ fp_flat -> (Sc, H*W): scene별로 accepted인 pose들의 footprint 합
                    contribution = accepted.float().T @ fp_flat  # (Sc, H*W)
                    N_occ[r][cam_name][s0:s1] += contribution.reshape(s1 - s0, H, W)

        if torch.cuda.is_available():
            gpu_mem_peak = max(gpu_mem_peak, torch.cuda.max_memory_allocated())

        if (b + 1) % 5 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            done = b + 1 - start_batch_idx
            eta = elapsed / done * (n_batches - b - 1) if done > 0 else 0
            print(f"  [batch {b+1}/{n_batches}, pose {min((b+1)*args.batch_size,len(poses))}/{len(poses)}] "
                  f"경과 {elapsed:.0f}s 예상잔여 {eta:.0f}s GPU메모리 peak={gpu_mem_peak/1e9:.2f}GB")

        if (b + 1) % args.checkpoint_every_batches == 0:
            _save_checkpoint(ckpt_path, ckpt_meta_path, N_all, N_occ, cam_names, b + 1, hist_counters, fingerprint)
            print(f"  [체크포인트 저장] batch_idx={b+1}")

    elapsed_total = time.time() - t0
    print(f"[5/6] 완료: {elapsed_total:.0f}초")
    _save_checkpoint(ckpt_path, ckpt_meta_path, N_all, N_occ, cam_names, n_batches, hist_counters, fingerprint)

    print(f"[6/6] map/coverage 저장...")
    cov_dir = os.path.join(out_dir, "_coverage")
    os.makedirs(cov_dir, exist_ok=True)
    N_all_np = {r: {cam: N_all[r][cam].cpu().numpy() for cam in cam_names} for r in RATIOS}
    for r in RATIOS:
        for cam in cam_names:
            n_all = N_all_np[r][cam]
            np.save(os.path.join(cov_dir, f"N_all_{r}_{cam}.npy"), n_all)
            cv2.imwrite(os.path.join(cov_dir, f"coverage_{r}_{cam}.png"), ((n_all > 0) * 255).astype(np.uint8))
    for cam in cam_names:
        union_cov = (N_all_np["legacy"][cam] > 0) | (N_all_np["corrected"][cam] > 0)
        np.save(os.path.join(cov_dir, f"union_coverage_{cam}.npy"), union_cov)

    for si, sk in enumerate(scene_keys):
        for cam in cam_names:
            scene_dir = os.path.join(out_dir, f"{sk}_{cam}")
            os.makedirs(scene_dir, exist_ok=True)
            for r in RATIOS:
                n_occ_np = N_occ[r][cam][si].cpu().numpy()
                n_all_np = N_all_np[r][cam]
                p_map = np.clip(n_occ_np / (n_all_np + EPS), 0.0, 1.0).astype(np.float32)

                # N_occ는 accepted(ratio>=threshold) pose 부분집합의 footprint 합, N_all은
                # 전체 valid pose의 footprint 합이므로 accepted ⊆ valid에 의해 N_occ<=N_all이
                # 픽셀별로 항상 성립해야 함(둘 다 0/1 값의 정수 카운트라 float32 오차 없이
                # 정확히 성립). 커버되지 않은 픽셀(N_all==0)은 확률이 정확히 0이어야 함.
                tag = f"{sk}/{cam}/{r}"
                assert np.isfinite(p_map).all(), f"[{tag}] probability_map에 NaN/Inf 존재"
                assert p_map.min() >= 0.0 and p_map.max() <= 1.0, f"[{tag}] probability_map이 [0,1] 범위 밖"
                assert (n_occ_np <= n_all_np).all(), f"[{tag}] N_occ > N_all인 픽셀 존재"
                zero_cov = n_all_np == 0
                assert not zero_cov.any() or (p_map[zero_cov] == 0).all(), \
                    f"[{tag}] coverage 없는(N_all==0) 픽셀의 probability가 0이 아님"

                np.save(os.path.join(scene_dir, f"map_{r}.npy"), p_map)
                cv2.imwrite(os.path.join(scene_dir, f"map_{r}.png"), (p_map * 255).astype(np.uint8))

    # 이 run이 의도한 scene_keys x cam_names개 디렉토리 외에 이전 실행(다른 --max_scenes 값 등)의
    # leftover가 out_dir에 남아있으면 안 됨 -- 실제로 15-scene pilot 이후 150-scene 정식 실행을
    # 돌렸을 때 두 linspace 샘플이 완전히 겹치지 않아 leftover가 섞이는 사고가 있었음(수동으로
    # 발견하고 정리함). 조용히 넘어가지 않고 즉시 중단시켜서 다음부터는 사람이 확인하게 한다.
    actual_scene_dirs = {d for d in os.listdir(out_dir)
                          if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("scene")}
    expected_scene_dirs = {f"{sk}_{cam}" for sk in scene_keys for cam in cam_names}
    unexpected = actual_scene_dirs - expected_scene_dirs
    assert not unexpected, (
        f"out_dir({out_dir})에 이번 run이 만들지 않은 scene 디렉토리 {len(unexpected)}개가 남아있음 "
        f"(예: {sorted(unexpected)[:3]}) -- 이전 실행(다른 --max_scenes 등)의 leftover로 보임. "
        f"자동으로 지우거나 무시하지 않음 -- 확인 후 별도 폴더로 옮기고 재실행할 것.")

    ratio_hist = {k: int(v.item()) for k, v in hist_counters.items()}
    total_ratio_samples = sum(ratio_hist.values())
    near_threshold_pct = (ratio_hist["0.69_0.70"] + ratio_hist["0.70_0.71"]) / total_ratio_samples * 100 \
        if total_ratio_samples > 0 else 0
    run_report = {
        "target": args.target, "scale": args.scale, "usd": usd_path, "mesh_faces_used": mesh_faces_used,
        "n_orig_faces": n_orig_faces, "n_used_faces": len(f_use), "batch_size": args.batch_size,
        "scene_batch_size": args.scene_batch_size, "clutter_target": clutter_target,
        "n_scenes": S, "n_poses": len(poses), "cameras": cam_names, "elapsed_sec": elapsed_total,
        "gpu_mem_peak_gb": gpu_mem_peak / 1e9, "ratio_histogram": ratio_hist,
        "near_0.7_threshold_pct": near_threshold_pct, "fingerprint": fingerprint,
        "generator_version": GENERATOR_VERSION,
    }
    _atomic_write_json(os.path.join(out_dir, "run_report.json"), run_report)

    print(f"\n=== 완료 보고 ===")
    print(f"  pose {len(poses)}개, scene {S}개, camera {len(cam_names)}개, scale={args.scale}")
    print(f"  소요시간: {elapsed_total:.0f}초, GPU 메모리 peak: {gpu_mem_peak/1e9:.2f}GB")
    print(f"  ratio 히스토그램: {ratio_hist}")
    print(f"  저장 위치: {out_dir}")


def _save_checkpoint(ckpt_path, ckpt_meta_path, N_all, N_occ, cam_names, batch_idx, hist_counters, fingerprint):
    arrays = {}
    for r in ["legacy", "corrected"]:
        for cam in cam_names:
            arrays[f"N_all_{r}_{cam}"] = N_all[r][cam].detach().cpu().numpy()
            arrays[f"N_occ_{r}_{cam}"] = N_occ[r][cam].detach().cpu().numpy()
    tmp_path = ckpt_path + ".tmp.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, ckpt_path)
    ratio_hist = {k: int(v.item()) for k, v in hist_counters.items()}
    _atomic_write_json(ckpt_meta_path, {"batch_idx": batch_idx, "ratio_hist": ratio_hist, "fingerprint": fingerprint})


if __name__ == "__main__":
    main()
