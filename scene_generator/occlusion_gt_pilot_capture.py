"""occlusion GT mesh-기반 계산 검증용 pilot capture. target_capture.py의 검증된 카메라/배치
패턴을 그대로 쓰되, 고정 pose 1개가 아니라 (x,y,z,yaw) 9개 pose x 5 카메라를 한 세션에서
전부 찍고, 카메라 실측 파라미터(camera_metadata.json)까지 저장한다.
기존 260714_data / scene_generator/output과 완전히 분리된 폴더에 저장(기존 데이터 안 건드림).

usage: python occlusion_gt_pilot_capture.py --target_name packaged_food_2 --headless
"""
import os
import argparse
import json
import numpy as np
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--target_name", type=str, required=True)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import cv2
from scipy.spatial.transform import Rotation as R
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
from pxr import UsdLux, UsdPhysics, UsdGeom, Usd, Gf
from semantics.schema.editor import PrimSemanticData

SCENE_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCENE_GEN_DIR)  # src/
ASSET_DIR = os.path.join(SRC_DIR, "asset")  # asset/<Category>/<instance>/*.usd
# GitHub 저장소가 src/ 아래에 있고 asset/이 저장소의 형제 폴더인 배치도 지원.
if not os.path.isdir(ASSET_DIR):
    ASSET_DIR = os.path.join(os.path.dirname(SRC_DIR), "asset")
WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "drawer.usd")
CATEGORIES = ["Book", "Toy", "Fruit", "Packaged_food"]

CAMERA_HEIGHT_OFFSET = 3.0
CAMERA_XY_OFFSET = 1.0
CAMERA_RESOLUTION = (640, 480)
CAMERA_CONFIGS = {
    "center": (0.0, 0.0),
    "left":   (-CAMERA_XY_OFFSET, 0.0),
    "right":  (CAMERA_XY_OFFSET, 0.0),
    "top":    (0.0, CAMERA_XY_OFFSET),
    "bottom": (0.0, -CAMERA_XY_OFFSET),
}

# vectorized_object_occlusion.py와 동일한 pose-grid 관례에서 뽑은 9개 pilot pose.
# BASE_Z는 target마다 다름(vectorized_object_occlusion.py 주석의 값을 그대로 사용).
BASE_Z_BY_TARGET = {
    "book_1": 0.01, "fruit_1": 0.01, "toy_3": 0.01, "packaged_food_2": 0.04,
}
Z_OFFSET = 0.03
base_z = BASE_Z_BY_TARGET.get(args.target_name, 0.01)
z_levels = [base_z, base_z + Z_OFFSET, base_z + 2 * Z_OFFSET]

POSES = [
    {"name": "center",        "x": 0.0,  "y": 0.0,  "yaw_deg": 0,   "z": z_levels[1]},
    {"name": "x_min",         "x": -0.17, "y": 0.0,  "yaw_deg": 0,   "z": z_levels[0]},
    {"name": "x_max",         "x": 0.17,  "y": 0.0,  "yaw_deg": 0,   "z": z_levels[0]},
    {"name": "y_min",         "x": 0.0,  "y": -0.17, "yaw_deg": 0,   "z": z_levels[0]},
    {"name": "y_max",         "x": 0.0,  "y": 0.17,  "yaw_deg": 0,   "z": z_levels[0]},
    {"name": "corner_1",      "x": -0.17, "y": -0.17, "yaw_deg": 90,  "z": z_levels[1]},
    {"name": "corner_2",      "x": 0.17,  "y": 0.17,  "yaw_deg": 90,  "z": z_levels[1]},
    {"name": "yaw90_center",  "x": 0.0,  "y": 0.0,  "yaw_deg": 90,  "z": z_levels[2]},
    {"name": "yaw150_offset", "x": -0.1, "y": 0.1,  "yaw_deg": 150, "z": z_levels[2]},
]


def discover_assets(usd_folder_dir, categories):
    assets = {}
    for category in categories:
        cat_dir = os.path.join(usd_folder_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for subdir in sorted(os.listdir(cat_dir)):
            subdir_path = os.path.join(cat_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().endswith((".usd", ".usdc")):
                    assets[subdir.lower()] = (os.path.splitext(f)[0], os.path.join(subdir_path, f), category)
                    break
    return assets


all_assets = discover_assets(ASSET_DIR, CATEGORIES)
target_key = args.target_name.lower()
if target_key not in all_assets:
    print(f"[오류] '{args.target_name}' 못 찾음. 사용 가능: {sorted(all_assets)}")
    simulation_app.close()
    exit(1)
target_usd_name, target_usd_path, target_category = all_assets[target_key]
print(f"[타겟] {args.target_name} -> {target_usd_name} ({target_category})")

OUT_DIR = os.path.join(SCENE_GEN_DIR, "output_occlusion_pilot", args.target_name)
rgb_dir, depth_dir, seg_dir = (os.path.join(OUT_DIR, d) for d in ("rgb", "depth", "seg"))
for d in (rgb_dir, depth_dir, seg_dir):
    os.makedirs(d, exist_ok=True)

world = World(physics_dt=1 / 120.0, backend="torch", device="cuda")
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000)
dl = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
dl.CreateIntensityAttr(1000)
dl.CreateAngleAttr(0.53)

add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace")


def make_static_collider(prim_path):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)


make_static_collider("/World/workspace")
add_reference_to_stage(usd_path=target_usd_path, prim_path="/World/target_object")
make_static_collider("/World/target_object")
PrimSemanticData(stage.GetPrimAtPath("/World/target_object")).add_entry("class", target_usd_name)


def look_at_rotation(cam_pos, target_pos=(0.0, 0.0, 0.0)):
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    rotation, _ = R.align_vectors([direction], [np.array([0.0, 0.0, -1.0])])
    q = rotation.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z


cameras = {}
seg_annotators = {}
for cam_name, offset in CAMERA_CONFIGS.items():
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    cam = Camera(prim_path=f"/World/camera_{cam_name}", position=cam_pos, resolution=CAMERA_RESOLUTION)
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    rp = rep.create.render_product(cam.prim_path, CAMERA_RESOLUTION)
    seg_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    seg_ann.attach([rp])
    cameras[cam_name] = cam
    seg_annotators[cam_name] = seg_ann

world.play()
for _ in range(10):
    world.step(render=True)

# 카메라 world orientation 세팅 (target_capture.py와 동일 패턴)
for cam_name, offset in CAMERA_CONFIGS.items():
    cam_view = XFormPrimView(f"/World/camera_{cam_name}")
    cam_pos = np.array([[offset[0], offset[1], CAMERA_HEIGHT_OFFSET]])
    quat_wxyz = look_at_rotation(cam_pos[0])
    import torch
    cam_view.set_world_poses(
        torch.tensor(cam_pos, dtype=torch.float32, device="cuda"),
        torch.tensor([quat_wxyz], dtype=torch.float32, device="cuda"),
    )
for _ in range(10):
    world.step(render=True)

# === 카메라 실측 metadata 저장 (하드코딩 방지) ===
camera_metadata = {"resolution": list(CAMERA_RESOLUTION), "cameras": {}}
for cam_name, cam in cameras.items():
    pos, quat = XFormPrimView(f"/World/camera_{cam_name}").get_world_poses()
    camera_metadata["cameras"][cam_name] = {
        "position": pos[0].cpu().numpy().tolist() if hasattr(pos[0], "cpu") else list(pos[0]),
        "orientation_wxyz": quat[0].cpu().numpy().tolist() if hasattr(quat[0], "cpu") else list(quat[0]),
        "focal_length": float(cam.get_focal_length()),
        "horizontal_aperture": float(cam.get_horizontal_aperture()),
        "vertical_aperture": float(cam.get_vertical_aperture()),
        "clipping_range": list(cam.get_clipping_range()),
    }
with open(os.path.join(OUT_DIR, "camera_metadata.json"), "w") as f:
    json.dump(camera_metadata, f, indent=2)
print(f"[camera_metadata.json 저장] {camera_metadata['cameras']['center']}")

# === target Xform ops 준비 ===
# 중요: add_reference_to_stage()가 asset의 metersPerUnit이 메인 stage(보통 1.0=미터)와 다르면
# 자동으로 보정용 Scale xformOp를 끼워넣는다(vectorized_object_occlusion.py에서 이미 겪은 문제와
# 동일). 이걸 모르고 ClearXformOpOrder()로 싹 지워버리면 (packaged_food_2처럼 metersPerUnit=1.0인
# asset은 문제없지만) book_1/fruit_1/toy_3처럼 metersPerUnit=0.01인 asset은 보정 scale이 날아가서
# 물체가 100배 크게 렌더링된다(실측: book_1 실루엣이 화면 100% 차지). 기존 scale op를 먼저
# 백업해뒀다가 그대로 복원하고, translate 값도 이 스케일로 나눠서 넣어야 한다.
target_prim = stage.GetPrimAtPath("/World/target_object")
xf = UsdGeom.Xformable(target_prim)

unit_scale = 1.0
scale_backups = []
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
        parts = op.GetOpName().split(":")
        suffix = parts[2] if len(parts) > 2 else ""
        type_str = str(op.GetAttr().GetTypeName())
        prec = UsdGeom.XformOp.PrecisionFloat if "float" in type_str else UsdGeom.XformOp.PrecisionDouble
        val = op.Get()
        scale_backups.append((suffix, val, prec))
        if val is not None:
            unit_scale = float(val[0])
print(f"[unit_scale 감지] {unit_scale} (1.0이 아니면 자동 삽입된 단위 보정 scale이 있었다는 뜻)")

xf.ClearXformOpOrder()
for suffix, val, prec in scale_backups:
    s_op = xf.AddScaleOp(prec, opSuffix=suffix)
    if val is not None:
        s_op.Set(val)
t_op = xf.AddTranslateOp()
o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

mapping_saved = [False]
poses_record = []

for pose_idx, pose in enumerate(POSES):
    yaw_rad = np.radians(pose["yaw_deg"])
    rot = R.from_euler("z", yaw_rad)
    q = rot.as_quat()  # x,y,z,w
    # scale op가 translate보다 먼저 적용되므로(로컬 좌표계가 이미 scale된 상태), world 좌표를
    # 그대로 넣으면 unit_scale배만큼 더 멀리 가버림 -- unit_scale로 나눠서 보정.
    t_op.Set(Gf.Vec3d(pose["x"] / unit_scale, pose["y"] / unit_scale, pose["z"] / unit_scale))
    o_op.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))

    for _ in range(10):
        world.step(render=True)

    poses_record.append({"pose_idx": pose_idx, **pose})

    for cam_name, cam in cameras.items():
        base = f"{pose_idx:02d}_{pose['name']}_{cam_name}"
        rgb = cam.get_rgb()
        if rgb is not None:
            cv2.imwrite(os.path.join(rgb_dir, f"{base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth = cam.get_depth()
        if depth is not None:
            np.save(os.path.join(depth_dir, f"{base}.npy"), depth)

        seg_data = seg_annotators[cam_name].get_data()
        if not seg_data or "data" not in seg_data:
            continue
        seg_ids = seg_data["data"]
        if seg_ids.ndim == 3:
            seg_ids = seg_ids[:, :, 0]
        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        seg_color = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)
        target_color = (30, 30, 220)  # BGR, 고정색 하나만 쓰므로 색상충돌 걱정 없음
        found = False
        # asset마다 semantic label 표기 방식이 달라서("target_object" 경로 문자열이 아예 안 뜨는
        # 경우가 있음 -- toy_3에서 실제로 발생) "target_object"/usd_name 둘 다 시도하고, 그래도
        # 안 잡히면 "이 scene엔 target 하나뿐"이라는 전제로 workspace/BACKGROUND/UNLABELLED가
        # 아닌 나머지 id를 전부 target으로 간주(fallback).
        candidate_uids = []
        for uid in np.unique(seg_ids):
            label = id_to_labels.get(str(int(uid)), "")
            label_lower = label.lower()
            if "target_object" in label_lower or target_usd_name.lower() in label_lower:
                seg_color[seg_ids == uid] = target_color
                found = True
            elif "workspace" not in label_lower and label != "BACKGROUND":
                # UNLABELLED도 후보에 포함 -- toy_3처럼 nested mesh 구조라 semantic class가 leaf
                # prim까지 전파되지 않아 target이 UNLABELLED로 잡히는 경우가 실제로 있었음.
                candidate_uids.append(uid)
        if not found and candidate_uids:
            for uid in candidate_uids:
                seg_color[seg_ids == uid] = target_color
            found = True
        cv2.imwrite(os.path.join(seg_dir, f"{base}.png"), seg_color)
        if found and not mapping_saved[0]:
            with open(os.path.join(OUT_DIR, "mapping.json"), "w") as f:
                json.dump({"target_usd_name": target_usd_name, "target_color_bgr": list(target_color)}, f, indent=2)
            mapping_saved[0] = True

    print(f"  [{pose_idx+1}/{len(POSES)}] {pose['name']} 완료")

with open(os.path.join(OUT_DIR, "poses.json"), "w") as f:
    json.dump(poses_record, f, indent=2)

print(f"\n=== 완료: {len(POSES)} poses x {len(CAMERA_CONFIGS)} cameras = {len(POSES)*len(CAMERA_CONFIGS)}장 ===")
print(f"저장 위치: {OUT_DIR}")

world.stop()
simulation_app.close()
