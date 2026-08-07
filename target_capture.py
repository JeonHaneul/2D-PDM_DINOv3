"""--target_name을 주면 그 하나만(이미 촬영했어도 강제로) 다시 찍고, 안 주면 미촬영 target 전체를 처리.

usage: python target_capture.py [--target_name book_1] [--force] [--headless]

기본 폴더 구조:
  src/
  ├── 2D-PDM_DINOv3/        # 이 파일
  ├── asset/
  └── scene_generator/output/

구조가 다르면 --asset_dir와 --output_root로 경로를 지정한다.
"""
import os
import argparse
import json
import numpy as np
from isaacsim import SimulationApp

# ==============================================================================
# 0. Argument Parsing (SimulationApp 시작 전에 파싱)
# ==============================================================================
parser = argparse.ArgumentParser(description="Target Object Capture (5-camera, single fixed pose)")
parser.add_argument("--target_name",  type=str, default=None, help="이 target 하나만 처리 (폴더 이름, 예: book_1)")
parser.add_argument("--force",        action="store_true",     help="이미 촬영된 target도 다시 촬영")
parser.add_argument("--headless",     action="store_true",     help="GUI 없이 실행")
parser.add_argument("--list_objects", action="store_true",     help="사용 가능한 오브젝트 목록 출력 후 종료")
parser.add_argument("--z",            type=float, default=None, help="스폰 z 높이 (m). 안 주면 TARGET_Z 기본값 사용")
parser.add_argument("--asset_dir",    type=str, default=None, help="asset 폴더 경로 (기본: ../asset)")
parser.add_argument("--output_root",  type=str, default=None,
                    help="촬영 출력 루트 (기본: ../scene_generator/output)")

args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

# ==============================================================================
# Isaac Sim imports (SimulationApp 시작 후에만 가능)
# ==============================================================================
import torch
import cv2
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
from pxr import UsdLux, UsdPhysics, Usd
from semantics.schema.editor import PrimSemanticData

# ==============================================================================
# 1. 설정
# ==============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(PROJECT_DIR)
ASSET_DIR = os.path.abspath(args.asset_dir) if args.asset_dir else os.path.join(SRC_ROOT, "asset")
OUTPUT_ROOT = (os.path.abspath(args.output_root) if args.output_root
               else os.path.join(SRC_ROOT, "scene_generator", "output"))
WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "drawer.usd")

CATEGORIES = ["Book", "Toy", "Fruit", "Packaged_food"]

CAMERA_HEIGHT_OFFSET = 3.0
CAMERA_XY_OFFSET     = 1.0
CAMERA_RESOLUTION    = (640, 480)
CAMERA_CONFIGS = {
    "center": {"offset": (0.0,               0.0)},
    "left":   {"offset": (-CAMERA_XY_OFFSET, 0.0)},
    "right":  {"offset": ( CAMERA_XY_OFFSET, 0.0)},
    "top":    {"offset": (0.0,               CAMERA_XY_OFFSET)},
    "bottom": {"offset": (0.0,              -CAMERA_XY_OFFSET)},
}

# target을 서랍 "가운데"(x=0, y=0)에, TARGET_Z 높이에 그대로 정적으로 스폰 (물리 낙하 없음).
# --z로 실행 시점에 덮어쓸 수 있음.
DROP_X, DROP_Y = 0.0, 0.0
TARGET_Z       = args.z if args.z is not None else 0.05
RENDER_STABILIZE_STEPS = 10  # 물리 안정화가 아니라 렌더링 반영을 위한 최소 대기

FRAME_ID = "000000"  # 고정 자세 1개뿐이므로 항상 같은 frame_id 사용 (target_utils.discover_target_frame_id와 호환)

# ==============================================================================
# 2. Asset 탐색 (vectorized_object_occlusion.py와 동일한 2단계 구조 스캔)
# ==============================================================================
def discover_assets(usd_folder_dir, categories, extensions=(".usd", ".usdc")):
    """asset/<Category>/<subdir>/<file>.usd(c) -> {folder_name(소문자): (usd_name, usd_path, category)}"""
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
                if f.lower().endswith(extensions):
                    usd_name = os.path.splitext(f)[0]
                    assets[subdir.lower()] = (usd_name, os.path.join(subdir_path, f), category)
                    break
    return assets

all_assets = discover_assets(ASSET_DIR, CATEGORIES)

if args.list_objects:
    print("\n=== 사용 가능한 오브젝트 목록 ===")
    for folder_name, (usd_name, _, category) in sorted(all_assets.items()):
        print(f"  --target_name {folder_name}  →  {usd_name} ({category})")
    print("================================\n")
    simulation_app.close()
    exit(0)

# ==============================================================================
# 3. 처리할 target 목록 결정
# ==============================================================================
def already_captured(folder_name: str) -> bool:
    mapping_path = os.path.join(OUTPUT_ROOT, folder_name, "target", "mapping.json")
    return os.path.isfile(mapping_path)

if args.target_name:
    key = args.target_name.lower()
    if key not in all_assets:
        print(f"\n오류: '{args.target_name}'을(를) asset 폴더에서 찾을 수 없습니다.")
        print(f"--list_objects 옵션으로 목록을 확인하세요.")
        simulation_app.close()
        exit(1)
    targets_to_process = [key]
else:
    targets_to_process = [
        name for name in sorted(all_assets)
        if args.force or not already_captured(name)
    ]

if not targets_to_process:
    print("\n촬영이 필요한 target이 없습니다 (모두 이미 촬영됨, --force로 강제 재촬영 가능).\n")
    simulation_app.close()
    exit(0)

print(f"\n처리할 target ({len(targets_to_process)}개): {targets_to_process}\n")

# ==============================================================================
# 4. World 설정 (환경 1개만 -- 격자 스캔이 없으니 클론/병렬화 불필요)
# ==============================================================================
world = World(physics_dt=1 / 120.0, backend="torch", device="cuda")
physics_context = world.get_physics_context()
physics_context.set_solver_type("TGS")
physics_context.enable_ccd(True)

GroundPlane(prim_path="/World/GroundPlane", z_position=0, color=np.array([1.0, 1.0, 1.0]))

stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000)
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateAngleAttr(0.53)

def make_static_collider(prim_path: str):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace")
make_static_collider("/World/workspace")

# ==============================================================================
# 5. 카메라 5개 (환경 1개뿐이라 클론 없이 바로 최종 위치에 생성)
# ==============================================================================
def look_at_rotation(cam_pos, target_pos=(0.0, 0.0, 0.0)):
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    rotation, _ = R.align_vectors([direction], [np.array([0.0, 0.0, -1.0])])
    q = rotation.as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

cameras = {}
seg_annotators = {}
for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])

    # Camera() 생성자에 orientation을 바로 주면 실제 촬영 방향이 어긋난다 -- 대신
    # vectorized_object_occlusion.py처럼 일단 position만 주고 initialize한 뒤,
    # XFormPrimView.set_world_poses()로 orientation을 따로 세팅해야 정확히 반영된다.
    cam = Camera(prim_path=f"/World/camera_{cam_name}", position=cam_pos, resolution=CAMERA_RESOLUTION)
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()

    quat_wxyz = look_at_rotation(cam_pos)
    cam_view = XFormPrimView(f"/World/camera_{cam_name}")
    pos_tensor = torch.tensor(cam_pos, dtype=torch.float32, device="cuda").unsqueeze(0)
    orient_tensor = torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda").unsqueeze(0)
    cam_view.set_world_poses(pos_tensor, orient_tensor)

    rp = rep.create.render_product(cam.prim_path, CAMERA_RESOLUTION)
    seg_ann = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
    seg_ann.attach([rp])

    cameras[cam_name] = cam
    seg_annotators[cam_name] = seg_ann

# ==============================================================================
# 6. Simulation 시작
# ==============================================================================
world.play()
for _ in range(20):
    world.step(render=True)

# ==============================================================================
# 7. target 하나 촬영
# ==============================================================================
def clear_current_target():
    """이전 target prim을 stage에서 제거 (다음 target 로드 전 정리)"""
    if stage.GetPrimAtPath("/World/target_object").IsValid():
        stage.RemovePrim("/World/target_object")

def capture_target(folder_name: str, usd_name: str, usd_path: str, category: str):
    print(f"[{folder_name}] {usd_name} ({category}) 촬영 시작...")

    output_base = os.path.join(OUTPUT_ROOT, folder_name, "target")
    rgb_dir   = os.path.join(output_base, "rgb")
    depth_dir = os.path.join(output_base, "depth")
    seg_dir   = os.path.join(output_base, "seg")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # --- target 로드 + 가운데(TARGET_Z 높이)에 정적으로 배치 (물리 낙하 없음) ---
    clear_current_target()
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/target_object")
    prim = stage.GetPrimAtPath("/World/target_object")
    PrimSemanticData(prim).add_entry("class", usd_name)
    make_static_collider("/World/target_object")

    SingleXFormPrim(
        prim_path="/World/target_object",
        name=f"target_{folder_name}",
        position=np.array([DROP_X, DROP_Y, TARGET_Z]),
    )

    for _ in range(RENDER_STABILIZE_STEPS):
        world.step(render=True)

    # --- 5카메라 캡처 ---
    class_colors: dict = {}
    scene_classes: dict = {}

    for cam_name, cam in cameras.items():
        filename_base = f"{FRAME_ID}_{cam_name}"

        rgb = cam.get_rgb()
        if rgb is not None:
            cv2.imwrite(os.path.join(rgb_dir, f"{filename_base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        depth = cam.get_depth()
        if depth is not None:
            np.save(os.path.join(depth_dir, f"{filename_base}.npy"), depth)

        seg_data = seg_annotators[cam_name].get_data()
        if seg_data is None or not isinstance(seg_data, dict) or "data" not in seg_data:
            continue
        seg_ids = seg_data["data"]
        if seg_ids.ndim == 3:
            seg_ids = seg_ids[:, :, 0]

        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        seg_color = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)

        for uid in np.unique(seg_ids):
            prim_label = id_to_labels.get(str(int(uid)), "")
            if not prim_label or "target_object" not in prim_label:
                continue
            if usd_name not in class_colors:
                idx = len(class_colors)
                hue = (idx * 23) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
                class_colors[usd_name] = color.tolist()
            color = class_colors[usd_name]
            seg_color[seg_ids == uid] = color
            scene_classes[usd_name] = {"color_bgr": color}

        cv2.imwrite(os.path.join(seg_dir, f"{filename_base}.png"), seg_color)

    if scene_classes:
        mapping_data = {
            "target_folder_name": folder_name,
            "target_usd_name":    usd_name,
            "category":           category,
            "classes":            scene_classes,
        }
        with open(os.path.join(output_base, "mapping.json"), "w") as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        print(f"  → 저장 완료: {output_base}")
    else:
        print(f"  [WARN] '{folder_name}' segmentation에서 안 보임 -- mapping.json 저장 안 됨 (재확인 필요)")


try:
    for i, folder_name in enumerate(targets_to_process):
        usd_name, usd_path, category = all_assets[folder_name]
        print(f"--- {i + 1}/{len(targets_to_process)} ---")
        capture_target(folder_name, usd_name, usd_path, category)

    clear_current_target()
    print(f"\n=== 전체 완료 ({len(targets_to_process)}개) ===\n")

finally:
    world.stop()
    simulation_app.close()
