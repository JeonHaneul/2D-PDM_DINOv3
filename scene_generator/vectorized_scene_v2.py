import os
import argparse
import json
import random
import time
import torch
import numpy as np
from isaacsim import SimulationApp


# ==============================================================================
# 0. Argument Parsing (SimulationApp 시작 전에 파싱)
# ==============================================================================
parser = argparse.ArgumentParser(description="Isaac Sim Scene Generator with Similarity-based Spawning")

parser.add_argument(
    "--target",
    type=str,
    default=None,
    help="타겟 오브젝트 이름 (예: book_1, fruit_2). 지정하면 유사도 기반 순차 스폰"
)
parser.add_argument(
    "--num_scenes",
    type=int,
    default=1,
    help="생성할 씬 개수 (각 환경에서 반복)"
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="GUI 없이 실행"
)
parser.add_argument(
    "--list_objects",
    action="store_true",
    help="사용 가능한 오브젝트 목록 출력 후 종료"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="병렬로 실행할 환경 개수 (기본값: 4)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="재현성을 위한 random seed. 지정 안 하면 자동 생성 후 run_metadata.json에 기록됨 "
         "(단, GPU physics 자체는 seed만으로 100%% 재현되지 않을 수 있어 최종 object pose를 "
         "별도로 저장함 -- seed는 참고용)"
)
parser.add_argument(
    "--diagnose_drift",
    action="store_true",
    help="캡처 루프 시작 직전/직후의 object pose를 비교해서 실제로 drift가 0인지 확인하는 "
         "진단을 실행(생성 시간이 약간 늘어남). world.render()로 전환한 게 실제로 물리를 "
         "안 건드리는지 검증용. 평소 생성 시에는 끄는 게 맞음"
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="지정하면 scene00001부터 다시 시작해서 기존 파일을 덮어씀. 지정 안 하면 "
         "output/<target>/scene/depth/에서 기존 scene 번호 중 최댓값 다음부터 자동 시작 "
         "(기존 데이터 보존)"
)

args, unknown = parser.parse_known_args()

if args.seed is None:
    args.seed = random.randint(0, 2**31 - 1)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ==============================================================================
# 1. Launch Simulation App
# ==============================================================================
simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import isaacsim.core.utils.torch.rotations as rot_utils

from object_spawner import ObjectSpawner

# ==============================================================================
# 2. Configuration
# ==============================================================================
NUM_ENVS = args.num_envs
GRID_SPACING = 3

# --- 카메라 파라미터 ---
CAMERA_HEIGHT_OFFSET = 3    # 모든 카메라의 z 높이
CAMERA_XY_OFFSET = 1       # left/right/top/bottom 카메라의 x 또는 y 오프셋

# --- 스폰 안정화 스텝 ---
STABILIZATION_STEPS       = 60   # 각 오브젝트 투하 후 안정화 스텝 수
FINAL_STABILIZATION_STEPS = 120  # 모든 오브젝트 투하 후 최종 안정화 스텝 수

# --- 타겟 레이어 확률 ---
TARGET_NOT_BOTTOM_PROB = 0.15  # target이 맨 아래가 아닐 확률 (0.0 = 항상 아래)
TARGET_TOP_PROB        = 0.6  # 맨 아래가 아닌 경우 중 맨 위(마지막)일 확률

# 카메라 설정: offset은 (x, y)만, z는 CAMERA_HEIGHT_OFFSET 사용
CAMERA_CONFIGS = {
    "center": {"offset": (0.0, 0.0)},
    "left":   {"offset": (-CAMERA_XY_OFFSET, 0.0)},
    "right":  {"offset": (CAMERA_XY_OFFSET, 0.0)},
    "top":    {"offset": (0.0, CAMERA_XY_OFFSET)},
    "bottom": {"offset": (0.0, -CAMERA_XY_OFFSET)},
}

SRC_DIR = os.path.dirname(__file__)
# src/scene_generator/asset가 아니라 한 단계 위 src/asset/가 실제 asset 위치 (원본의 버그 수정)
ASSET_DIR = os.path.join(os.path.dirname(SRC_DIR), "asset")
# GitHub 저장소 안의 scene_generator/에서 실행하면서 asset이 저장소의 형제 폴더에 있는 경우도 지원.
if not os.path.isdir(ASSET_DIR):
    ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(SRC_DIR)), "asset")
USD_FILE_DIR = ASSET_DIR  # asset/<Category>/ 가 바로 있음 (예전 asset/260303/<Category> 구조 아님)

WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "drawer.usd")  # asset/USD/drawer.usd 아니라 asset/drawer.usd

# Workspace surface bounds for random object placement
WORKSPACE_BOUNDS = {
    "x": (-0.15, 0.15),
    "y": (-0.15, 0.15),
    "z_surface": 0.01,
    "z_drop": 0.2,  # 오브젝트 투하 높이
}

# ==============================================================================
# 3. Create World & Ground Plane
# ==============================================================================
world = World(physics_dt=1 / 120.0, backend="torch", device="cuda")  # 더 작은 timestep

# 물리 안정성 설정 (충돌 투과 방지)
physics_context = world.get_physics_context()
physics_context.set_solver_type("TGS")  # Temporal Gauss-Seidel (더 안정적)
physics_context.enable_ccd(True)  # Continuous Collision Detection 활성화

GroundPlane(
    prim_path="/World/GroundPlane",
    z_position=0,
    color=torch.tensor([1.0, 1.0, 1.0]),
)

# 3-1. Add Lighting (Stage Light에서 보이도록)
from pxr import UsdLux, UsdPhysics, Usd
import omni.usd
stage = omni.usd.get_context().get_stage()

# Dome Light - 환경 전체를 비추는 조명 (HDRI 대신 사용)
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000)

# Distant Light - 태양광처럼 평행한 방향성 조명
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateAngleAttr(0.53)

# ==============================================================================
# 4. Build Base Environment (env_0)
# ==============================================================================
# 4-1. Workspace (Static Collider로 설정 - 고정된 물체)
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace_0")

# drawer의 모든 하위 prim에서 Rigid Body 제거 (Static Collider만 유지)
def make_static_collider(prim_path: str):
    """prim과 모든 하위 prim에서 Rigid Body를 제거하고 Static Collider로 만듦"""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    for prim in Usd.PrimRange(root_prim):
        # Rigid Body API 제거 (있으면)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

make_static_collider("/World/workspace_0")

# 4-2. Create objects at default position (outside workspace)
object_spawner = ObjectSpawner(
    world=world,
    categories=["Book", "Toy", "Fruit", "Packaged_food"],
    usd_folder_dir=USD_FILE_DIR,
    container_prim_path="/World/Objects_0",
    workspace_bounds=WORKSPACE_BOUNDS,
    default_position=torch.tensor([0.0, 0.7, 0.05]),
    num_to_spawn=None,          # load all available assets (asset 폴더에 물체가 늘어나면 자동으로 같이 늘어남)
    extensions=(".usd",),       # only .usd files
)

# 각 오브젝트 prim에 semantic 레이블 추가 (클론 전에 해야 클론에도 상속됨)
# instance_segmentation이 오브젝트 단위로 그룹핑하기 위해 필요
from semantics.schema.editor import PrimSemanticData
for prim_path, obj_name in zip(object_spawner._spawned_paths, object_spawner._spawned_names):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        sem_data = PrimSemanticData(prim)
        sem_data.add_entry("class", obj_name)

# 4-3. Cameras (5개: center, left, right, top, bottom)
# 모든 카메라가 같은 높이(z=3.5)에서 서랍 중앙(0,0)을 바라봄
import numpy as np
from scipy.spatial.transform import Rotation as R

def look_at_rotation(cam_pos, target_pos=[0, 0, 0]):
    """카메라가 target을 바라보는 quaternion (w,x,y,z) 반환"""
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    # 기본 카메라 방향: -Z
    cam_forward = np.array([0, 0, -1])
    rotation, _ = R.align_vectors([direction], [cam_forward])
    q = rotation.as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

cameras = {}

for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])

    # 카메라 생성 (orientation은 클론 후 설정)
    cam = Camera(
        prim_path=f"/World/camera_{cam_name}_0",
        position=np.array(cam_pos),
        resolution=(640, 480),
    )
    cam.initialize()

    cameras[cam_name] = cam

# center 카메라에 depth/segmentation 어노테이터 추가 (캡처에 사용되는 카메라)
cameras["center"].add_distance_to_image_plane_to_frame()

# Camera 래퍼가 colorize 파라미터를 지원하지 않으므로 Replicator API 직접 사용
import omni.replicator.core as rep
_rep_rp = rep.create.render_product(cameras["center"].prim_path, (640, 480))
_seg_annotator = rep.AnnotatorRegistry.get_annotator(
    "instance_segmentation", init_params={"colorize": False}
)
_seg_annotator.attach([_rep_rp])

# ==============================================================================
# 5. Clone Environments
# ==============================================================================
cloner = GridCloner(spacing=GRID_SPACING)

workspace_paths = cloner.generate_paths("/World/workspace", NUM_ENVS)
object_paths = cloner.generate_paths("/World/Objects", NUM_ENVS)

# 각 카메라 타입별로 경로 생성 및 클론
camera_paths_dict = {}
for cam_name in CAMERA_CONFIGS.keys():
    camera_paths_dict[cam_name] = cloner.generate_paths(f"/World/camera_{cam_name}", NUM_ENVS)

cloner.clone(source_prim_path="/World/workspace_0", prim_paths=workspace_paths)
cloner.clone(source_prim_path="/World/Objects_0", prim_paths=object_paths)

for cam_name in CAMERA_CONFIGS.keys():
    cloner.clone(
        source_prim_path=f"/World/camera_{cam_name}_0",
        prim_paths=camera_paths_dict[cam_name]
    )

# ==============================================================================
# 6. Arrange Cloned Poses
# ==============================================================================
workspaces_view = XFormPrimView("/World/workspace_*")
objects_view = XFormPrimView("/World/Objects_*")

# 각 카메라 타입별 View 생성
camera_views_dict = {}
for cam_name in CAMERA_CONFIGS.keys():
    camera_views_dict[cam_name] = XFormPrimView(f"/World/camera_{cam_name}_*")

# Align object containers with workspace positions
positions, orientations = workspaces_view.get_world_poses()
workspaces_view.set_world_poses(positions, orientations)
objects_view.set_world_poses(positions, orientations)

# 각 카메라를 workspace 위치 기준으로 배치
# 모든 카메라: z = CAMERA_HEIGHT_OFFSET, xy는 각 카메라별 오프셋 적용
# offset 값에 따라 자동으로 서랍 중앙(0,0,0)을 바라보는 orientation 계산

# 카메라 orientation 저장 (텔레포트 시 재사용)
camera_orientations = {}

for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_view = camera_views_dict[cam_name]

    # workspace 위치 + xy오프셋 + z높이
    cam_positions = positions.clone()
    cam_positions[:, 0] += offset[0]  # x 오프셋
    cam_positions[:, 1] += offset[1]  # y 오프셋
    cam_positions[:, 2] += CAMERA_HEIGHT_OFFSET  # z 높이 (모든 카메라 동일)

    # offset 기반으로 서랍 중앙을 바라보는 orientation 계산
    cam_pos_local = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    quat_wxyz = look_at_rotation(cam_pos_local, target_pos=[0, 0, 0])

    # orientation 저장
    camera_orientations[cam_name] = quat_wxyz

    # 모든 환경에 동일한 orientation 적용
    cam_orients = torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(NUM_ENVS, 1)
    cam_view.set_world_poses(cam_positions, cam_orients)

# workspace 원점 위치 저장 (텔레포트 시 사용)
workspace_origins = positions.clone()


def _atomic_write_json(path: str, obj: dict):
    """임시 파일에 쓴 뒤 os.replace()로 교체 -- 쓰는 도중 프로세스가 죽어도 기존 파일이나
    완전히 쓰인 새 파일 중 하나만 존재하게 되어(중간에 잘린 JSON이 남지 않음) 안전함."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _update_run_metadata(target_name: str, run_id: str, **updates):
    """run_metadata.json에서 run_id가 일치하는 run entry를 찾아 필드를 갱신(atomic write)."""
    output_base = os.path.join(os.path.dirname(__file__), "output", target_name, "scene")
    run_metadata_path = os.path.join(output_base, "run_metadata.json")
    with open(run_metadata_path) as f:
        run_metadata = json.load(f)
    for run in run_metadata["runs"]:
        if run.get("run_id") == run_id:
            run.update(updates)
            break
    _atomic_write_json(run_metadata_path, run_metadata)


def save_run_metadata(target_name: str, scene_start: int, scene_end: int) -> str:
    """카메라 intrinsics/extrinsics(occlusion_gt_pilot_capture.py와 동일 필드로 저장 --
    depth_rasterizer.load_camera_metadata()로 바로 재사용 가능)와 생성 설정값을 씬 생성
    시작 전 1회 저장. GPU physics는 seed만으로 완전히 재현되지 않을 수 있으므로 seed는
    참고용이고, 실제 재현/추적은 save_scene_images()가 씬별로 저장하는 최종 object pose가
    핵심임. scene_start/scene_end로 이 run이 실제로 어느 scene 번호를 생성했는지 기록해서
    여러 번 나눠 캡처해도 run<->scene 대응이 추적 가능하게 함. run_id를 반환(poses.json에
    같이 기록해서 역방향 조회도 가능하게)."""
    output_base = os.path.join(os.path.dirname(__file__), "output", target_name, "scene")
    os.makedirs(output_base, exist_ok=True)

    camera_metadata = {"resolution": [640, 480], "cameras": {}}
    for cam_name, cam_config in CAMERA_CONFIGS.items():
        offset = cam_config["offset"]
        cam = cameras[cam_name]
        camera_metadata["cameras"][cam_name] = {
            "position": [offset[0], offset[1], CAMERA_HEIGHT_OFFSET],
            "orientation_wxyz": list(camera_orientations[cam_name]),
            "focal_length": float(cam.get_focal_length()),
            "horizontal_aperture": float(cam.get_horizontal_aperture()),
            "vertical_aperture": float(cam.get_vertical_aperture()),
            "clipping_range": list(cam.get_clipping_range()),
        }
    with open(os.path.join(output_base, "camera_metadata.json"), "w") as f:
        json.dump(camera_metadata, f, indent=2)

    run_id = f"{target_name}_{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    this_run = {
        "run_id": run_id,
        "target_name": target_name,
        "status": "started",  # 캡처 도중 실패해도 이 값이 남아 있으면 미완성임을 알 수 있음.
                               # save_scene_images()가 씬을 하나 끝낼 때마다 scene_end_completed를
                               # 갱신하고, 전체 루프가 끝나면 mark_run_completed()가 completed로 바꿈
        "scene_end_completed": scene_start - 1,
        "seed": args.seed,
        "seed_note": "GPU physics는 seed만으로 완전히 재현 보장 안 됨 -- 씬별 poses.json이 authoritative",
        "scene_start": scene_start,
        "scene_end": scene_end,
        "num_envs": NUM_ENVS,
        "num_scenes": args.num_scenes,
        "overwrite": args.overwrite,
        "grid_spacing": GRID_SPACING,
        "camera_height_offset": CAMERA_HEIGHT_OFFSET,
        "camera_xy_offset": CAMERA_XY_OFFSET,
        "stabilization_steps": STABILIZATION_STEPS,
        "final_stabilization_steps": FINAL_STABILIZATION_STEPS,
        "target_not_bottom_prob": TARGET_NOT_BOTTOM_PROB,
        "target_top_prob": TARGET_TOP_PROB,
        "workspace_bounds": WORKSPACE_BOUNDS,
        "resolution": [640, 480],
        "script": os.path.abspath(__file__),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # run_metadata.json은 실행마다 덮어쓰지 않고 "runs" 리스트에 누적 -- 여러 번 나눠 캡처해도
    # 각 실행의 seed/설정이 유실되지 않게 함
    run_metadata_path = os.path.join(output_base, "run_metadata.json")
    if os.path.exists(run_metadata_path):
        with open(run_metadata_path) as f:
            run_metadata = json.load(f)
        if "runs" not in run_metadata:
            run_metadata = {"runs": [run_metadata]}  # 이전(구버전) 단일 run 포맷 마이그레이션
    else:
        run_metadata = {"runs": []}
    run_metadata["runs"].append(this_run)
    _atomic_write_json(run_metadata_path, run_metadata)
    print(f"[run_metadata.json / camera_metadata.json 저장 완료] run_id={run_id}")
    return run_id

# 텔레포트용 카메라 뷰 (center 카메라 하나만 사용)
# XFormPrimView를 통해 텔레포트하면 GUI와 동일하게 작동
teleport_cam = cameras["center"]
teleport_cam_view = XFormPrimView("/World/camera_center_0")

# Create per-item views that span all cloned environments
object_spawner.setup_cloned_views(num_envs=NUM_ENVS)

# ==============================================================================
# 7. Simulation Loop
# ==============================================================================
world.play()

# Start with objects at default (outside workspace)
object_spawner.initialize()
world.step(render=True)

# 사용 가능한 오브젝트 목록 출력 (항상)
available_objects = object_spawner.get_target_candidates()
print("\n=== 사용 가능한 오브젝트 목록 ===")
print("  [입력 가능한 이름] → [실제 USD 파일]")
for folder_name, usd_name in available_objects:
    category = object_spawner.objects_class.get(usd_name, "unknown")
    print(f"  --target {folder_name}  →  {usd_name} ({category})")
print("================================\n")

# --list_objects 옵션: 목록 출력 후 종료
if args.list_objects:
    simulation_app.close()
    exit(0)

# ==============================================================================
# 이미지 캡처 함수 (텔레포트 방식: 카메라 1개를 5개 위치로 이동하며 캡처)
# ==============================================================================
import cv2

def save_scene_images(scene_idx: int, target_name: str, target_asset_name: str = None, run_id: str = None):
    """target_name: output 폴더명(예: fruit_1). target_asset_name: 실제 USD/오브젝트 이름
    (예: Apple) -- resolve_target_name()으로 변환된 값, is_target 판정에 사용. 지정 안 하면
    target_name과 동일하다고 가정(랜덤 스폰 모드 등 폴더명=오브젝트명인 경우 대비)."""
    if target_asset_name is None:
        target_asset_name = target_name
    """
    텔레포트 방식: 카메라 1개를 각 위치로 이동하며 캡처
    260305의 look_at_rotation 사용 (로컬 좌표 기준)

    저장 구조:
    output/{target_name}/scene/
        ├── rgb/scene{:05d}_env{:04d}_{camera}.png
        ├── depth/scene{:05d}_env{:04d}_{camera}.npy
        └── seg/scene{:05d}_env{:04d}_{camera}.png
    """
    output_base = os.path.join(os.path.dirname(__file__), "output", target_name, "scene")
    rgb_dir = os.path.join(output_base, "rgb")
    depth_dir = os.path.join(output_base, "depth")
    seg_dir = os.path.join(output_base, "seg")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    saved_count = 0
    all_scene_classes = {}  # 모든 env/카메라에서 보이는 물체 누적

    # 렌더링 안정화 -- world.render()는 SimulationContext.render()를 그대로 상속(physics
    # step 없이 app.update()만 호출, playSimulations를 잠시 꺼둠)하므로 물체가 계속
    # 안정화 물리를 받는 world.step(render=True)와 달리 pose를 조금도 움직이지 않음.
    for _ in range(20):
        world.render()

    pre_capture_poses = None
    if args.diagnose_drift:
        pre_capture_poses = {}
        for name, view in zip(object_spawner._spawned_names, object_spawner._item_views):
            pos, orient = view.get_world_poses()
            pre_capture_poses[name] = (pos.cpu().numpy(), orient.cpu().numpy())

    # XFormPrimView를 통해 텔레포트 (GUI와 동일하게 작동)
    for env_idx in range(NUM_ENVS):
        env_origin = workspace_origins[env_idx].cpu().numpy()

        for cam_name, cam_config in CAMERA_CONFIGS.items():
            offset = cam_config["offset"]

            # 월드 좌표에서의 카메라 위치
            cam_world_pos = env_origin + np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])

            # 저장된 orientation 사용
            quat_wxyz = camera_orientations[cam_name]

            # XFormPrimView로 텔레포트 (Camera.set_world_pose 대신!)
            pos_tensor = torch.from_numpy(np.array(cam_world_pos, dtype=np.float32)).unsqueeze(0).to("cuda")
            orient_tensor = torch.from_numpy(np.array(quat_wxyz, dtype=np.float32)).unsqueeze(0).to("cuda")
            teleport_cam_view.set_world_poses(pos_tensor, orient_tensor)

            # 렌더링 대기 (render-only, physics 미진행 -- object pose가 캡처 도중 전혀
            # 움직이지 않으므로 캡처 후 읽는 poses.json이 촬영 시점과 구조적으로 정확히 일치)
            for _ in range(5):
                world.render()

            filename_base = f"scene{scene_idx+1:05d}_env{env_idx:04d}_{cam_name}"

            # RGB 캡처
            rgb = teleport_cam.get_rgb()
            if rgb is not None:
                cv2.imwrite(os.path.join(rgb_dir, f"{filename_base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # Depth 캡처
            depth = teleport_cam.get_depth()
            if depth is not None:
                np.save(os.path.join(depth_dir, f"{filename_base}.npy"), depth)

            # Segmentation 캡처 (Replicator annotator에서 직접 취득)
            seg_data = _seg_annotator.get_data()
            if seg_data is not None and isinstance(seg_data, dict) and "data" in seg_data:
                seg_ids = seg_data["data"]  # (H, W) 또는 (H, W, 1) uint32
                if seg_ids.ndim == 3:
                    seg_ids = seg_ids[:, :, 0]

                # id_to_labels: {id_str → prim_path_str}
                # 예: {'6': '/World/Objects_0/object_4', ...}
                info = seg_data.get("info", {})
                id_to_labels = info.get("idToLabels", {})

                # object_N 키 → 오브젝트 이름 매핑 (클론 환경 통합)
                # /World/Objects_2/object_4 → "object_4" → _spawned_names[4]
                obj_key_to_name = {
                    p.split("/")[-1]: n
                    for p, n in zip(object_spawner._spawned_paths, object_spawner._spawned_names)
                }

                # 클래스명 → 결정론적 색상 (클래스명 해시 기반, 전 씬 동일)
                if not hasattr(save_scene_images, "_class_colors"):
                    save_scene_images._class_colors = {}
                class_colors = save_scene_images._class_colors

                seg_color = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)
                scene_classes = {}

                for uid in np.unique(seg_ids):
                    prim_label = id_to_labels.get(str(int(uid)), "")
                    if not prim_label or prim_label in ("BACKGROUND", "UNLABELLED"):
                        continue
                    obj_key = prim_label.split("/")[-1]  # "object_4"
                    class_name = obj_key_to_name.get(obj_key, "")
                    if not class_name:
                        continue

                    if class_name not in class_colors:
                        idx = len(class_colors)
                        hue = (idx * 23) % 180  # prime stride → 충돌 없는 고유 색상
                        color = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
                        class_colors[class_name] = color.tolist()

                    color = class_colors[class_name]  # COLOR_HSV2BGR 결과라 실제로는 BGR 순서
                    seg_color[seg_ids == uid] = color
                    # color_bgr이 정식 필드(cv2.imwrite/imread와 스왑 없이 바로 맞음).
                    # legacy_color는 예전 "color_rgb" 키로 저장되던 것과 정확히 같은 값(실제로는
                    # BGR인데 이름만 RGB였음, QC 스크립트에서 스왑 버그를 유발한 적 있음) --
                    # 이름이 잘못된 옛 필드를 그대로 legacy_color로 이름만 바꿔 호환 유지.
                    # color_rgb는 이제 진짜 RGB 순서(color를 뒤집은 값)로 정확하게 저장.
                    scene_classes[class_name] = {
                        "color_bgr": color,
                        "color_rgb": color[::-1],
                        "legacy_color": color,
                    }

                cv2.imwrite(os.path.join(seg_dir, f"{filename_base}.png"), seg_color)

                # 모든 env/카메라에서 보이는 물체 누적
                all_scene_classes.update(scene_classes)

            saved_count += 1

    # 씬별 매핑 JSON 저장 (모든 env/카메라 누적 후 1회)
    json_path = os.path.join(seg_dir, f"scene{scene_idx+1:05d}_mapping.json")
    with open(json_path, "w") as f:
        json.dump(all_scene_classes, f, indent=2, ensure_ascii=False)

    # env별 최종 object pose 저장 (env_origin 기준 상대 좌표로 변환).
    # 모든 카메라 캡처가 끝난 뒤에 수행 -- 캡처 도중(카메라 teleport + world.step 사이)에
    # GPU pose 조회(.get_world_poses().cpu().numpy())를 끼워넣으면 렌더링 파이프라인이
    # traceback 없이 응답 없는 상태로 종료되는 현상을 재현했음. 원인은 확인되지 않았고
    # (CUDA 동기화 충돌은 추정일 뿐 증명되지 않음) 이 위치가 재현 가능하게 안전했다는
    # 것만 확인됨. 주의: 이 시점에 읽는 pose는 "capture 시점"이 아니라 "capture 종료 후"
    # 값이며, physics가 활성 상태로 capture 전체(첫 env부터 마지막 env까지)를 거치는 동안
    # 물체가 계속 미세하게 움직였을 수 있음 -- 아래에서 실제 drift 크기를 측정해 기록한다.
    env_objects_meta = [[] for _ in range(NUM_ENVS)]
    for name, view in zip(object_spawner._spawned_names, object_spawner._item_views):
        pos, orient = view.get_world_poses()
        pos_np = pos.cpu().numpy()
        orient_np = orient.cpu().numpy()
        category = object_spawner._objects_class.get(name, "")
        is_target = name == target_asset_name
        for env_idx in range(NUM_ENVS):
            env_origin = workspace_origins[env_idx].cpu().numpy()
            env_objects_meta[env_idx].append({
                "name": name,
                "category": category,
                "is_target": is_target,
                "position_local": (pos_np[env_idx] - env_origin).tolist(),
                "position_world": pos_np[env_idx].tolist(),
                "orientation_wxyz": orient_np[env_idx].tolist(),
            })
    for env_idx in range(NUM_ENVS):
        poses_path = os.path.join(output_base, f"scene{scene_idx+1:05d}_env{env_idx:04d}_poses.json")
        _atomic_write_json(poses_path, {"scene_idx": scene_idx + 1, "env_idx": env_idx,
                                         "target_name": target_name, "run_id": run_id,
                                         "objects": env_objects_meta[env_idx]})

    if run_id is not None:
        _update_run_metadata(target_name, run_id, scene_end_completed=scene_idx + 1)

    print(f"  → {saved_count}개 이미지 세트 저장 (rgb/depth/seg), 매핑 {len(all_scene_classes)}개 오브젝트, "
          f"pose {len(object_spawner._spawned_names)}개 오브젝트 x {NUM_ENVS} env 저장")

    # === (--diagnose_drift 전용) 캡처 루프 시작 직전 vs 종료 직후 pose 비교 ===
    # world.render()가 정말로 physics를 안 건드리는지 실측으로 확인. 0에 가까워야
    # world.step(render=True)->world.render() 전환이 drift를 구조적으로 없앴다는 증거가 됨.
    if args.diagnose_drift and pre_capture_poses is not None:
        max_pos_drift, sum_pos_drift = 0.0, 0.0
        max_rot_drift_deg, sum_rot_drift_deg = 0.0, 0.0
        max_component_err = 0.0
        n_drift = 0
        for name in object_spawner._spawned_names:
            pre_pos, pre_orient = pre_capture_poses[name]
            for env_idx in range(NUM_ENVS):
                obj = env_objects_meta[env_idx][
                    [o["name"] for o in env_objects_meta[env_idx]].index(name)]
                after_pos = np.array(obj["position_world"])
                after_orient = np.array(obj["orientation_wxyz"])
                d_pos = float(np.linalg.norm(after_pos - pre_pos[env_idx]))

                # float64로 승격 + 재정규화 후 atan2 기반 각도(arccos는 dot~1 근처에서
                # 수치적으로 불안정 -- 이전에 관측된 0.02~0.08deg가 실제 회전인지 float32
                # 노이즈의 arccos 증폭인지 구분하기 위해 안정적인 방식으로 다시 계산)
                q1 = pre_orient[env_idx].astype(np.float64)
                q2 = after_orient.astype(np.float64)
                q1 /= np.linalg.norm(q1)
                q2 /= np.linalg.norm(q2)
                dot = float(np.dot(q1, q2))
                d_rot_deg = float(np.degrees(2.0 * np.arctan2(
                    np.sqrt(max(0.0, 1.0 - dot * dot)), abs(dot))))
                # quaternion 성분 자체의 최대 절대 차이(부호 모호성 보정: q와 -q는 같은 회전)
                q2_aligned = q2 if dot >= 0 else -q2
                component_err = float(np.max(np.abs(q1 - q2_aligned)))

                max_pos_drift = max(max_pos_drift, d_pos)
                sum_pos_drift += d_pos
                max_rot_drift_deg = max(max_rot_drift_deg, d_rot_deg)
                sum_rot_drift_deg += d_rot_deg
                max_component_err = max(max_component_err, component_err)
                n_drift += 1
        pos_ok = max_pos_drift < 1e-5           # <= 0.01mm
        component_ok = max_component_err < 1e-6  # float32 eps(~1.19e-7)의 몇 배 수준
        verdict = "PASS (render-only 확인됨)" if (pos_ok and component_ok) else "FAIL (여전히 물리가 진행됨)"
        print(f"  [drift 진단] 캡처 루프 시작 직전 vs 종료 직후 실제 변위: "
              f"위치 mean={sum_pos_drift/n_drift*1000:.4f}mm max={max_pos_drift*1000:.4f}mm | "
              f"회전(atan2) mean={sum_rot_drift_deg/n_drift:.6f}deg max={max_rot_drift_deg:.6f}deg | "
              f"quaternion 성분 최대차={max_component_err:.2e} (n={n_drift}) -> {verdict}")

    return output_base


try:
    # 타겟이 지정된 경우: 유사도 기반 순차 스폰
    if args.target:
        # 타겟 이름 검증 (폴더 이름 또는 USD 이름)
        resolved_target = object_spawner.resolve_target_name(args.target)
        target_found = resolved_target in object_spawner._spawned_names

        if not target_found:
            print(f"\n⚠️  경고: '{args.target}'을(를) 찾을 수 없습니다!")
            print(f"   위 목록에서 폴더 이름(예: book_1)을 사용하세요.")
            exit(1)

        print(f"\n[유사도 기반 스폰] 타겟: {args.target} → {resolved_target}")
        print(f"생성할 씬 개수: {args.num_scenes}\n")

        # 덮어쓰기 방지: --overwrite 없으면 기존 scene 번호 최댓값 다음부터 자동 시작
        start_scene_idx = 0
        target_depth_dir = os.path.join(os.path.dirname(__file__), "output", args.target, "scene", "depth")
        if not args.overwrite and os.path.isdir(target_depth_dir):
            existing = [f for f in os.listdir(target_depth_dir) if f.startswith("scene")]
            if existing:
                import re as _re
                nums = [int(m.group(1)) for f in existing if (m := _re.match(r"scene(\d+)_", f))]
                if nums:
                    start_scene_idx = max(nums)
                    print(f"[기존 데이터 감지] scene{max(nums):05d}까지 존재 -- "
                          f"scene{start_scene_idx+1:05d}부터 이어서 생성 (--overwrite로 덮어쓰기 가능)\n")
        elif args.overwrite:
            print("[--overwrite 지정됨] scene00001부터 덮어씀\n")

        run_id = save_run_metadata(args.target, scene_start=start_scene_idx + 1,
                                    scene_end=start_scene_idx + args.num_scenes)

        for scene_idx_rel in range(args.num_scenes):
            scene_idx = start_scene_idx + scene_idx_rel
            print(f"--- Scene {scene_idx_rel + 1}/{args.num_scenes} (scene{scene_idx+1:05d}) ---")

            # 오브젝트 초기화 (대기 위치로)
            object_spawner.initialize()
            for _ in range(30):
                world.step(render=True)

            # 유사도 기반 순차 스폰
            object_spawner.spawn_with_similarity(
                target_name=args.target,
                world=world,
                stabilization_steps=STABILIZATION_STEPS,
                final_stabilization_steps=FINAL_STABILIZATION_STEPS,
                target_not_bottom_prob=TARGET_NOT_BOTTOM_PROB,
                target_top_prob=TARGET_TOP_PROB,
            )

            # 이미지 캡처 및 저장
            save_scene_images(scene_idx, args.target, target_asset_name=resolved_target, run_id=run_id)

            print(f"Scene {scene_idx_rel + 1}/{args.num_scenes} (scene{scene_idx+1:05d}) 완료!")

            # 잠시 대기 (결과 확인용)
            for _ in range(30):
                world.step(render=True)

        _update_run_metadata(args.target, run_id, status="completed")
        print("\n모든 씬 생성 완료!")

    # 타겟 미지정: 기존 랜덤 스폰 방식
    else:
        print("⚠️  --target 옵션이 지정되지 않았습니다. 랜덤 스폰 모드로 실행합니다.")
        print("   유사도 기반 스폰을 원하면: --target <오브젝트이름>\n")

        object_spawner.spawn(randomize=True)

        count = 0
        while simulation_app.is_running():
            if count % 200 == 0 and count > 0:
                object_spawner.spawn()
            world.step(render=True)
            count += 1

finally:
    world.stop()
    simulation_app.close()
