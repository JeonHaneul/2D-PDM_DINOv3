"""PyTorch 기반 z-buffer 삼각형 rasterizer. open3d/pyrender/pytorch3d가 이 환경(Python 3.14)에서
설치가 안 돼서 직접 작성함 -- mesh를 카메라 pose에서 depth 이미지로 렌더링하는 최소 기능만 구현.

좌표계/카메라 관례 (scene_generator/vectorized_object_occlusion.py, target_capture.py와 반드시
일치해야 함 -- 여기서 하나라도 틀리면 이후 전체 GT가 어긋남):
  - world: Isaac Sim 기본 (x,y,z), z가 위.
  - 카메라 local: USD 관례대로 카메라는 자신의 -Z축을 바라봄, +Y가 위, +X가 오른쪽.
  - 카메라 orientation: look_at_rotation()으로 특정 target_pos를 바라보게 함 (기존 코드와 동일 함수).
  - depth 값: cam.add_distance_to_image_plane_to_frame()과 동일한 의미 -- 카메라의 이미지
    평면에 수직인 거리(= 카메라 공간 -Z 값의 절댓값), 점까지의 유클리드 거리가 아님.
"""
import numpy as np
import torch


def look_at_rotation_matrix(cam_pos, target_pos=(0.0, 0.0, 0.0)) -> np.ndarray:
    """scene_generator의 look_at_rotation()과 동일한 결과를 내는 3x3 world<-camera 회전행렬
    (카메라 local 축들을 world 좌표로 표현한 것, 즉 world = R @ local + t 에서의 R)."""
    cam_pos = np.array(cam_pos, dtype=np.float64)
    target_pos = np.array(target_pos, dtype=np.float64)
    forward = target_pos - cam_pos
    forward = forward / np.linalg.norm(forward)  # 카메라가 바라보는 방향 (world 기준)
    # USD 카메라는 local -Z를 바라보므로, local -Z 축이 world forward와 같아야 함 -> local +Z = -forward
    cam_z = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(cam_z, world_up)) > 0.999:  # 카메라가 거의 수직으로 아래/위를 보는 경우 gimbal 회피
        world_up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(world_up, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    # R의 각 열이 world 기준으로 표현된 카메라의 local x,y,z 축
    R = np.stack([cam_x, cam_y, cam_z], axis=1)  # (3,3)
    return R


def camera_intrinsics(resolution=(640, 480), focal_length_mm=50.0, horizontal_aperture_mm=20.955,
                       vertical_aperture_mm=None):
    """pinhole intrinsic 계산. focal_length/aperture는 카메라별로 다를 수 있으니 하드코딩 기본값
    (USD 스키마 기본값, 실측으로 검증된 50.0mm/20.955mm) 대신 camera_metadata.json에서 실제
    값을 읽어 넘기는 걸 권장(load_camera_metadata()가 이걸 해줌). focal_length와 aperture가
    같은 단위(둘 다 USD raw 값이든 둘 다 /10 된 값이든)이기만 하면 fx,fy는 그 비율만으로 정확히
    계산되므로 어느 쪽 단위를 쓰든 상관없음(실측으로 확인됨)."""
    W, H = resolution
    if vertical_aperture_mm is None:
        vertical_aperture_mm = horizontal_aperture_mm * (H / W)
    fx = focal_length_mm * (W / horizontal_aperture_mm)
    fy = focal_length_mm * (H / vertical_aperture_mm)
    cx, cy = W / 2.0, H / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H}


def quat_wxyz_to_rotmat(q_wxyz) -> np.ndarray:
    """world<-camera 회전행렬 R (world = R @ local + t 에서의 R). USD/Isaac Sim의 (w,x,y,z)
    쿼터니언 관례."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_camera_metadata(metadata_path: str) -> dict:
    """occlusion_gt_pilot_capture.py가 저장한 camera_metadata.json을 읽어서 카메라별로
    render_depth()에 바로 넘길 수 있는 형태(position, R, intrinsics)로 반환."""
    import json
    with open(metadata_path) as f:
        meta = json.load(f)
    resolution = tuple(meta["resolution"])
    cams = {}
    for cam_name, c in meta["cameras"].items():
        cams[cam_name] = {
            "position": c["position"],
            "R": quat_wxyz_to_rotmat(c["orientation_wxyz"]),
            "intrinsics": camera_intrinsics(resolution, c["focal_length"], c["horizontal_aperture"],
                                             c["vertical_aperture"]),
        }
    return {"resolution": resolution, "cameras": cams}


def render_depth(vertices_world: np.ndarray, faces: np.ndarray, cam_pos, cam_target=None,
                  resolution=(640, 480), device: str = "cuda", R: np.ndarray = None,
                  intr: dict = None) -> np.ndarray:
    """mesh(world 좌표)를 카메라 pose에서 depth 이미지로 렌더링.
    R, intr을 직접 주면(권장: load_camera_metadata()로 실측값을 읽어서) 그걸 그대로 쓰고,
    안 주면 cam_target을 바라보는 것으로 가정해 look_at으로 계산 + 기본 intrinsic 사용(스모크
    테스트/약식용, off-axis 카메라 검증에는 실측 R을 쓰는 쪽이 더 정확함).
    반환: (H,W) float32 depth (안 덮인 픽셀=0, add_distance_to_image_plane 관례와 동일하게
    카메라 이미지평면에 수직인 거리)."""
    if intr is None:
        intr = camera_intrinsics(resolution)
    W, H = intr["W"], intr["H"]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if R is None:
        if cam_target is None:
            raise ValueError("R 또는 cam_target 둘 중 하나는 있어야 함")
        R = look_at_rotation_matrix(cam_pos, cam_target)  # world <- camera local, (3,3)
    R_inv = R.T  # camera <- world (회전행렬은 orthonormal이라 전치 = 역행렬)
    t = np.array(cam_pos, dtype=np.float64)

    verts = torch.tensor(vertices_world, dtype=torch.float64, device=dev)  # (N,3)
    faces_t = torch.tensor(faces, dtype=torch.int64, device=dev)  # (M,3)

    # world -> camera space
    verts_cam = (verts - torch.tensor(t, device=dev)) @ torch.tensor(R_inv.T, dtype=torch.float64, device=dev)
    # (주의) R_inv @ v == v @ R_inv.T (row-vector 관례로 맞추기 위해 전치해서 곱함)

    z = -verts_cam[:, 2]  # 카메라는 -Z를 바라보므로, 앞쪽(화면에 찍히는 쪽)은 z>0이 되도록 부호 반전
    x_ndc = verts_cam[:, 0] / z.clamp_min(1e-6)
    y_ndc = verts_cam[:, 1] / z.clamp_min(1e-6)
    px = intr["cx"] + intr["fx"] * x_ndc
    py = intr["cy"] - intr["fy"] * y_ndc  # 이미지 y축은 아래로 증가하므로 부호 반전

    depth_buf = torch.zeros((H, W), dtype=torch.float64, device=dev)
    depth_buf.fill_(float("inf"))

    tri_px = px[faces_t]   # (M,3)
    tri_py = py[faces_t]   # (M,3)
    tri_z  = z[faces_t]    # (M,3)

    valid_tri = (tri_z > 1e-6).all(dim=1)  # 카메라 뒤쪽 삼각형 제외
    tri_px, tri_py, tri_z = tri_px[valid_tri], tri_py[valid_tri], tri_z[valid_tri]

    x_min = tri_px.min(dim=1).values.floor().clamp(0, W - 1).long()
    x_max = tri_px.max(dim=1).values.ceil().clamp(0, W - 1).long()
    y_min = tri_py.min(dim=1).values.floor().clamp(0, H - 1).long()
    y_max = tri_py.max(dim=1).values.ceil().clamp(0, H - 1).long()

    # 삼각형 개수가 많아 pixel-loop을 다 벡터화하면 메모리가 크므로, 삼각형 단위로만 반복
    # (pilot 스케일: 물체당 1~2만 삼각형, pose 수십~수백 개 정도라 이 정도로도 충분히 빠름)
    for i in range(tri_px.shape[0]):
        xs0, xs1 = int(x_min[i]), int(x_max[i])
        ys0, ys1 = int(y_min[i]), int(y_max[i])
        if xs1 < xs0 or ys1 < ys0:
            continue
        gy, gx = torch.meshgrid(
            torch.arange(ys0, ys1 + 1, device=dev), torch.arange(xs0, xs1 + 1, device=dev), indexing="ij"
        )
        px0, py0 = tri_px[i, 0], tri_py[i, 0]
        px1, py1 = tri_px[i, 1], tri_py[i, 1]
        px2, py2 = tri_px[i, 2], tri_py[i, 2]

        denom = (py1 - py2) * (px0 - px2) + (px2 - px1) * (py0 - py2)
        if abs(float(denom)) < 1e-9:
            continue
        gxf, gyf = gx.double() + 0.5, gy.double() + 0.5
        w0 = ((py1 - py2) * (gxf - px2) + (px2 - px1) * (gyf - py2)) / denom
        w1 = ((py2 - py0) * (gxf - px2) + (px0 - px2) * (gyf - py2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z_interp = w0 * tri_z[i, 0] + w1 * tri_z[i, 1] + w2 * tri_z[i, 2]

        region = depth_buf[ys0:ys1 + 1, xs0:xs1 + 1]
        update = inside & (z_interp < region)
        region[update] = z_interp[update]

    depth_buf[torch.isinf(depth_buf)] = 0.0
    return depth_buf.cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # 스모크 테스트: packaged_food_2 target을 실제 촬영 pose(world (0,0,0.05), 회전 없음)에 놓고
    # center 카메라(world (0,0,3.0), 원점을 바라봄)로 렌더링 -> 실제 촬영된 seg 실루엣과 비교
    import cv2
    from mesh_utils import extract_world_mesh

    usd_path = "/home/haneul/isaacsim/src/asset/Packaged_food/packaged_food_2/010_potted_meat_can.usd"
    mesh = extract_world_mesh(usd_path)
    verts = mesh["vertices"] + np.array([0.0, 0.0, 0.05])  # target_capture.py와 동일한 배치: (0,0,0.05)

    depth = render_depth(verts, mesh["faces"], cam_pos=(0.0, 0.0, 3.0), cam_target=(0.0, 0.0, 0.0),
                          resolution=(640, 480))
    my_mask = depth > 0
    print(f"렌더링된 실루엣 픽셀 수: {my_mask.sum()}")
    ys, xs = np.where(my_mask)
    if my_mask.any():
        print(f"  bbox: x=[{xs.min()},{xs.max()}]  y=[{ys.min()},{ys.max()}]")
        print(f"  depth 범위(실루엣 내부): {depth[my_mask].min():.4f} ~ {depth[my_mask].max():.4f} m "
              f"(카메라 높이 3.0m에서 target까지이므로 3.0m 근처가 나와야 함)")

    # 실제 촬영된 seg와 비교 (방금 target_capture.py로 새로 찍은, 확실한 pose의 참조사진)
    real_seg = cv2.imread("/home/haneul/isaacsim/src/scene_generator/output/packaged_food_2/target/seg/000000_center.png")
    real_mask = ~((real_seg[:, :, 0] == 0) & (real_seg[:, :, 1] == 0) & (real_seg[:, :, 2] == 0))
    print(f"\n실제 촬영된 seg 실루엣 픽셀 수: {real_mask.sum()}")
    ys2, xs2 = np.where(real_mask)
    if real_mask.any():
        print(f"  bbox: x=[{xs2.min()},{xs2.max()}]  y=[{ys2.min()},{ys2.max()}]")

    intersection = (my_mask & real_mask).sum()
    union = (my_mask | real_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    print(f"\n실루엣 IoU (계산 vs 실제 촬영): {iou:.4f}")
