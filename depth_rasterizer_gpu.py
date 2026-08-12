"""nvdiffrast 기반 진짜 GPU rasterizer. depth_rasterizer.py(삼각형마다 Python for문 -- 정확하지만
toy_3 같은 고polygon mesh에서 극도로 느림, 577초/장)를 대체하기 위한 hardware-accelerated 버전.
같은 카메라 관례(world<-camera 회전행렬 R, pinhole intrinsics, depth=이미지평면에 수직인 거리)를
그대로 따르므로 depth_rasterizer.render_depth()와 결과가 일치해야 함(검증 대상)."""
import numpy as np
import torch
import nvdiffrast.torch as dr

_ctx = None


def get_context():
    global _ctx
    if _ctx is None:
        _ctx = dr.RasterizeCudaContext()
    return _ctx


def _opengl_projection_matrix(intr: dict, near: float, far: float) -> np.ndarray:
    """pinhole intrinsic(fx,fy,cx,cy,W,H) -> OpenGL 스타일 4x4 projection matrix.
    카메라가 로컬 -Z를 바라보는 convention(우리 world<-camera R과 이미 일치)에서 그대로 사용 가능.
    Y행에 -1을 곱함: nvdiffrast의 픽셀 row 0=상단 관례가 우리 pinhole intr(cy가 아래로 증가하는
    이미지 좌표계) 그대로 넣으면 상하가 뒤집힘 -- top/bottom camera(Y축으로 오프셋된 카메라)에서
    IoU가 0.33~0.38까지 떨어지는 것으로 실측 확인, 이 부호 반전으로 5개 카메라 전부 IoU 0.997+."""
    fx, fy, cx, cy, W, H = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["W"], intr["H"]
    return np.array([
        [2 * fx / W, 0, 1 - 2 * cx / W, 0],
        [0, -(2 * fy / H), -(2 * cy / H - 1), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


def render_depth_gpu(vertices_world: np.ndarray, faces: np.ndarray, cam_pos, R: np.ndarray, intr: dict,
                      near: float = 0.1, far: float = 20.0, device: str = "cuda") -> np.ndarray:
    """depth_rasterizer.render_depth()와 동일한 입출력 관례(world 좌표 mesh, world<-camera R,
    pinhole intr) -> (H,W) float32 depth(안 덮인 픽셀=0, 카메라 이미지평면에 수직인 거리)."""
    ctx = get_context()
    W, H = intr["W"], intr["H"]
    dev = torch.device(device)

    R_inv = R.T
    t = np.array(cam_pos, dtype=np.float64)
    verts_cam_np = (vertices_world - t) @ R_inv.T  # depth_rasterizer.py와 동일 world->camera 변환

    # pymeshlab 등 외부 라이브러리가 non-contiguous 배열(Fortran order 등)을 돌려주는 경우가 있어서
    # nvdiffrast의 "must be contiguous tensors" 요구를 만족시키려면 명시적으로 보장해야 함
    verts_cam_np = np.ascontiguousarray(verts_cam_np)
    faces = np.ascontiguousarray(faces)

    verts_cam = torch.tensor(verts_cam_np, dtype=torch.float32, device=dev)  # (N,3), OpenGL camera space
    faces_t = torch.tensor(faces, dtype=torch.int32, device=dev)  # (M,3)

    proj = torch.tensor(_opengl_projection_matrix(intr, near, far), dtype=torch.float32, device=dev)  # (4,4)
    ones = torch.ones((verts_cam.shape[0], 1), dtype=torch.float32, device=dev)
    verts_cam_h = torch.cat([verts_cam, ones], dim=1)  # (N,4)
    clip = verts_cam_h @ proj.T  # (N,4) clip space

    pos_batch = clip.unsqueeze(0)  # (1,N,4)
    rast, _ = dr.rasterize(ctx, pos_batch, faces_t, resolution=[H, W])

    # 실제 metric depth("이미지평면에 수직인 거리" = -zc)를 별도 attribute로 보간해서 정확히 얻음
    # (NDC z를 역변환하는 것보다 이 방식이 부동소수점 오차가 적고 depth_rasterizer.py 관례와
    # 정확히 같은 값을 준다)
    metric_depth_attr = (-verts_cam[:, 2:3]).unsqueeze(0)  # (1,N,1)
    depth_interp, _ = dr.interpolate(metric_depth_attr.contiguous(), rast, faces_t)  # (1,H,W,1)

    coverage = rast[0, :, :, 3] > 0  # triangle_id>0인 픽셀만 유효
    depth_out = depth_interp[0, :, :, 0] * coverage.float()
    return depth_out.detach().cpu().numpy().astype(np.float32)


def upload_mesh(vertices: np.ndarray, faces: np.ndarray, device: str = "cuda"):
    """mesh를 GPU에 한 번만 올려서 재사용(배치 렌더링에서 매 pose마다 다시 업로드하지
    않도록). 반환된 (verts_t, faces_t)를 render_depth_gpu_batch()에 그대로 넘긴다."""
    dev = torch.device(device)
    verts_t = torch.tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device=dev)
    faces_t = torch.tensor(np.ascontiguousarray(faces), dtype=torch.int32, device=dev)
    return verts_t, faces_t


def yaw_translate_matrices(x, y, z, yaw_deg, device="cuda"):
    """place_mesh()(z축 회전 후 평행이동)와 정확히 같은 변환을, pose batch 전체에 대해
    한 번에 GPU 위에서 만든다. x,y,z,yaw_deg: 같은 길이 B의 1D array/tensor.
    반환: (B,4,4) homogeneous world transform (mesh-local 좌표를 world로 보냄)."""
    dev = torch.device(device)
    x = torch.as_tensor(x, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    z = torch.as_tensor(z, dtype=torch.float32, device=dev)
    yaw = torch.as_tensor(yaw_deg, dtype=torch.float32, device=dev)
    B = x.shape[0]
    yaw_rad = torch.deg2rad(yaw)
    cos_a, sin_a = torch.cos(yaw_rad), torch.sin(yaw_rad)
    T = torch.zeros((B, 4, 4), dtype=torch.float32, device=dev)
    T[:, 0, 0] = cos_a
    T[:, 0, 1] = -sin_a
    T[:, 1, 0] = sin_a
    T[:, 1, 1] = cos_a
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0
    T[:, 0, 3] = x
    T[:, 1, 3] = y
    T[:, 2, 3] = z
    return T


def render_depth_gpu_batch(verts_local_t: torch.Tensor, faces_t: torch.Tensor,
                             world_transforms: torch.Tensor, cam_pos, R: np.ndarray, intr: dict,
                             near: float = 0.1, far: float = 20.0, device: str = "cuda") -> torch.Tensor:
    """render_depth_gpu()의 batch 버전. pose B개를 nvdiffrast의 instanced-batch rasterize
    한 번으로 처리(개별 GPU 호출 B번 대신 1번). mesh는 GPU에 미리 올려둔 verts_local_t/faces_t
    를 재사용(upload_mesh()로 준비), world_transforms는 yaw_translate_matrices()로 생성.

    반환: depth (B,H,W) float32 torch tensor, GPU 위에 그대로 유지(호출자가 필요할 때만
    .cpu()로 내림 -- 매 pose마다 CPU 왕복하던 이전 구조의 병목을 없앰)."""
    ctx = get_context()
    W_res, H_res = intr["W"], intr["H"]
    dev = torch.device(device)
    B, N = world_transforms.shape[0], verts_local_t.shape[0]

    ones = torch.ones((N, 1), dtype=torch.float32, device=dev)
    verts_h = torch.cat([verts_local_t, ones], dim=1)  # (N,4)
    verts_world = torch.einsum("bij,nj->bni", world_transforms, verts_h)[..., :3]  # (B,N,3)

    R_t = torch.tensor(R, dtype=torch.float32, device=dev)
    cam_pos_t = torch.tensor(np.array(cam_pos, dtype=np.float32), dtype=torch.float32, device=dev)
    # render_depth_gpu()의 (vertices_world - t) @ R_inv.T = (vertices_world - t) @ (R.T).T = (vertices_world - t) @ R
    # 와 정확히 같아야 함 -- 여기서 @ R_t.T를 썼던 게 버그(전치를 잘못 취함, off-axis 카메라에서
    # 크게 틀어짐을 validate_batch_rasterizer.py로 실측 발견).
    verts_cam = (verts_world - cam_pos_t) @ R_t  # (B,N,3), world<-camera 변환(단일 카메라, 전체 batch 공통)

    proj = torch.tensor(_opengl_projection_matrix(intr, near, far), dtype=torch.float32, device=dev)  # (4,4)
    ones_bn = torch.ones((B, N, 1), dtype=torch.float32, device=dev)
    verts_cam_h = torch.cat([verts_cam, ones_bn], dim=-1)  # (B,N,4)
    clip = verts_cam_h @ proj.T  # (B,N,4) -- nvdiffrast instanced mode가 요구하는 정확한 형태

    faces_t = faces_t.contiguous()
    rast, _ = dr.rasterize(ctx, clip.contiguous(), faces_t, resolution=[H_res, W_res])  # (B,H,W,4)

    metric_depth_attr = (-verts_cam[..., 2:3]).contiguous()  # (B,N,1)
    depth_interp, _ = dr.interpolate(metric_depth_attr, rast, faces_t)  # (B,H,W,1)

    coverage = rast[..., 3] > 0
    depth_out = depth_interp[..., 0] * coverage.float()  # (B,H,W)
    return depth_out


if __name__ == "__main__":
    # 스모크 테스트: packaged_food_2 center pose/camera에서 depth_rasterizer.py(검증된 CPU-loop
    # 버전)와 결과가 일치하는지 확인
    import time
    from mesh_utils import extract_world_mesh
    from depth_rasterizer import load_camera_metadata, render_depth

    mesh = extract_world_mesh(
        "/home/haneul/isaacsim/src/asset/Packaged_food/packaged_food_2/010_potted_meat_can.usd")
    verts = mesh["vertices"] + np.array([0.0, 0.0, 0.07])  # pilot capture의 실제 center pose z값
    cam_meta = load_camera_metadata(
        "/home/haneul/isaacsim/src/scene_generator/output_occlusion_pilot/packaged_food_2/camera_metadata.json")
    cam = cam_meta["cameras"]["center"]

    t0 = time.time()
    d_cpu = render_depth(verts, mesh["faces"], cam_pos=cam["position"], resolution=cam_meta["resolution"],
                          R=cam["R"], intr=cam["intrinsics"])
    t_cpu = time.time() - t0

    t0 = time.time()
    d_gpu = render_depth_gpu(verts, mesh["faces"], cam_pos=cam["position"], R=cam["R"], intr=cam["intrinsics"])
    t_gpu_first = time.time() - t0  # 첫 호출은 CUDA warm-up 포함

    t0 = time.time()
    d_gpu2 = render_depth_gpu(verts, mesh["faces"], cam_pos=cam["position"], R=cam["R"], intr=cam["intrinsics"])
    t_gpu_warm = time.time() - t0

    mask_cpu, mask_gpu = d_cpu > 0, d_gpu2 > 0
    inter = (mask_cpu & mask_gpu).sum()
    union = (mask_cpu | mask_gpu).sum()
    iou = inter / union if union > 0 else float("nan")
    common = mask_cpu & mask_gpu
    depth_mae = np.abs(d_cpu[common].astype(np.float64) - d_gpu2[common].astype(np.float64)).mean() if common.any() else float("nan")

    print(f"CPU(Python loop) 렌더 시간: {t_cpu:.4f}초")
    print(f"GPU(nvdiffrast) 첫 호출(warm-up 포함): {t_gpu_first:.4f}초")
    print(f"GPU(nvdiffrast) 이후 호출: {t_gpu_warm:.4f}초")
    print(f"CPU vs GPU silhouette IoU: {iou:.6f}")
    print(f"CPU vs GPU depth MAE: {depth_mae*1000:.6f}mm")
    print(f"속도 향상(warm 기준): {t_cpu/t_gpu_warm:.1f}배")
