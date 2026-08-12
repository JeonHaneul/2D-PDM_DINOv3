"""USD asset에서 실제 크기(미터) 단위의 mesh(정점+삼각형)를 추출한다.
occlusion GT를 "실제로 놓고 촬영"하는 대신 "3D 모델로 계산"하기 위한 첫 단계 -- 여기서 만든
world-space mesh를 depth_rasterizer.py가 카메라 pose마다 depth로 렌더링한다.

핵심 주의점 (USD를 다룰 때 실수하기 쉬운 부분):
  - USD의 스케일은 stage 전체의 metersPerUnit과, prim 계층마다 걸릴 수 있는 Xform scale이 같이
    작용한다. 이 둘을 다 곱해야 "진짜 미터 단위" 좌표가 나온다.
  - 자산 하나(usd 파일)에 mesh prim이 여러 개 있을 수 있다(예: 몸통 + 뚜껑). 전부 모아야 함.
  - face가 삼각형이 아닐 수 있어서(사각형 등) fan triangulation이 필요하다.
"""
import numpy as np
from pxr import Usd, UsdGeom, Gf


def _mesh_local_to_world_matrix(mesh_prim, meters_per_unit: float) -> np.ndarray:
    """mesh prim의 local->world 4x4 변환행렬(미터 단위)을 구한다.
    UsdGeom.Xformable.ComputeLocalToWorldTransform()이 이미 조상 prim들의 Xform을 전부
    합성해주므로(nested transform 문제를 직접 안 풀어도 됨), 여기에 stage의 metersPerUnit만
    곱하면 된다."""
    xformable = UsdGeom.Xformable(mesh_prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Gf.Matrix4d는 row-vector 관례(v' = v * M)라서 numpy로 옮길 때 그대로 두고 vertex도
    # row-vector로 곱한다(아래 transform_points 참고).
    mat = np.array(world_transform, dtype=np.float64)  # (4,4), row-vector 관례
    mat[:3, :3] *= meters_per_unit  # scale 성분에 unit 변환 적용
    mat[3, :3] *= meters_per_unit   # translate 성분(4번째 행)에도 적용
    return mat


def _transform_points(points_local: np.ndarray, mat_row_vector: np.ndarray) -> np.ndarray:
    """points_local: (N,3). mat_row_vector: USD 관례(row-vector, v'=v*M)의 (4,4) 행렬."""
    n = points_local.shape[0]
    homo = np.concatenate([points_local, np.ones((n, 1))], axis=1)  # (N,4)
    world = homo @ mat_row_vector  # (N,4) = (N,4) @ (4,4), row-vector 관례
    return world[:, :3] / world[:, 3:4]


def _triangulate_faces(face_vertex_counts, face_vertex_indices) -> np.ndarray:
    """USD의 general polygon face를 fan triangulation으로 삼각형 인덱스 배열 (M,3)으로 변환.
    face가 이미 삼각형(count=3)이면 그대로, 사각형 이상이면 첫 정점 기준으로 부채꼴 분할."""
    triangles = []
    offset = 0
    for count in face_vertex_counts:
        idx = face_vertex_indices[offset: offset + count]
        for i in range(1, count - 1):
            triangles.append((idx[0], idx[i], idx[i + 1]))
        offset += count
    return np.array(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)


def extract_world_mesh(usd_path: str, prim_path: str = None) -> dict:
    """USD 파일을 열어서, 그 안의 모든 UsdGeom.Mesh prim을 실제 미터 단위 world-space로 합쳐
    반환한다.

    Returns:
        {"vertices": (N,3) float64 world-space 미터 좌표,
         "faces": (M,3) int64 삼각형 정점 인덱스,
         "bbox_min": (3,), "bbox_max": (3,)}
    """
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(f"USD 파일을 열 수 없음: {usd_path}")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)  # 보통 1.0(미터) 또는 0.01(cm) 등

    root = stage.GetPrimAtPath(prim_path) if prim_path else stage.GetPseudoRoot()

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        points_attr = mesh.GetPointsAttr()
        if not points_attr or not points_attr.HasValue():
            continue
        points_local = np.array(points_attr.Get(), dtype=np.float64)  # (N,3), mesh local space
        if points_local.shape[0] == 0:
            continue

        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not face_counts or not face_indices:
            continue

        mat = _mesh_local_to_world_matrix(prim, meters_per_unit)
        points_world = _transform_points(points_local, mat)

        tris_local_idx = _triangulate_faces(face_counts, face_indices)
        if tris_local_idx.shape[0] == 0:
            continue

        all_vertices.append(points_world)
        all_faces.append(tris_local_idx + vertex_offset)
        vertex_offset += points_world.shape[0]

    if not all_vertices:
        raise RuntimeError(f"'{usd_path}'에서 mesh를 하나도 못 찾음 (prim_path={prim_path})")

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    return {
        "vertices": vertices,
        "faces": faces,
        "bbox_min": vertices.min(axis=0),
        "bbox_max": vertices.max(axis=0),
    }


def place_mesh(vertices: np.ndarray, x: float, y: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    """mesh를 world (x,y,z)에 놓고 z축 기준 yaw_deg만큼 회전(scene_generator의 set_all_targets와
    동일한 convention: 물체 local 원점 기준 z-회전 후 평행이동)."""
    yaw_rad = np.radians(yaw_deg)
    cos_a, sin_a = np.cos(yaw_rad), np.sin(yaw_rad)
    rot_z = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    rotated = vertices @ rot_z.T
    return rotated + np.array([x, y, z])


def scale_mesh_bottom_center(vertices: np.ndarray, scale: float) -> np.ndarray:
    """물체 바닥 중심을 기준점(anchor)으로 삼아 scale배 확대/축소.
    anchor = (x,y는 bbox 중심, z는 bbox 최솟값=바닥) -- 이 기준으로 늘려야 커진 물체가 바닥에서
    뜨거나 파묻히지 않는다."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    anchor = np.array([
        (bbox_min[0] + bbox_max[0]) / 2.0,
        (bbox_min[1] + bbox_max[1]) / 2.0,
        bbox_min[2],
    ])
    return anchor + scale * (vertices - anchor)


if __name__ == "__main__":
    # 스모크 테스트: packaged_food_2의 target USD가 실제로 합리적인 크기(미터 단위)로 나오는지 확인.
    # 010_potted_meat_can(스팸 캔 종류)이니 대략 10cm 안팎이어야 정상.
    usd_path = "/home/haneul/isaacsim/src/asset/Packaged_food/packaged_food_2/010_potted_meat_can.usd"
    mesh = extract_world_mesh(usd_path)
    size = mesh["bbox_max"] - mesh["bbox_min"]
    print(f"vertices={mesh['vertices'].shape[0]}  faces={mesh['faces'].shape[0]}")
    print(f"bbox_min={mesh['bbox_min']}  bbox_max={mesh['bbox_max']}")
    print(f"size(m)={size}  (참고: 실제 potted meat can은 대략 8~10cm 크기)")

    scaled = scale_mesh_bottom_center(mesh["vertices"], 1.3)
    scaled_size = scaled.max(axis=0) - scaled.min(axis=0)
    print(f"1.3배 스케일 후 size(m)={scaled_size} (원래의 1.3배가 나와야 함: {size * 1.3})")
    bottom_z_before = mesh["vertices"][:, 2].min()
    bottom_z_after = scaled[:, 2].min()
    print(f"바닥 z: before={bottom_z_before:.5f}  after={bottom_z_after:.5f} (같아야 함, bottom anchor 검증)")
