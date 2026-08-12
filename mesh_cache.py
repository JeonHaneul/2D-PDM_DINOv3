"""asset별 단순화 mesh를 캐시. 원본 USD 파일의 내용 hash를 키로 써서, USD가 바뀌면
자동으로 재생성하고 안 바뀌었으면 재사용한다(매번 몇십 초 걸리는 pymeshlab 단순화를
반복하지 않기 위함). validate_simplification.py에서 검증된 절차(중복 정점/면 제거,
non-manifold edge 복구 후 quadric edge collapse)를 그대로 사용."""
import hashlib
import json
import os

import numpy as np
import pymeshlab

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_cache")


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _simplify(verts, faces, target_faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True)
    m = ms.current_mesh()
    return np.ascontiguousarray(m.vertex_matrix()), np.ascontiguousarray(m.face_matrix())


def mesh_content_hash(vertices, faces) -> str:
    """단순화된 mesh의 실제 vertex/face 내용 해시. 원본 USD hash와 target_faces가 같아도
    pymeshlab 버전이 바뀌거나 캐시 파일이 수동으로 교체되면 실제 단순화 결과가 달라질 수
    있으므로, checkpoint fingerprint에는 USD hash가 아니라 이 값을 넣어야 함."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(vertices).tobytes())
    h.update(np.ascontiguousarray(faces).tobytes())
    return h.hexdigest()


def get_simplified_mesh(usd_path: str, asset_name: str, target_faces: int, orig_verts=None, orig_faces=None):
    """캐시에 (asset_name, usd_hash, target_faces)와 일치하는 게 있으면 그걸 반환.
    없으면 orig_verts/orig_faces(없으면 extract_world_mesh로 새로 추출)로 단순화 후 캐시에 저장.
    반환: (vertices, faces, was_cached: bool)"""
    usd_hash = _file_hash(usd_path)
    asset_dir = os.path.join(CACHE_DIR, asset_name)
    os.makedirs(asset_dir, exist_ok=True)
    meta_path = os.path.join(asset_dir, "meta.json")
    mesh_path = os.path.join(asset_dir, f"simplified_{target_faces}.npz")

    if os.path.exists(meta_path) and os.path.exists(mesh_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("usd_hash") == usd_hash and meta.get("target_faces") == target_faces:
            data = np.load(mesh_path)
            return data["vertices"], data["faces"], True

    if orig_verts is None or orig_faces is None:
        from mesh_utils import extract_world_mesh
        mesh = extract_world_mesh(usd_path)
        orig_verts, orig_faces = mesh["vertices"], mesh["faces"]

    v_simp, f_simp = _simplify(orig_verts, orig_faces, target_faces)
    np.savez(mesh_path, vertices=v_simp, faces=f_simp)
    with open(meta_path, "w") as f:
        json.dump({"usd_hash": usd_hash, "target_faces": target_faces,
                   "usd_path": usd_path, "orig_faces": int(len(orig_faces)),
                   "simplified_faces": int(len(f_simp))}, f, indent=2)
    return v_simp, f_simp, False
