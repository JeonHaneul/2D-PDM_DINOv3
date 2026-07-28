"""다른 컴퓨터로 옮길 때 여기(이 파일)만 고치면 됩니다.

지금처럼 src/ 폴더(2D-PDM_DINOv3/, model/, asset/, 260714_data/가 형제 폴더)를 통째로
복사하면 아래 계산식이 자동으로 맞는 경로를 잡아주므로 아무것도 안 고쳐도 됩니다.
구조가 다르면(예: 데이터가 NAS/외장 드라이브에 있음) 아래 변수 중 필요한 것만
절대경로 문자열로 직접 덮어쓰면 됩니다."""
import os

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 2D-PDM_DINOv3/의 부모 폴더

MODEL_DIR = os.path.join(_SRC_ROOT, "model")            # DINOv3 로컬 가중치(.pth) 폴더
ASSET_DIR = os.path.join(_SRC_ROOT, "asset")             # asset/<Category>/<instance>/*.usd
DATA_ROOT = os.path.join(_SRC_ROOT, "260714_data")       # 학습 데이터 루트

SCENE_DIR = os.path.join(DATA_ROOT, "scene")             # scene/<target>/{rgb,seg}
TARGET_DIR = os.path.join(DATA_ROOT, "target")           # target/<target>/{rgb,seg,mapping.json}
GT_DIR = os.path.join(DATA_ROOT, "GT_data")              # GT_data/<target>/*.png (사전계산 캐시)

# similarity_map/, occlusion_map/ 폴더는 현재 target당 zip 파일 1개(미압축)뿐이라
# 아직 어떤 코드도 참조하지 않음 -- occlusion 학습을 실제로 시작할 때 추가 예정.
# SIMILARITY_MAP_DIR = os.path.join(DATA_ROOT, "similarity_map")
# OCCLUSION_MAP_DIR = os.path.join(DATA_ROOT, "occlusion_map")
