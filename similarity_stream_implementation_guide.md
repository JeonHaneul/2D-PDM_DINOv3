# Similarity Stream Implementation Guide

## 목적

Cluttered drawer의 scene RGB와 target RGB를 입력받아 target 및 의미적으로 관련된 물체의 위치를 similarity probability map으로 출력합니다. 현재 구현은 frozen DINOv3 ViT-B/16의 dense appearance와 frozen SigLIP의 image/text semantics를 결합합니다.

## 실행 파일

| 파일 | 역할 |
|---|---|
| `backbone.py` | DINOv3 ViT-B/16의 layer 2, 5, 8, 11 feature 추출 |
| `target_utils.py` | Target mask crop, 입력 변환, mask-weighted pooling |
| `similarity_model.py` | Layer별 scene–target interaction과 learned matching head |
| `train_similarity_v2.py` | DINOv3 + SigLIP 학습 |
| `inference_zeroshot.py` | 학습에 사용하지 않은 target 추론 |
| `target_capture.py` | Isaac Sim에서 target RGB/depth/segmentation 촬영 |
| `paths_config.py` | Dataset, asset, DINOv3 weight 경로 설정 |

`train_similarity.py`는 이전 실험을 보존한 파일입니다. 현재 학습 기준은 `train_similarity_v2.py`입니다.

## 입력과 출력

### 학습

- Scene RGB: `260714_data/scene/<target>/rgb/`
- Scene segmentation: `260714_data/scene/<target>/seg/`
- Target RGB/segmentation: `260714_data/target/<target>/`
- Precomputed similarity GT: `260714_data/GT_data/<target>/`

### Zero-shot inference

- Target directory: `rgb/`, `seg/`, `mapping.json`
- Scene RGB 한 장
- 학습된 `similarity_head_best.pt`

출력은 patch-resolution probability map과 원 영상 크기로 보간한 probability map입니다.

## 모델 구조

```text
Scene RGB
  └─ Frozen DINOv3 ViT-B/16
       └─ layer 2, 5, 8, 11 dense patch features X_s^l

Target RGB + mask
  ├─ Frozen DINOv3 + mask-weighted pooling → appearance a_t^l
  └─ Frozen SigLIP image/text encoder
       └─ layer별 projection → semantics s_t^l

q_t^l = a_t^l + s_t^l

Z^l = Concat(
    X_s^l,
    Broadcast(q_t^l),
    ShiftedCosine(X_s^l, q_t^l)
)

Z^l → MatchingBlock_l
F_S = Fuse(F_2, F_5, F_8, F_11)
P_S = Sigmoid(Head(F_S))
```

DINOv3와 SigLIP encoder는 고정합니다. SigLIP projection, layer별 MatchingBlock, fusion 및 output head만 학습합니다.

현재 baseline에는 output logit으로 직접 연결되는 cosine shortcut이 없습니다. Shortcut 및 layer-selective shortcut은 exact-instance 순위를 안정적으로 개선하지 못해 제거했습니다.

## 차원

ViT-B/16과 640×480 입력 기준:

| Tensor | Shape |
|---|---|
| Scene patch feature | `B × 768 × 30 × 40` |
| Target appearance | `B × 768` |
| SigLIP embedding | `B × 1152` |
| Projected semantics | `B × 768` |
| Layer interaction | `B × 1537 × 30 × 40` |
| Similarity map | `B × 1 × 30 × 40` |

## 환경 준비

```bash
pip install -r requirements.txt
```

DINOv3 ViT-B/16 weight는 GitHub에 포함하지 않습니다. 기본 폴더 구조는 다음과 같습니다.

```text
src/
├── 2D-PDM_DINOv3/
├── model/
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── asset/
└── 260714_data/
```

DINOv3 네트워크 정의는 첫 실행 시 공식 `facebookresearch/dinov3` torch hub repository에서 읽으며, weight는 `model/`의 로컬 파일을 사용합니다. SigLIP은 Hugging Face model ID `google/siglip-so400m-patch14-384`를 사용합니다.

## 학습

```bash
python train_similarity_v2.py
```

주요 기본 설정:

- Backbone: DINOv3 ViT-B/16
- Layers: 2, 5, 8, 11
- Patch size: 16
- SigLIP: SO400M patch14-384
- DINOv3/SigLIP: frozen
- Loss: patch-resolution MSE
- Optimizer: AdamW

## Target 촬영

`target_capture.py`는 Isaac Sim Python 환경에서 실행해야 합니다.

```bash
python target_capture.py --target_name fruit_5 --headless
```

폴더 구조가 다르면 명시적으로 지정합니다.

```bash
python target_capture.py \
  --target_name fruit_5 \
  --asset_dir /path/to/asset \
  --output_root /path/to/scene_generator/output \
  --headless
```

## Zero-shot 추론

```bash
python inference_zeroshot.py \
  --checkpoint outputs/<run>/similarity_head_best.pt \
  --target_dir ../scene_generator/output/fruit_5/target \
  --scene_image ../scene_generator/output/fruit_5/scene/rgb/scene00001_env0000_center.png \
  --label banana \
  --out zeroshot_result.png
```

Checkpoint는 다음 항목을 포함해야 합니다.

- `model_state`
- `semantic_proj_state`

Target mask와 `mapping.json`은 target instance를 crop하고 DINOv3 appearance를 pooling하는 데 사용합니다. `--label`은 선택 사항이며 제공하면 SigLIP text hint로 결합합니다.

## 검증 항목

- Target instance가 train split에 포함되지 않은 object-held-out 평가
- Category 전체가 제외된 category-held-out 평가
- DINOv3-only, SigLIP image-only, image+text ablation
- Exact target과 same-category object의 순위 비교
- Target camera 변화에 대한 안정성
- Checkpoint와 모델 구조의 호환성

Zero-shot은 frozen encoder를 사용했다는 사실만으로 성립하지 않습니다. 학습에 사용하지 않은 target을 별도로 분리해 평가해야 합니다.
