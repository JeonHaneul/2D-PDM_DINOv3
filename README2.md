# 2D-PDM

### Zero-Shot Probability Distribution Mapping for Occluded Object Search in Cluttered Drawers

> **Research in progress**  
> RGB-D 관측으로 가려진 target object의 위치를 추론하고, 탐색 행동을 위한 pixel-wise probability map 생성

---

## Overview

사람의 물체 탐색에 사용되는 세 단서인 유사도, 가림 가능성, 장면 복잡도를 독립적인 probability stream으로 모델링함.

| Stream | 핵심 질문 | 출력 |
|---|---|---|
| **Similarity** | 타겟과 시각적·의미적으로 관련된 물체는 어디에 있는가? | Similarity feature `F_S` |
| **Occlusion** | 타겟이 다른 물체 아래에 물리적으로 가려질 수 있는가? | Occlusion feature `F_O` |
| **Complexity** | 해당 영역의 물체 더미가 얼마나 조밀하고 복잡한가? | Complexity feature `F_C` |

```mermaid
flowchart LR
    RGB["Scene RGB"] --> S["Similarity Stream"]
    TARGET["Target Image + Prompt"] --> S

    RGB --> O["Occlusion Stream"]
    DEPTH["Scene Depth"] --> O
    GEOM["Target Depth / Geometry"] --> O

    RGB --> C["Complexity Stream"]
    DEPTH --> C

    S --> FS["F_S"]
    O --> FO["F_O"]
    C --> FC["F_C"]

    FS --> CONCAT["Feature Concatenation"]
    FO --> CONCAT
    FC --> CONCAT
    CONCAT --> GATE["Fusion Gate"]
    GATE --> DECODER["Decoder"]
    DECODER --> PDM["2D-PDM"]
    PDM --> POLICY["Exploration Policy"]
```

```text
F_fuse = FusionGate(Concat(F_S, F_O, F_C))
P_2D   = Sigmoid(Decoder(F_fuse))
```

최종 `P_2D`는 target이 존재할 확률이 높은 영역을 나타내며, DRL이 확률이 높은 영역부터 우선 탐색할 수 있도록 사용됨.

---

## From Shelf Search to Drawer Search

기존 선반 환경 연구를 비정형 drawer 환경으로 확장함.

> H. Jeon et al., *A study on deep reinforcement learning-based exploration intelligence for occluded object search*, Engineering Applications of Artificial Intelligence, 2026.

기존 연구는 similarity와 occlusion 기반 column-wise distribution을 사용함. 물체 유사도를 수동 정의한 category score에 의존하여 unseen object에 대한 zero-shot 탐색이 불가능했음.

주요 확장:

- 정규적인 shelf column에서 **비정형 cluttered drawer**로 확장
- Column-wise distribution에서 **pixel-wise 2D-PDM**으로 확장
- DINOv3의 dense appearance와 SigLIP의 language-aligned semantics 결합
- 학습하지 않은 target instance를 입력할 수 있는 zero-shot target conditioning
- Similarity와 Occlusion에 scene-level **Complexity stream** 추가

---

## Similarity Stream

> **Status: Implemented**

Similarity stream은 target 및 의미적으로 관련된 물체 영역을 활성화함. DINOv3의 dense appearance와 SigLIP의 image/text semantics를 결합하여 instance-level 외형과 category-level 의미를 함께 표현함.

### Design Rationale

| Component | 담당 정보 | 필요한 이유 |
|---|---|---|
| **DINOv3 Scene Encoder** | 위치가 보존된 dense visual feature | Scene의 어느 patch에 어떤 형상·질감·appearance가 있는지 표현 |
| **DINOv3 Target Encoder** | Target-specific appearance | “이 물체가 어떻게 생겼는가?”를 layer별 query로 표현 |
| **SigLIP Vision Encoder** | Target image semantics | 외형을 넘어선 open-vocabulary 개념 표현 |
| **SigLIP Text Encoder** | Target category semantics | 이미지가 모호해도 이름과 category 정보로 의미를 보완 |
| **Matching Head** | Scene–target spatial relation | Appearance, semantics, cosine cue를 함께 해석해 probability map 생성 |

### Architecture

```mermaid
flowchart TB
    subgraph SCENE["Scene Branch"]
        SRGB["Scene RGB"]
        SDINO["DINOv3 ViT-B/16<br/>Frozen"]
        SFEAT["Layers 2, 5, 8, 11<br/>X_s^l: B × 768 × H_p × W_p"]
        SRGB --> SDINO --> SFEAT
    end

    subgraph APPEARANCE["Target Appearance Branch"]
        TRGB["Target RGB + Mask"]
        CROP["Masked Crop<br/>224 × 224"]
        TDINO["DINOv3 ViT-B/16<br/>Frozen"]
        POOL["Mask-weighted Pooling"]
        AVEC["a_t^l: B × 768"]
        TRGB --> CROP --> TDINO --> POOL --> AVEC
    end

    subgraph SEMANTIC["Target Semantic Branch"]
        SIGIMG["SigLIP Vision<br/>Frozen"]
        PROMPT["a photo of a category"]
        SIGTXT["SigLIP Text<br/>Frozen"]
        SEMFUSE["Image–Text Fusion<br/>1152-D"]
        PROJ["Layer-wise Projection<br/>1152 → 768<br/>Trainable"]
        SVEC["s_t^l: B × 768"]
        CROP --> SIGIMG --> SEMFUSE
        PROMPT --> SIGTXT --> SEMFUSE
        SEMFUSE --> PROJ --> SVEC
    end

    AVEC --> QFUSE["q_t^l = a_t^l + s_t^l"]
    SVEC --> QFUSE
    SFEAT --> COS["Patch-wise Cosine Similarity"]
    QFUSE --> COS
    SFEAT --> INTERACT["Concat<br/>Scene + Target + Cosine"]
    QFUSE --> INTERACT
    COS --> INTERACT
    INTERACT --> BLOCKS["MatchingBlock × 4"]
    BLOCKS --> LFUSE["Multi-layer Fusion"]
    LFUSE --> HEAD["Similarity Head"]
    HEAD --> SIGMOID["Sigmoid"]
    SIGMOID --> PMAP["Similarity Map P_S"]
```

### Model Specification

| Stage | Module | Input | Operation | Output | State |
|---:|---|---|---|---|---|
| 1 | Target preprocessing | Target RGB, mask | Crop, resize, mask alignment | `224 × 224` target crop | Fixed |
| 2 | Scene encoder | Scene RGB | DINOv3 ViT-B/16 layers `2, 5, 8, 11` | `X_s^l: B × 768 × H_p × W_p` | Frozen |
| 3 | Target encoder | Target crop, mask | DINOv3 + mask-weighted pooling | `a_t^l: B × 768` | Frozen |
| 4 | SigLIP vision | Target crop | Image encoding, L2 normalization | `s_img: B × 1152` | Frozen |
| 5 | SigLIP text | Category prompt | Text encoding, L2 normalization | `s_text: B × 1152` | Frozen |
| 6 | Semantic fusion | `s_img`, `s_text` | Average fusion, L2 normalization | `s: B × 1152` | Fixed |
| 7 | Semantic adapter | `s` | Independent `Linear(1152, 768)` per layer | `s_t^l: B × 768` | **Trainable** |
| 8 | Target fusion | `a_t^l`, `s_t^l` | Element-wise addition | `q_t^l: B × 768` | Fixed |
| 9 | Cosine matching | `X_s^l`, `q_t^l` | Patch-wise cosine, range shift | `c_hat^l: B × 1 × H_p × W_p` | Fixed |
| 10 | Interaction | Scene, target, cosine | Channel concatenation | `Z^l: B × 1537 × H_p × W_p` | Fixed |
| 11 | MatchingBlock | `Z^l` | `3×3 Conv → GN → ReLU → 1×1 Conv → GN → ReLU` | `F_l: B × 64 × H_p × W_p` | **Trainable** |
| 12 | Layer fusion | Four `F_l` tensors | Concat + 1×1 convolution | `F_S: B × 64 × H_p × W_p` | **Trainable** |
| 13 | Output head | `F_S` | 1×1 convolution + sigmoid | `P_S: B × 1 × H_p × W_p` | **Trainable** |
| 14 | Reconstruction | `P_S` | Bilinear interpolation | Full-resolution similarity map | Fixed |

### Target Representation

Segmentation mask 내부 patch만 pooling하여 target appearance 계산.

```text
a_t^l = L2Norm(
    Sum[M_t(u,v) · X_t^l(:,u,v)] /
    (Sum[M_t(u,v)] + epsilon)
)
```

SigLIP image/text embedding을 동일한 semantic space에서 결합.

```text
s_img  = L2Norm(SigLIP_Vision(target_crop))
s_text = L2Norm(SigLIP_Text("a photo of a {category}"))
s      = L2Norm((s_img + s_text) / 2)
s_t^l  = W_l · s + b_l
```

Appearance와 semantics를 합산하여 최종 target query 생성.

```text
q_t^l = a_t^l + s_t^l
```

| Vector | 의미 |
|---|---|
| `a_t^l` | Target이 시각적으로 어떻게 생겼는가 |
| `s_t^l` | Target이 의미적으로 어떤 category에 속하는가 |
| `q_t^l` | Appearance와 semantics가 결합된 layer-wise target condition |

### Scene–Target Interaction

Scene patch와 target query의 cosine similarity를 계산한 뒤 `[0, 1]`로 변환.

```text
c^l(u,v)     = CosineSimilarity(X_s^l(:,u,v), q_t^l)
c_hat^l(u,v) = (c^l(u,v) + 1) / 2
```

Scalar cosine score의 정보 손실을 보완하기 위해 scene feature와 target query를 함께 concat.

```text
Z^l(u,v) = Concat[
    X_s^l(:,u,v),   # scene appearance: 768 channels
    q_t^l,          # target condition: 768 channels
    c_hat^l(u,v)    # explicit matching cue: 1 channel
]
```

각 DINOv3 layer를 독립적인 MatchingBlock으로 처리한 뒤 channel 방향으로 결합.

```text
F_l     = MatchingBlock_l(Z^l)
F_S     = Fuse(Concat[F_2, F_5, F_8, F_11])
L_head  = Head(F_S)
P_S     = Sigmoid(L_head)
```

### Trainable Parameters

DINOv3와 SigLIP은 고정하고 semantic adapter와 matching head만 학습.

| Module | Parameters | State |
|---|---:|---|
| DINOv3 ViT-B/16 | Backbone parameters | Frozen |
| SigLIP SO400M | Encoder parameters | Frozen |
| Semantic projection `Linear(1152, 768) × 4` | 3,542,016 | **Trainable** |
| MatchingBlock × 4 | 3,559,168 | **Trainable** |
| Multi-layer fusion | 16,576 | **Trainable** |
| Similarity head | 65 | **Trainable** |
| **Total trainable** | **7,117,825** | |

Checkpoint에는 task-specific layer만 저장하며, frozen backbone은 별도로 로드.

---

## Ground-Truth Similarity Map

Similarity GT는 scene segmentation과 asset category로 생성. 기존 연구와의 비교를 위해 명시적인 category relation 유지.

| Target–scene relation | Score |
|---|---:|
| Exact target instance | `1.0` |
| Same category | `0.8` |
| Related category | `0.5` |
| Other category | `0.2` |
| Background / unknown | `0.0` |

Category-level score:

| Target ↓ / Scene → | Book | Toy | Fruit | Packaged food |
|---|---:|---:|---:|---:|
| **Book** | 0.8 | 0.5 | 0.2 | 0.2 |
| **Toy** | 0.5 | 0.8 | 0.2 | 0.2 |
| **Fruit** | 0.2 | 0.2 | 0.8 | 0.5 |
| **Packaged food** | 0.2 | 0.2 | 0.5 | 0.8 |

Precomputed grayscale GT를 우선 사용하며, cache가 없으면 segmentation과 scene mapping으로 생성. 학습 시 DINOv3 patch resolution으로 average pooling.

```text
Y_patch = AvgPool2D(Y_full, kernel=16, stride=16)
L_sim   = MSE(P_S, Y_patch)
```

> 학습 GT의 category relation과 zero-shot evaluation은 별개임. Unseen target은 DINOv3와 SigLIP으로 직접 인코딩되며 해당 instance는 학습에서 제외.

---

## Training Protocol

### Data Split

동일 scene의 여러 camera view가 train/validation에 섞이지 않도록 `scene_id` 단위로 분할.

| Setting | Value |
|---|---|
| Train / validation | `80% / 20%` per target |
| Scene cameras | center, top, left, right, bottom |
| Target cameras | center, top, left, right, bottom |
| Environment stride | `10` |
| DINOv3 layers | `2, 5, 8, 11` |
| Batch size | `128` |
| Epochs | `100` |
| Optimizer | AdamW |
| Initial learning rate | `1e-3` |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience `5`, relative improvement `5%` |
| Objective | Patch-resolution MSE |

### Cached and Trainable Paths

```mermaid
flowchart LR
    SCENE["Scene RGB"] --> DINO_S["DINOv3<br/>Frozen"] --> SF["Scene Features"]
    TARGET["Target RGB"] --> DINO_T["DINOv3<br/>Frozen"] --> ACACHE["Appearance Cache"]
    TARGET --> SIG["SigLIP<br/>Frozen"] --> SCACHE["Semantic Cache"]
    SCACHE --> PROJ["Semantic Projection<br/>Gradient"]
    ACACHE --> ADD["Appearance + Semantics"]
    PROJ --> ADD
    SF --> HEAD["Matching Head<br/>Gradient"]
    ADD --> HEAD
    HEAD --> LOSS["MSE Loss"]
```

- Target별 5개 camera view의 DINOv3 appearance 사전 계산
- 학습 시 camera view 무작위 선택
- Validation 시 camera view 고정 순회
- Target별 SigLIP image/text embedding caching
- Gradient 유지를 위해 semantic projection은 training step 내부에서 적용

### Evaluation and Checkpoints

평가 지표:

- Mean squared error
- Pixel accuracy
- Balanced accuracy
- Intersection over Union

Checkpoint에는 trainable module만 저장.

```python
{
    "model_state": similarity_model.state_dict(),
    "semantic_proj_state": semantic_projection.state_dict(),
}
```

---

## Zero-Shot Target Conditioning

Target image를 query로 직접 인코딩하므로 unseen object 입력 가능.

| Mode | Target input | 특징 |
|---|---|---|
| **Image-only** | Target RGB | 별도의 category label 없이 visual/semantic image embedding 사용 가능 |
| **Image + text** | Target RGB + prompt | 물체 이름이나 category semantics로 모호한 외형을 보완 |

현재 category prompt 사용:

```text
a photo of a {category}
```

### Qualitative Similarity Maps

![Book target similarity result](img/panel_Book-Book_1_scene00002_env0168_top.png)
![Avocado target similarity result](img/panel_Fruit-Avocado_scene00005_env0224_right.png)
![Orange target similarity result](img/panel_Fruit-Orange_scene00003_env0274_center.png)

### Unseen Banana

Unseen banana 입력 시 fruit 영역 활성화 확인.

- DINOv3: target–scene dense appearance
- SigLIP: banana–fruit open-vocabulary semantics

> **Planned evaluation:** object-held-out, category-held-out, DINOv3-only, SigLIP-only, image-only, image+text ablation.

---

## Occlusion Stream

> **Status: GT generation pilot / Model design completed**

Occlusion stream은 target의 크기와 형상을 고려하여 물체 더미 아래에 가려질 가능성이 높은 영역을 예측함.

### Occlusion GT Generation

학습 GT는 target을 모든 pose에서 직접 촬영하는 대신 USD/OBJ mesh로 target depth를 계산하여 생성. 기존 pose grid와 depth 판정 기준을 유지하되, 중간 RGB·depth·mask를 대량 저장하지 않음.

```text
Target mesh + scale + candidate pose
                ↓
        Rendered target depth
                ↓
Compare with scene depth
                ↓
Occluded-pixel ratio ≥ threshold
                ↓
Legacy GT + normalized probability GT
```

확률 GT는 해당 위치를 target이 덮는 전체 후보 pose 중 충분히 가려지는 pose의 비율로 정의. Scene별 min–max normalization과 visible-target 강조는 사용하지 않음.

```text
P_O(u,v) = N_occluded(u,v) / (N_candidate(u,v) + epsilon)
```

Mesh와 target 입력은 같은 scale로 구성하며, pilot에서는 `0.7`, `1.0`, `1.3`을 사용. Target RGB와 mask는 bottom-center를 기준으로 scale을 맞춤.

### Zero-Shot Occlusion Model

```mermaid
flowchart LR
    SRGB["Scene RGB"] --> DINO_S["DINOv3<br/>Frozen"] --> FRGB["RGB Features"]
    SDEPTH["Scene Depth + Valid Mask"] --> DENC["Depth Encoder<br/>ResNet-18"] --> FDEPTH["Depth Features"]

    TRGB["Target RGB"] --> DINO_T["DINOv3<br/>Frozen"] --> TAPP["Appearance"]
    TMASK["Target Mask"] --> SHAPE["Silhouette Encoder"] --> TGEO["Size + Shape Condition"]
    TMASK --> SIZE["BBox / Area / Aspect Ratio"] --> TGEO

    TGEO --> FILM["Depth-only FiLM"]
    FDEPTH --> FILM
    FRGB --> MATCH["MatchingBlocks × 4"]
    FILM --> MATCH
    TAPP --> MATCH
    TGEO --> MATCH
    MATCH --> HEAD["Fusion + Occlusion Head"] --> PO["Occlusion Map P_O"]
```

FiLM은 size·silhouette condition으로 depth feature만 조절함.

```text
gamma, beta = MLP(E_size, E_shape)
F_depth'    = gamma * F_depth + beta
```

Target appearance는 depth를 조절하지 않고 MatchingBlock에서 scene RGB-D feature와 결합. DINOv3는 frozen으로 유지하며 depth encoder, silhouette/size encoder, FiLM generator, MatchingBlocks와 output head는 하나의 loss로 함께 학습함.

### Deployment Inputs

매 추론 시 입력은 scene RGB, scene depth, target RGB 세 가지. Target mask는 고정된 target 촬영 환경의 empty-background reference를 이용해 내부 전처리에서 자동 생성함.

| 구분 | 필요 정보 |
|---|---|
| 매 추론 입력 | Scene RGB, scene depth, target RGB |
| 고정 시스템 자산 | Empty-background RGB, camera calibration |
| GT 생성에만 사용 | Target USD/OBJ, mesh scale, occlusion GT |

Zero-shot 성능은 frozen encoder만으로 가정하지 않고, 학습에 포함되지 않은 target instance와 scale을 분리하여 평가함.

---

## Complexity Stream

> **Status: Formulation in progress**

Complexity stream은 object density, overlap, depth irregularity를 표현하는 target-independent scene prior.

| Cue | 의미 |
|---|---|
| Local object density | 단위 면적에 포함된 instance 수 |
| Local depth variance | 물체의 높이와 적층 변화 |
| Depth discontinuity | 물체 경계와 급격한 깊이 변화 |
| Overlap structure | 물체 간 가림과 적층 정도 |

```text
Y_C = lambda_n · ObjectDensity
    + lambda_d · LocalDepthVariance
    + lambda_e · DepthEdgeDensity
```

Fusion gate에서 Similarity·Occlusion evidence와 함께 Complexity의 상대적 중요도 조절.

---

## Project Status

| Component | Status |
|---|---|
| Drawer RGB-D and target-reference dataset | Complete |
| DINOv3 ViT-B/16 dense similarity baseline | Complete |
| DINOv3 + SigLIP Similarity stream | Complete |
| Unseen banana qualitative test | Complete |
| Cosine shortcut ablation | Evaluated and excluded |
| Object/category-held-out benchmark | Planned |
| Occlusion stream training | Planned |
| Complexity stream training | Planned |
| Three-stream fusion | Planned |
| Exploration policy integration | Planned |
| Sim-to-real drawer experiment | Planned |

### Core Files

| File | Purpose |
|---|---|
| `backbone.py` | Frozen DINOv3 multi-layer feature extractor |
| `target_utils.py` | Target crop, mask alignment, masked feature pooling |
| `similarity_model.py` | Scene–target interaction, MatchingBlocks, similarity head |
| `train_similarity_v2.py` | Multi-target DINOv3 + SigLIP training pipeline |
| `train_common.py` | Scene split, target caches, metrics, early stopping |
| `gt_similarity.py` | Category-based similarity-map generation |
| `precompute_gt.py` | Precomputed similarity GT cache |
| `paths_config.py` | Model, asset, dataset path configuration |

---

## Roadmap

- [x] Drawer scene and target RGB-D acquisition
- [x] Multi-layer DINOv3 ViT-B/16 representation
- [x] SigLIP image/text semantic fusion
- [x] Pixel-wise Similarity map training
- [x] Unseen target qualitative evaluation
- [x] Cosine shortcut evaluation and removal
- [x] Reproducible scene-level split seed
- [ ] Held-out object and category benchmark
- [ ] Occlusion stream
- [ ] Complexity stream
- [ ] Learned three-stream fusion
- [ ] DRL-based exploration policy
- [ ] Sim-to-real validation

---

## Development Log

Similarity stream의 가설, 실험 결과, 한계, 수정 사항을 단계별로 기록.

### 2026-07-21 · Phase 1 — DINOv3 Appearance Matching

**Reference:** `code_260721/train_similarity.py`

**가설:** Frozen DINOv3 feature 비교만으로 unseen target의 zero-shot similarity map 생성 가능.

```text
Scene RGB  → DINOv3 patch features X_s^l
Target RGB → DINOv3 masked-pooled vector a_t^l

c_hat^l = ShiftTo01(CosineSimilarity(X_s^l, a_t^l))
Z^l     = Concat[X_s^l, a_t^l, c_hat^l]
P_S     = CNN_Head(Z^l)
```

**결과:** 색상·재질·형상 등 appearance가 유사한 영역은 탐지했으나, category-level semantic relation은 표현하지 못함.

**결론:** Dense appearance matching만으로 target–category–scene 관계 표현 불가.

---

### 2026-07-27 · Phase 2 — CLS Prototype Category Conditioning

**Reference:** `code_260727/train_similarity.py`, `code_260727/train_common.py`

**가설:** Image-level CLS token을 category prototype으로 사용하면 patch feature보다 추상적인 category 정보 제공 가능.

Category별 CLS vector 평균으로 네 개의 prototype을 구성하고 unseen target CLS와 cosine similarity 계산.

```text
prototype_k = L2Norm(Mean[CLS(target_i) | category_i = k])

category_prob = Softmax(
    CosineSimilarity(CLS(unseen_target), prototype_k) / temperature
)
```

정답 누설 방지를 위해 leave-one-out prototype 사용. Category 확률을 spatial location에 broadcast한 뒤 interaction feature에 concat.

```text
Z^l = Concat[
    scene patch,
    target appearance,
    patch-wise cosine,
    category probability
]
```

**결과:** Category prior는 제공했으나 prototype도 DINOv3 appearance history의 평균이므로 외부 semantic grounding이 없음. 기존 prototype과 외형 차이가 큰 unseen object에서 category 추론 불안정.

**결론:** CLS prototype만으로 open-vocabulary semantics 확보 불가.

---

### 2026-07-28 · Phase 3 — DINOv3 + SigLIP Semantic Fusion

**Reference:** `code_260728_ver2/train_similarity_v2.py`

**수정:** DINOv3의 dense spatial representation을 유지하고, SigLIP의 language-aligned semantics를 target query에 추가.

```text
DINO appearance : a_t^l ∈ R^768
SigLIP semantics: s ∈ R^1152
Projection      : s_t^l = W_l · s + b_l
Target query    : q_t^l = a_t^l + s_t^l
```

SigLIP image/text embedding을 평균하고 layer별 projection으로 DINOv3 차원에 정렬. 두 backbone은 frozen으로 유지하고 projection과 matching head만 학습.

**결과:** 학습에 포함되지 않은 packaged-food target에서 same-category 영역 활성화 확인.

![Unseen packaged-food target: image-only result](img/packaged_food_5_zeroshot_nolabel_2.png)
![Unseen packaged-food target: image-and-text result](img/packaged_food_5_zeroshot_v2.png)

**결론:** SigLIP 결합으로 category-level zero-shot activation 확보.

---

### 2026-07-28–29 · Phase 4 — Exact-Instance Shortcut Evaluation

**문제:** SigLIP 결합으로 unseen target의 category-level activation은 확보했으나, visible exact target이 same-category object보다 높게 출력되지 않는 사례 확인.

**실험:** DINOv3 cosine을 output logit에 직접 더하는 residual shortcut과 layer `2 + 5` 선택 방식을 평가. Global pooling, raw appearance cosine, visibility, patch matching도 함께 분석.

**결과:** 공통 held-out scene에서 no-shortcut과 shortcut의 exact-target positive ranking은 각각 `29.0%`, `29.7%`로 거의 동일. Median gap은 소폭 개선됐지만 두 모델 모두 instance-level ranking에 실패. 공간 구조를 사용하지 않는 patch matching도 competitor score를 함께 높여 문제를 해결하지 못함.

**결론:** Shortcut은 계산 비용보다 구조와 해석의 복잡성을 늘리는 반면 실질적인 개선이 제한적이므로 최종 Similarity stream에서 제거. 현재 모델은 DINOv3–SigLIP interaction과 learned matching head만 사용함.

추가로 target별로 서로 다른 무작위 split seed가 생성되던 문제를 수정. 실행마다 seed를 한 번만 확정하여 모든 target의 scene-level train/validation split을 재현할 수 있도록 정리.

---

### 2026-08-04 · Phase 5 — Zero-Shot Occlusion Pipeline Design

**문제:** 기존 방식은 target당 `44,100`개 pose와 5개 camera를 촬영하여 중간 RGB·depth·mask를 대량 저장함. 최종 GT도 유효 pose의 depth-weighted occupancy를 scene별 min–max로 정규화하므로 서로 다른 scene에서 흰색의 절대적 의미가 일정하지 않음.

**GT 설계:** Target을 물리적으로 반복 촬영하는 과정을 USD/OBJ mesh depth 계산으로 대체. 기존 pose grid와 `70%` occlusion 판정은 legacy 재현에 유지하고, 실제 학습용 GT는 다음 확률로 별도 생성함.

```text
P_O(u,v) = N_occluded(u,v) / (N_candidate(u,v) + epsilon)
```

Legacy GT와 probability GT를 동시에 출력하여 기존 방식 재현 여부와 새 정규화 효과를 분리해 평가. Visible-target 강조는 Similarity stream과 역할이 겹치므로 Occlusion GT에서 제외함.

**Scale conditioning:** Mesh scale과 target RGB·mask scale을 반드시 동일하게 구성. 동일한 target 입력에 서로 다른 scale GT를 주면 모델이 scale별 정답의 평균으로 수렴하므로 size-effect를 학습할 수 없음. Pilot은 `0.7`, `1.0`, `1.3`으로 시작함.

**모델 설계:** Scene RGB는 frozen DINOv3, scene depth와 valid mask는 ResNet-18로 처리. Target mask에서 추출한 size·silhouette condition으로 depth feature에만 FiLM을 적용하고, target appearance는 MatchingBlock에서 별도로 결합함. 검증되지 않은 cosine shortcut은 baseline에서 제외함.

**배포 조건:** 매 추론 입력은 scene RGB, scene depth, target RGB. Target mask는 고정 촬영 환경의 empty-background reference로 자동 생성하며, USD/OBJ와 scale 정보는 GT 생성에만 사용함.

**Zero-shot 검증:** Frozen encoder 사용만으로 zero-shot을 가정하지 않고, 학습에서 제외한 target instance와 unseen intermediate scale을 별도 test split으로 평가함.

**다음 단계:** `packaged_food_2`, center camera, scale `1.0`, scene 20~50장으로 mesh-depth 재현 pilot을 먼저 수행. Target silhouette IoU와 depth MAE를 확인한 뒤 legacy/probability GT 생성 및 전체 target 확장 여부를 결정함.
