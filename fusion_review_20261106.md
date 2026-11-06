# Fusion의 stream 활용 검토와 다음 검증 계획

> 2026-11-06 (Asia/Seoul). 기존 구현과 완료된 평가의 재해석임. 이번 문서 갱신에서 새 학습·추론·물리 실행은 수행하지 않음. 아래 지도 기반 fusion과 gating은 미실행 제안이며 채택된 모델이 아님.

[전체 개요](README.md) · [Fusion](fusion_stream.md) · [Development Log](development_log.md#fusion-review-20261106) · [최신 인계](agent.md#current-handoff-20261106) · [비교 수치 JSON](docs/fusion_comparison_20261106.json)

## 검토 배경과 현재 판단

사용자는 결과 그림을 보고, 마지막 CNN이 유사도·가림·복잡도 판단을 활용하기보다 최종 target label만 맞히도록 학습한 것인지 질문함. 세 stream GT를 같은 투명도로 섞고 현재 fusion GT를 더하는 방법, 또는 세 stream의 활용 정도를 학습하는 다른 fusion 구조를 제안함.

현재 fusion은 세 stream의 **최종 1채널 지도가 아니라 각 지도 생성 직전의 64채널 특징**을 입력받음. GT는 학습 오차 계산에만 사용하며 추론 입력이 아님. 따라서 stream 없이 정답만 보고 추론한다는 해석은 부정확함. 그러나 특징에 남은 외형 정보나 특정 stream에 편중될 가능성이 있고, **유사도·가림·복잡도의 의미를 각각 활용하는지는 아직 확인하지 않았음.** 결과 그림의 인상만으로 두 경우를 구분할 수 없음.

이번 요청은 이 판단과 근거·후속 제안을 로컬 및 GitHub에 기록하는 범위임. 후속 실험 실행 승인이나 모델 변경 확정으로 확대하지 않음.

## 실제 구현과 확인된 결과

```text
단일 scene RGB-D + 기존 target reference 조건
  → 고정된 S/O/C 모델
  → F_S, F_O, F_C 각 64×30×40
  → concat 192채널
  → 1×1 Conv64 → 3×3 Conv64 두 층 → 1×1 Conv1
  → sigmoid → target 전체 영역(amodal) 예측 지도
```

중간 층은 GroupNorm8·GELU임. DINOv3·SigLIP·MultiMAE와 세 stream의 parameter·buffer를 고정하고 fusion head만 학습함. GT는 native 640×480 amodal mask를 16×16 평균한 30×40 값이며 BCEWithLogits로 감독함. 실제 S/O/C 지도는 별도 출력이고 위 head의 입력이 아님. Attention이나 명시적 stream gate도 현재 구현에 없음. MultiMAE는 관측 depth를 인코딩하며 RGB에서 depth를 생성하는 모델로 쓰지 않음.

2026-10-27 완료된 공동 실행의 SO/SOC를 같은 기존 16종·5camera test 24,000장에서 비교함. 두 CSV의 sample ID·순서·GT 정보·가시성 분류가 같고 CSV 재집계와 원본 evaluation JSON이 일치함. 학습/validation/test는 192,000/24,000/24,000장이며 새 물체 zero-shot 결과가 아님.

| 평가 | S+O（SO best20） | S+O+C（SOC best10） | SOC−SO |
|---|---:|---:|---:|
| Target 전체 영역의 최고점 적중 /24,000 | 19,862/24,000 = 82.758333% | 20,414/24,000 = 85.058333% | +2.30%p |
| Target 전체 영역의 마스크 mIoU /24,000 | 52.103077% | 50.920985% | −1.182092%p |
| 실제 보이는 target 최고점 적중 /15,235 | 6,062/15,235 = 39.789957% | 6,315/15,235 = 41.450607% | +1.660650%p |

최고점은 원시 30×40 예측을 640×480로 bilinear 확대한 지도의 최고점 한 점임. mIoU는 같은 확대 예측을 0.5로 이진화한 영역과 native amodal GT의 영상별 IoU 평균임. 일부라도 target이 보이는 15,235장에서만 visible 적중을 계산함. 완전 가림 8,755장과 amodal 영역이 없는 10장을 구분하고, 전체 24,000 분모에는 빈 amodal 10장도 포함함. 일반 pixel accuracy나 실제 파지 성공률이 아님.

Complexity를 추가한 SOC는 최고점 적중이 높고 mIoU는 낮음. 단일 seed에서 SO는 20epoch·best20, SOC는 13epoch·best10이며 각 validation 기준으로 선택됨. 입력 projection 채널 수도 다름. 동일 모델에서 C만 끈 실험이 아니므로 C의 의미 활용이나 모든 stream의 필요성을 확정하지 않음. 반올림 전 수치·분모·원본 hash는 비교 JSON을 따름.

기존 32scene×5독립camera×4정책 제거 실험에서 성공 조건의 평균 제거는 Fusion 2.1290, S 3.9355, O 7.7226, C 6.0516회였음. 각 정책 155/160 성공이며 원본부터 target이 바닥 아래에 있는 한 scene의 5뷰가 실패함. **이 실험에는 SO 정책이 없으므로 C를 추가해 제거 횟수가 줄었다고 분리해 해석할 수 없음.** 기존 결과·그림은 [제거 탐색](closed_loop_search.md)에 보존함.

## GT 혼합을 권하지 않는 이유

Similarity는 target과의 유사성, Occlusion은 target이 가려질 가능성, Complexity는 지역 복잡도, fusion GT는 실제 target 전체 영역을 나타냄. 네 값의 의미와 범위가 다름. 같은 투명도로 겹치면 target이 없는 복잡한 영역도 양성 점수가 올라가고, target 영역의 점수는 혼합 비율에 따라 낮아질 수 있음. 이는 세 정보를 유용하게 쓰도록 보장하는 감독이 아니라 목표 자체를 임의의 혼합 점수로 바꾸는 방법임.

현재 amodal GT를 유지하는 것을 제안함. 사용자가 혼합 GT나 대체 방법을 최종 승인한 상태가 아니며, 기존 GT를 수정하지 않음. 세 stream의 비중을 항상 같게 만드는 것 역시 연구 목표가 아님.

## CNN과 attention, 지도 기반 fusion 후보

Concat은 균등 평균이 아니며 CNN도 관측에 따라 다른 출력을 학습함. Attention으로 교체해도 특정 stream 편중이나 외형 정보 의존이 자동으로 해결되지 않음. Attention/gate의 값 자체를 실제 기여도로 단정하지 않음.

다중모달 성능이 단일 입력보다 좋아져도 일부 입력을 충분히 활용하지 않을 수 있다는 문제는 [CVPR 2022 Balanced Multimodal Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html)에서도 다룸. 해당 연구는 주로 audio-visual 학습의 최적화 불균형을 다루므로, backbone과 stream이 고정된 현재 모델의 원인이나 그 논문의 학습 기법 적용 효과를 증명하는 근거로 확대하지 않음.

사용자의 의도에 가까운 **미실행 비교 후보**는 `예측 S/O/C 1채널 지도 → 값 범위 보정 → 작은 fusion 모델 → amodal 지도`임. 이 비교에서는 64채널 특징이나 원본 RGB-D의 별도 우회 입력을 주지 않아 최종 판단이 세 예측 지도를 거치게 함. 입력은 GT 지도가 아닌 모델의 예측 지도이며, 범위 보정은 train에서 정한 규칙으로 고정하고 영상마다 최대값을 1로 바꾸지 않음. Complexity의 표시용 0–6 범위를 이론적 GT 상한으로 취급하지 않음.

먼저 작은 CNN으로 지도 조합의 효과를 비교하고 필요하면 위치별 반영 정도를 학습하는 gating을 검토함. 단순 가중평균을 target 확률로 바로 간주하지 않음. 지도 세 장으로 압축하면서 정보가 줄어 성능이 낮아질 수 있으며, 이 경로만으로 세 의미의 올바른 사용이나 세 stream 모두의 필요성이 보장되는 것은 아님. 현재 배포 구조나 기존 framework 그림을 이 후보로 교체하지 않음.

## 다음 Step — 미실행 제안

1. **현재 모델의 의존성 진단.** 같은 관측에서 S/O/C 특징을 하나씩 교란하여 출력·위치 적중의 변화를 확인함. 0 입력만으로 판정하지 않으며, 다른 sample의 특징 교환도 실제 장면과 불일치하는 입력을 만들므로 성능 하락을 의미 활용의 완전한 증거로 쓰지 않음. 장면을 고정하고 같은 장면의 다른 target으로 reference 조건을 일관되게 바꾸었을 때 해당 target으로 예측이 이동하는지도 확인함. GT는 평가용으로만 사용함.
2. **지도 기반 fusion 비교.** 진단과 사용자 논의 뒤 필요한 경우, 기존 amodal GT·split·backbone 고정을 유지한 채 특징 기반 CNN과 예측 지도만 받는 fusion을 비교함. 필요하면 위치별 gating을 추가 후보로 삼음. 모델 선택·범위 보정은 train/validation에서 결정하고 이미 확인한 test 결과를 반복 튜닝 기준으로 쓰지 않음. 새로운 최종 검증의 범위는 실험 실행 전에 정함.

현재 amodal GT는 **target이 어디 있는가**를 감독하며 **다음에 무엇을 제거해야 빨리 찾는가**를 직접 감독하지 않음. Complexity가 제거 행동에 유용하더라도 위치 예측 loss에서는 추가 가치가 작을 수 있음. 이 차이는 가능한 설명이며 확인된 원인이 아님. 지금 바로 제거 횟수·노출량 GT로 대체하거나 새 데이터 수집·장시간 학습을 시작하는 계획으로 확대하지 않음.

## 다른 agent의 읽기 순서와 원본 경로

웹 agent는 [최신 인계](agent.md#current-handoff-20261106) → 이 문서 → [현재 Fusion](fusion_stream.md) → [비교 수치 JSON](docs/fusion_comparison_20261106.json) 순서로 읽음. 과거 그림과 탐색 결과가 필요하면 [제거 탐색](closed_loop_search.md), 그림 작성 시 [Framework Guide](framework_guide.md)를 확인함.

로컬 agent는 `<DEV_ROOT>/AGENTS.md` → `PROJECT_CONTEXT.md` → `PROJECT_LOG_INDEX.md` → `docs/fusion_review_20261106.md` 순서로 읽음. `<DEV_ROOT>`는 실제 개발 폴더이고 GitHub clone과 다름.

| 로컬 근거 | 확인할 내용 |
|---|---|
| `experiments/fusion_training_20261026/runtime.py` | `extract()`의 실제 head 이전 특징 반환 |
| `experiments/fusion_training_20261026/model.py` | concat/CNN 구조와 amodal BCE |
| `experiments/fusion_training_20261026/train.py` | frozen 추출, head 학습, validation 모델 선택 |
| `outputs/fusion_training_20261026/{protocol,completion}.json` | 고정 조건과 SO/SOC 완료 |
| `outputs/fusion_training_20261026/runs/{so,soc}/evaluation/{evaluation.json,per_image.csv}` | 동일 test 수치와 sample별 근거 |
| `outputs/closed_loop_search_20261027/full/report/summary.json` | SO를 포함하지 않는 네 정책 제거 결과 |

원본 코드·checkpoint·전체 raw 자료는 로컬에 보존되며 위 경로를 GitHub에 전부 공개했다는 뜻이 아님. 기존 완료 결과, 현 구현의 한계, 원인 가설, 미실행 제안을 구분함.
