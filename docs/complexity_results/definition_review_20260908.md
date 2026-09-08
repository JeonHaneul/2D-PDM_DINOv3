# Complexity 정의 재검토 — 2026-09-08

## 판단과 범위

현재 RGB-D v2는 visible count/occupancy 예측 pilot으로 보존한다. 물체 더미 내부의 구조적 복잡도를
검증한 모델로 채택하지 않는다. 22.973%는 count MAE 개선이며, 복잡도 정의의 타당성이나 탐색
효용을 측정한 결과가 아니다. 기존 checkpoint·GT·실험 protocol은 수정하지 않았다.

이번 검토는 코드·기존 GT의 기술 통계와 최근 문헌을 확인한 결과다. 새 GT나 학습 모델은 만들지
않았다. 아래 구조적 복잡도 후보는 제안이며, 검증을 마친 공식 정답이 아니다.

## 현재 GT의 문제

- Occupancy는 물체가 영상 면적을 채운 비율이다. 넓은 평판 하나와 여러 물체가 겹친 곳 모두
  높은 값을 가질 수 있다. 빈 공간/물체 영역 구분 또는 조작 공간 분석에는 사용할 수 있지만,
  더미 내부 복잡도의 정답값으로 쓰는 근거는 부족하다.
- Count는 같은 영역 안의 물체 수를 구분하지만 간격·물체 간 가림·접촉·배치 방향을 구분하지
  않는다. 고정된 큰 window는 주변 count를 여러 위치에 넓게 퍼뜨린다. 모델의 smoothing까지
  더해지지만, GT 정의와 공간 범위의 한계를 모델 확대나 loss 수정으로 해결할 수는 없다.
- 실제 그림의 count GT에는 차이가 있다. 모든 map이 상수이거나 saturation됐다고 주장하지
  않는다. Occupancy의 밝은 영역을 count GT 또는 합성 complexity scalar로 혼동하지 않는다.
- `N/A_heap`도 공식 정답으로 정하지 않는다. 작은 독립 물체 하나의 면적이 작으면 비율이
  커지고, 작은 물체 여러 개를 큰 물체 하나로 대체하면 크기 구성에 따라 비율이 달라진다.
  겹침은 visible N과 projected union area를 함께 바꾸어 어느 방향으로 변할지 보장되지 않는다.
  더미 전체의 비율 하나를 broadcast하면 국소 차이는 사라진다. 영상 면적은 거리·시점에도 의존한다.
- 기존 depth 잔차는 넓은 window에서 물체와 바닥의 높이 차이, 단일 물체의 곡면에도 반응한다.
  Empty-reference 보정은 빈 서랍의 고정 구조 문제를 해결했지만, 이것으로 물체 간 복잡도가
  검증되는 것은 아니다.

[GT 기술 통계](gt_definition_audit_20260908.json): 기존에 평가한 v2 test 영상 960장의 유효
occupancy patch 중 `occupancy>0`인 위치를 모아 보면, **56.721%가 occupancy≥0.95**다.
이는 해당 foreground patch 집합에 대한 비율이며, 전체 영상 pixel 비율이나 scene별 평균이 아니다.
같은 foreground 안에서도 count는 변하며, 96px GT의 10/50/90 percentile은 2/5/7개다
(count-valid이고 occupancy≥0.95인 patch 기준). 이번 수치는 이미 본 GT의 기술적 재검토로,
새로운 독립 test 성능 검증에 해당하지 않는다.

## 최근 5년 문헌: 2021-09-08 ~ 2026-09-08

오래된 원전을 현재 GT의 충분한 근거로 제시하는 방식을 수정한다. 아래는 최근 로봇 연구에서
실제로 어떤 정의를 사용했는지에 대한 비교다. 최신 논문이 과거 지표를 재사용하는 경우도
있으며, 발행 시점과 지표의 최초 제안 시점은 구분한다.

### Domain-Independent Disperse and Pick — IJCNN 2022

RGB의 색·대비·방향 feature의 국소 변동으로 pixel별 Feature Congestion map을 계산한다.
장면 전체 평균과 grasp 후보 주변의 local score를 구분하며, local ROI 지름은 영상에 투영한
그리퍼 opening 크기다. 점수에 따라 grasp와 push를 선택한다. 정답 segmentation의 count를
복잡도로 정의하는 방식이 아니다. [저자 공개 논문 §III-A](https://arxiv.org/html/2312.12637v1#S3.SS1)

우리에 대한 시사점: 실제 행동과 연결된 크기의 국소 영역을 평가하는 선례다. 색·텍스처에 대한
반응이 물체 간 구조와 일치하는지는 별도로 확인해야 한다. 본 프로젝트에는 비교 baseline
후보로 적합하며, 바로 GT로 사용할 충분한 근거는 아니다. 학회 연도는 2022이며 arXiv 업로드는
2023이다. [저자 publication 목록](https://www.cse.iitk.ac.in/users/praj/)

### ARMOR — Autonomous Robots 2025 및 저자 공개 workshop 논문

Pixel 위치에서 각 물체의 크기 s_i와 거리 d_i에 따른 기여를 더하는 `sum_i s_i/d_i` 형태의
clutter map을 사용한다. 공개 workshop 논문은 map 유무를 비교하여 gamma=.90에서 평균
행동 수 20.5→15.4를 보고한다. [저자 공개 방법·비교표](https://autonomousrobots.nl/assets/images/workshops/2025_iros/accepted_papers/paper_5_Autonomous.pdf)
[학술지 출판 기록](https://link.springer.com/article/10.1007/s10514-025-10214-7)

우리에 대한 시사점: 국소 map을 탐색/조작 행동 결과로 검증한 직접적인 선례다. 다만 큰 물체는
단독으로도 높은 기여를 가질 수 있어, 이 공식을 그대로 적용하면 사용자가 지적한 문제가 남을
수 있다. 이는 공식에 대한 본 프로젝트의 해석이다. 25% 수치는 해당 연구의 조건에 한정한다.
학술지 본문 전체는 접근되지 않아 위 수식·수치는 공개 workshop 버전에서 확인했다.

### ClutterDexGrasp — CoRL 2025

개수 4–8/9–15/16–25로 sparse/dense/ultra-dense 장면을 나눈다. 정책의 teacher 표현은 손가락과
목표·주변 물체 표면 사이 거리를 사용하고, student는 단일 camera의 부분 point cloud를 받는다.
[논문 §4.1.1, §5.1](https://arxiv.org/html/2506.14317v2)

우리에 대한 시사점: 개수는 장면 난이도를 나누는 지표로 사용될 수 있다. 그러나 이 연구가
국소 pixel count GT를 검증한 것은 아니다. 가까운 장애물과의 관계를 명시적으로 다루는 부분이
우리의 국소 구조 재검토에 더 관련된다. 이 역시 동일한 GT를 재현한다는 뜻은 아니다.

### Distracted Robot — 2025 arXiv preprint

물체 개수만으로 평가하지 않고 색·대비·방향의 Feature Congestion을 사용한다. Robot view와
별도의 top-down view를 결합한 DvFC로 장면을 나눠 VLA 성공률과 비교한다. 실험 생성에서는
stack/pile을 피하고 목표 가림을 50% 이하로 제한하며 grasp affordance를 유지한다.
[논문 §III-B/C](https://arxiv.org/html/2511.22780v1#S3)

우리에 대한 시사점: 개수와 clutter의 효과를 구분해서 평가한다. 그러나 단일 RGB-D 관측,
심한 적층/완전 가림이 있는 서랍과 관측·실험 조건이 다르므로 DvFC를 그대로 사용할 수 없다.
Preprint이며 현재 검토에서 peer-reviewed 출판은 확인하지 않았다.

## 새 정의의 후보와 stream 역할

우선 검토할 질문은 **고정된 물리적 근방에서 서로 다른 물체의 표면이 얼마나 가까이 놓이고,
가림 경계를 형성하며, 서로 다른 방향·깊이의 구조를 만드는가**다. 이를 관측된 물체 간 국소
구조로 한정한다. 다음 단서는 독립적으로 비교할 후보이며 임의의 가중합을 공식 C_GT로 정하지 않는다.

| 후보 | 기대하는 공간 반응 | 반드시 확인할 반례 |
|---|---|---|
| 서로 다른 물체 사이의 경계·근접 관계 | 여러 물체의 경계가 만나고 간격이 좁은 곳 | 단일 물체 외곽·포장지 무늬, 간격이 충분한 정렬 물체 |
| 국소 가림 관계·관측 surface의 깊이 순서 | 여러 물체가 서로 가리는 접경 부근 | 한 큰 물체의 바닥 경계, depth noise, 투영상 가까우나 실제 떨어진 물체 |
| 물체 간 표면 방향·깊이 변화 | 다른 물체 표면들이 여러 방향으로 이어지는 곳 | 비스듬한 책 한 권, 곡면 과일 하나, 동일 높이로 밀착한 여러 물체 |

Segmentation/instance ID와 camera calibration을 학습 정답 생성에 사용하는 것은 RGB-D 추론
제약과 양립한다. 다만 관측되지 않는 완전 가림 관계를 simulation에서 얻었다고 해서 단일
RGB-D로 항상 알아낼 수 있다고 가정하지 않는다. 관측 가능한 구조와 숨겨진 상태/불확실성을
구분해야 한다. 기존 asset별 색은 동일 asset 복수 instance를 구분하지 못한다.

Occupancy는 계산할 위치와 빈 공간을 구분하는 보조 정보로 취급한다. Count와 N/A는 비교군 또는
보조 지표로 남긴다. 후보 관측 구조는 RGB-D에서 추론하며, 별도 VLM 도입 필요성은 아직 결정되지
않았다. Depth만으로 같은 높이의 다른 물체를 구분하기 어렵고, RGB만으로는 인쇄 무늬가 물체
경계처럼 보일 수 있으므로 양쪽의 역할은 유지한다.

Similarity는 target과의 시각·의미 관계, Occlusion은 해당 target이 가려질 수 있는 가능성을
담당한다. Complexity 후보는 target identity와 무관한 현재 scene 물체 사이 구조를 다룬다.
그리퍼 충돌률이나 행동 후 드러난 면적은 유용한 독립 검증값이지만, 이를 Complexity GT로
선택하면 행동/도구에 조건화된 문제로 연구 정의가 바뀐다. Target 존재 확률과 같은 값도 아니다.

## 검증 순서

1. 물체 수·점유율을 가능한 한 맞춘 scene 안에서 간격과 겹침을 바꾼다. 넓은 평판/단독 곡면,
   간격이 충분한 여러 물체/서로 밀착한 물체, 텍스처만 바뀐 물체 등의 반례를 포함한다.
   가지런해도 접근이 막힐 수 있으므로 정렬 여부 자체를 정답으로 사용하지 않는다.
2. Calibration으로 공간 범위를 정하고, 물체 더미 내부의 low/high 구조 영역을 독립 기준으로
   고른 뒤 후보 map의 상대 순서·국소화와 시점 안정성을 비교한다. GT를 예측한 MAE만으로
   그 GT의 정의가 타당하다고 결론내리지 않는다.
3. 물체/빈 서랍 구분만으로 높은 점수를 얻지 않도록 foreground 내부 평가를 별도로 수행한다.
   Scene별 min–max나 상위 일정 비율을 강제로 밝히는 방식은 쓰지 않는다. 균일한 실제 구조에는
   균일한 값도 허용한다. 현재 30×40 feature grid가 필요한 공간 차이를 보존하는지도 확인한다.
4. 이 조건을 통과한 뒤 GT와 normalization을 고정하고 학습·새 scene 검증으로 진행한다.
   마지막으로 S+O와 S+O+C의 탐색 성공/행동 수를 비교해 추가 stream의 독립적인 효용을 평가한다.

현재 완료한 milestone은 **기존 GT 한계 확인과 최근 문헌에 기반한 재검토**다. 다음 Step은
더미 내부 구조의 후보 정의를 반례로 비교하는 것이며, fusion 학습과 새 Complexity 대규모 학습은
아직 이 검증을 통과하지 않았다.
