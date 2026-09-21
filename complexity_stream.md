# Complexity Stream

<!-- navigation:start -->
[전체 개요](README.md) · [Similarity](similarity_stream.md) · [Occlusion](occlusion_stream.md) · **Complexity** · [Development Log](development_log.md) · [연구 문맥](agent.md)
<!-- navigation:end -->

> **현재 상태 — Phase 38 및 입력 조건 정정:** 단일 RGB-D에서 접경을 예측하는 모델을 평가했으며 RGB 대비 일관된 개선 기준을 통과하지 못했음. GT 관계를 넣은 제거 회귀는 참고 진단으로 보존하고 실환경 추론 경로에서 제외함. 다음 구현은 RGB-D만으로 물체·관계 표현을 내부 생성하는 경로이며, 최종 구조·Complexity GT·fusion은 아직 확정하지 않음.

## 1. 연구 목적

Complexity는 **같은 더미 안에서도 어느 부분에 여러 물체가 밀집해 있는지, 그 물체들이 어떻게 놓여 있는지**를 RGB-D로 표현하려는 stream임. 찾는 target과의 유사도나 target별 가림 가능성은 다른 stream이 담당하며, 여기서는 현재 scene 자체를 다룸.

출발 가정은 **같은 넓이 안에 물체가 많으면 그 부분의 수량 밀도가 높다**는 것이었음. 이 가정을 사용하려면 먼저 RGB-D에서 무엇을 한 물체로 셀 것인지가 중요함. 책 표지의 글자·그림이 다르게 보여도 한 책의 일부로 다루고, 서로 붙은 두 책은 구분할 정보가 필요함.

Phase 36–37의 DINO 진단은 이 물체 구분 정보를 확인한 실험임. 개수 가설을 버리고 RGB edge 개수로 대체한 과정으로 설명하지 않음. Phase 37에서는 물체 대응 판별과 GT 기반 접경 계산을 분리하여, **물체를 구분할 정보가 있는가**와 **구분된 물체 사이에서 count 외에 무엇을 관측할 수 있는가**를 함께 점검함.

Phase 38은 그 정보를 원본 영상의 접경 위치에 연결하는 첫 학습 비교임. 동시에 GT 관계의 제거 효용을 따로 검사하여, **관계를 정확히 알 때 유용한가**와 **RGB-D에서 그 관계를 충분히 잘 추정하는가**를 구분함.

## 2. 연구 질문의 연결

①–⑧은 아래 본문의 설명 순서임. 근접도와 정적 제거를 검사한 Phase 34–35는 별도로 시도한 관계 점수 후보이므로 옆 경로로 표시함. **⑦의 공동 진단과 ⑧의 접경 학습·제거 utility 비교는 완료함.** ⑤의 feature 집계 범위 비교와 ⑥의 다중 뷰 활용은 여전히 미실행 제안임.

```mermaid
flowchart TD
    N1["① 무엇을 세고 싶은가<br/>같은 넓이 안의 서로 다른 물체 개수"]
    N2["② 무엇을 한 물체로 볼 것인가<br/>책 내부의 무늬와 다른 물체를 RGB-D로 구분"]
    N3["③ 기존 개수 예측 실험<br/>48·96·160px 범위에서 RGB-D와 depth-only 비교"]
    N4["④ DINO의 물체 구분 정보 확인<br/>두 patch가 같은 책의 부분인지 판별"]
    N5["⑤ 집계 범위 비교안 · 미실행<br/>어려운 같은·다른 물체 쌍을 고정<br/>주변 feature 범위와 추가 표현의 효과 비교"]
    N6["⑥ 다중 뷰 활용 검토 · 미실행<br/>다른 각도가 물체 구분 실패를 보완하는지 확인<br/>학습은 다중 뷰, 추론은 단일 RGB-D 방향"]
    N7["⑦ Phase 37 공동 진단 · 완료<br/>어려운 순수 patch 대응과 가시 접경을 별도 평가<br/>혼합 patch·경계·물체 묶음 보완 과제 확인"]
    N8["⑧ Phase 38 접경 학습·제거 utility · 완료<br/>RGB-D 결합 decoder의 일관된 개선은 미확인<br/>GT 관계의 추가 효용 확인 → 위치·물체 소속을 함께 보완"]
    H["별도 관계 후보: Phase 34–35<br/>GT 소속을 사용한 근접도·정적 제거 진단<br/>5절에 목적과 결과 보존"]
    N1 --> N2 --> N3
    N3 -->|"개수 오차 외에 물체 구분 정보를 직접 점검"| N4
    N4 -->|"고정한 DINO·판별기로 어려운 조건 재평가"| N7
    N7 -->|"전체 영상에서 접경을 직접 예측"| N8
    H -. "정적 제거 자료의 별도 utility 재분석" .-> N8
    N4 -. "문맥 범위의 이득을 비교할 후보" .-> N5
    N4 -. "추가 관측이 제공하는 정보 검토" .-> N6
    N3 -. "별도로 검토한 후보" .-> H
```

## 3. 가정·실험·현재 판단

### ① Count와 occupancy는 서로 다른 값을 측정함

**Count는 영역 안에 보이는 서로 다른 물체 그룹의 수임.** 같은 그룹이 여러 pixel에 걸쳐 있어도 한 번 셈. 예를 들어 같은 96×96 영역 안에 A·B·C 세 물체가 보이면 count는 3임.

| 같은 크기의 영역 안에서 | 보이는 그룹 | Count |
|---|---|---:|
| A·B·C가 서로 떨어져 있음 | A, B, C | 3 |
| A·B·C가 일부 겹치지만 각각의 일부가 보임 | A, B, C | 3 |

두 번째에서도 세 그룹이 보이므로 3임. B가 완전히 가려져 아무 부분도 보이지 않으면 현재의 가시 count에는 B가 들어가지 않음.

이 예에서 **단위 면적당 가시 개수가 같다는 판단은 맞음.** Count만으로 간격이나 겹침의 차이까지 표시하지는 않는다는 뜻임. 겹친 쪽의 Complexity가 반드시 더 높다고 증명한 예도, count 가설이 틀렸다는 근거도 아님. 배치 차이를 추가로 표현할지는 별도 검증 대상임.

**Occupancy는 물체가 덮은 면적의 비율임.** 아래 예는 workspace 안에 있으며 256pixel 모두의 물체·배경 여부를 아는 16×16 patch를 기준으로 함.

| Patch 안의 구성 | 물체가 덮은 pixel | Occupancy |
|---|---:|---:|
| 큰 책 한 권의 표면이 patch 전체를 덮음 | 256 | 256/256 = 1 |
| 두 물체가 빈틈없이 각각 절반을 덮음 | 128 + 128 | 256/256 = 1 |
| 물체가 절반만 덮고 나머지는 배경임 | 128 | 128/256 = 0.5 |

Occupancy 1은 **그 patch가 물체로 채워졌다**는 뜻임. 한 물체인지 여러 물체인지는 알려주지 않음. 따라서 occupancy는 물체 영역과 빈 공간을 구분하는 보조 정보로 사용하고, 물체 수를 구분하는 count와 같은 값으로 취급하지 않음.

실제 pilot은 각 위치 주변에서 저장된 segmentation 색 그룹을 세어 `count/16`으로 학습했음. 더미 전체 면적으로 나눈 값은 아님. 서로 다른 asset이 같은 색으로 저장된 경우도 있어 정확한 보고 명칭은 **가시 label-group 개수**임. 이 정답을 잘 예측하는지와, 국소 count가 최종 Complexity에 얼마나 유용한지는 구분함.

### ② 물체를 세려는 문제와 물체 경계를 구분하는 문제의 연결

세려는 대상이 책이라면 표지의 글자·그림·단색 부분을 각각 한 개로 세면 안 됨. 이 부분들은 다르게 보여도 **같은 책 A의 부분**임. 반대로 서로 붙어 있는 책 A와 책 B는 색이 비슷해도 다른 물체임.

```text
책 A의 그림 부분 ── 책 A의 글자 부분 → 같은 물체로 다룸
책 A의 표지     ── 책 B의 표지     → 다른 물체로 구분함
```

RGB edge는 색이나 밝기가 변하는 곳임. 글자 테두리에도, 책 A와 책 B 사이에도 생길 수 있음. 따라서 여기서 경계를 검토하는 이유는 **edge 개수를 새 복잡도 값으로 쓰기 위해서가 아니라, 변화의 양쪽이 같은 물체인지 다른 물체인지 확인하기 위해서**임. 이는 처음의 “무엇을 한 개로 셀 것인가”와 연결된 문제임.

Depth는 다른 단서를 제공함. 무늬가 복잡한 책 표지라도 평평한 한 면이면 depth는 비슷할 수 있음. 그러나 같은 높이에 놓인 두 책도 depth가 비슷할 수 있어, depth만으로 항상 둘을 구분할 수 있는 것은 아님.

| 가상 장면 | RGB가 주는 단서 | Depth가 주는 단서 |
|---|---|---|
| 무늬가 복잡한 책 한 권 | 글자·그림 때문에 변화가 많음 | 같은 표면의 깊이가 비슷할 수 있음 |
| 높이가 같은 두 책 | 외형·윤곽·문맥에서 구분 단서를 얻을 수 있음 | 두 물체의 깊이가 비슷할 수 있음 |
| 기울어진 책 한 권 | 하나의 책으로 이어지는 외형·문맥 | 한 물체 안에서도 깊이가 크게 변함 |

그래서 RGB와 depth를 함께 검토함. Depth cue에서는 빈 서랍 자체의 벽·바닥에도 반응하는 오류를 확인해 empty-depth 차이로 보정했음. 위의 책 예들은 구분해야 할 상황을 설명하는 가상 예이며, 특정 segmentation 모델의 실패를 실측한 결과와 구분함.

현재 density 모델이 매번 완성된 segmentation을 만든 뒤 개수를 세는 구조는 아님. **Segmentation은 학습 정답 생성에 사용하고, 모델은 RGB-D에서 개수를 직접 예측함.** 다만 개수를 잘 예측했다는 결과만으로 한 책의 서로 다른 부분까지 올바르게 연결하는지 알 수는 없으므로, ④에서 그 정보를 직접 진단함.

### ③ 범위별 count 실험이 답한 질문

Phase 33에서 비교한 질문은 **“같은 범위의 count를 예측할 때 RGB 정보를 추가하면 depth-only보다 나아지는가”**임. DINOv3는 고정하고 작은 RGB-D head를 학습했으며, 비교 모델의 구조·초기값·학습 순서를 맞춤.

| 설정 | 의미 |
|---|---|
| DINO patch 16×16px | Feature와 출력을 두는 간격. 480×640 영상에서 30×40개 위치를 만듦 |
| Count window 48·96·160px | 같은 출력 위치 주변에서 개수를 세는 정사각형의 한 변 길이 |
| Count ÷16 | 기존 asset library를 기준으로 정한 출력 scale. Patch의 16px와 별개임 |

한 위치에서 세 값을 동시에 예측함. 예를 들어 위치 A의 작은 영역에 한 그룹, 중간 영역에 세 그룹, 큰 영역에 다섯 그룹이 보인다면 정답은 다음처럼 달라짐.

```text
                         정답 개수       모델의 예측값, 설명용 가상값
A 주변 48×48 영역          1개             1.2개 → 오차 0.2
A 주변 96×96 영역          3개             3.5개 → 오차 0.5
A 주변 160×160 영역        5개             5.8개 → 오차 0.8

모델은 같은 위치에서 count48, count96, count160 세 출력을 냄.
표시는 개수 단위로 풀었으며 실제 학습 출력은 각 값을 16으로 나눈 값임.
```

48px의 오차가 작다는 것은 **그 작은 영역의 한 그룹을 더 정확히 셌다**는 뜻임. 160px에서 보이는 다섯 그룹까지 더 잘 구분했거나 넓은 관계 정보를 더 잘 담았다는 뜻은 아님. 서로 다른 질문의 오차를 한 줄로 줄 세운 것만으로 어느 범위를 쓸지 결정할 수 없는 이유임.

실제 결과는 다음과 같음. 기존 16개 source pool·고정 five-camera rig, test 960영상·3 seeds에서 유효하며 GT count가 0보다 큰 window의 **가시 label-group 개수 MAE**를 계산함. MAE는 예측과 정답의 차이에 절댓값을 취해 평균한 값이며, 작을수록 좋음.

| Count window | RGB-D MAE | Depth-only MAE |
|---|---:|---:|
| 48×48px | 0.352517 | 0.497709 |
| 96×96px | 0.621799 | 0.805782 |
| 160×160px | 0.949932 | 1.194659 |
| 세 범위 평균 | **0.641416** | **0.832717** |

이 표에서는 **같은 행의 RGB-D와 depth-only를 비교함.** 세 범위 모두 RGB-D가 더 정확했고, 평균 count MAE가 22.973% 감소함. 이 결과로 RGB 표현의 추가 정보를 확인했음.

범위가 커지면 workspace 밖을 포함하는 위치도 늘어 기존 평가의 유효 위치가 달라짐. 따라서 지금은 세 출력을 보존하고 **48px 하나를 최적값으로 선택하지 않음.** 다음 범위 비교에서는 정답과 평가 위치를 고정하고 주변 feature를 모으는 범위만 바꾸는 안을 ⑤에 구체화함.

![RGB-D density pilot in five views](img/complexity/book_1_five_views.png)

열은 scene RGB / 96px count GT / RGB-D prediction / 절대 오차 / occupancy GT / 직접 depth occupancy / depth plane residual임. 현재 그림은 density pilot의 결과임.

### ④ ‘물체 소속’을 DINO에서 읽는다는 의미

여기서 **소속은 영상의 한 부분이 어느 물체의 일부인가**라는 뜻임. Category를 묻는 것과 구분함. 책 A와 책 B는 둘 다 book이지만, 책 A의 그림 부분과 책 B의 글자 부분은 서로 다른 물체에 속함.

```text
사진 속 세 위치
p: 책 A의 그림 부분
q: 책 A의 글자 부분
r: 책 B의 표지

(p,q) → 같은 물체의 부분인가?  정답 1
(p,r) → 같은 물체의 부분인가?  정답 0
```

Phase 36은 이 질문을 DINO feature로 답할 수 있는지 검사한 실험임. DINO는 각 16×16 patch를 768개 숫자로 표현함. 두 patch의 숫자 묶음을 작은 판별기에 주고 **같은 asset인 쌍에 높은 점수, 다른 asset인 쌍에 낮은 점수**를 내도록 학습함.

```text
Scene RGB → 고정 DINO → p의 feature, q의 feature ─┐
                                                ├→ 작은 판별기 → 같은 물체인지의 점수
                                  위치 정보 ────┘

저장된 GT segmentation → p·q가 같은 asset인지 정답 지정 → 학습·평가에 사용
```

DINO+position 비교군을 나타낸 그림임. Depth 비교군과 RGB-D 비교군도 같은 크기의 판별기를 사용함. GT가 평가 위치와 정답을 제공했으며 모델 입력에 물체 이름이나 GT label 번호를 넣지는 않았음. Density pilot의 개수 예측 head와도 별도의 진단임.

먼저 한 patch 면적의 90% 이상이 같은 알려진 label인 위치를 골랐음. 여러 물체가 섞인 patch보다, **이미 한 물체의 내부라고 확인한 두 patch를 연결하는 기본 능력**부터 본 것임. 기존 16개 asset이 train/test에 모두 들어간 조건이며, 같은 asset의 복제품 여러 개를 구분한 실험은 아님.

**평가 지표인 AUROC는 같은 물체 쌍을 다른 물체 쌍보다 높은 점수로 정렬하는지를 측정함.** 정식 명칭은 Area Under the Receiver Operating Characteristic Curve이며, 현재 실험에서는 아래 순위 비교로 이해할 수 있음.

| 설명용 가상 쌍 | 정답 | 판별 점수 |
|---|---|---:|
| 같은 책 A의 두 부분 | 같음 | 0.4 |
| 같은 책 B의 두 부분 | 같음 | 0.3 |
| 책 A와 책 B의 부분 | 다름 | 0.2 |
| 책 C와 책 D의 부분 | 다름 | 0.1 |

같은 물체 쌍 하나와 다른 물체 쌍 하나를 골라 비교하면 `0.4>0.2`, `0.4>0.1`, `0.3>0.2`, `0.3>0.1` 네 비교가 모두 맞음. 이 예의 AUROC는 **4/4=1**임. 순위가 무작위에 가까우면 약 0.5이고, 동점은 절반 기여로 계산함.

한편 위 점수를 0.5 이상이면 ‘같음’이라고 판정하면 네 쌍을 전부 ‘다름’으로 분류하여 정확도는 50%가 됨. **AUROC는 특정 기준값에서의 정답률과 다른 지표**임. 높은 AUROC는 두 종류를 구분할 점수 순서가 잘 형성됐다는 뜻임.

실제 same-category 평가 결과는 다음과 같음. 책끼리처럼 같은 category 안에서도 다른 asset을 구분하는 조건임.

| 입력 | AUROC |
|---|---:|
| DINO + position | **0.998953** |
| Depth + position | 0.773882 |
| RGB-D + position | 0.998908 |

Same-category 80,024 pairs·604 views·8 test scene keys를 사용하고, view별 지표를 scene key와 3 seeds에 걸쳐 평균했음. **선택한 내부 patch의 같은/다른 asset을 구분할 정보는 기존 DINO에서 매우 잘 읽혔음.** 0.998953을 전체 영상 segmentation 정확도 99.8953%로 해석하지 않음.

![Frozen-feature visible-asset correspondence](img/complexity/representation_comparison.png)

순수성 ≥90%와 workspace·유효 depth 각각 ≥95% 조건을 함께 만족한 위치는 알려진 foreground patch의 **42.40%**였음. 당시 다음 질문은 **외형 차이·실제 물체 경계·분리된 부분에서도 같은 구분이 유지되는가**였으며, 이후 ⑦에서 기존 판별기를 고정해 조건을 나누어 검사함. DINO의 출력으로 물체 mask 전체를 복원하거나 segmentation 모델과 성능을 비교한 단계와 구분함.

### ⑤ 다음 범위 비교의 과제와 선택 기준

‘공통 기준을 정해야 함’에서 설명을 끝내지 않고, **물체 구분을 점검하는 다음 비교안**을 아래처럼 둠. 이 안은 아직 실행하지 않았으며 최종 Complexity GT를 새로 만든 것도 아님.

먼저 실제 더미의 경계·분리된 조각에서 정답의 모호함과 모델의 오류를 구분함. 주변 문맥을 모으는 범위가 부족한 것이 실패 원인으로 의심될 때 아래 비교를 진행하는 안임. 다른 각도의 관측이 필요한 실패인지 확인하는 별도 후보는 ⑥에 정리함.

비교 질문은 **“같은 두 위치가 같은 물체의 부분인지 판단할 때, 주변 정보를 어느 범위까지 함께 읽는 것이 도움이 되는가”**임. 관측 범위를 바꾸더라도 두 위치와 정답은 그대로 둠.

| 비교 항목 | 고정할 내용 |
|---|---|
| Scene | 기존 책·과일·포장식품·장난감이 쌓인 cluttered asset scene. 비교군 모두 동일한 scene-key 분할 사용 |
| 평가 대상 | 아래 세 유형의 같은/다른 물체 쌍을 한 번 정하고 모든 비교군에 동일하게 사용 |
| 정답·유효 위치 | GT의 같은/다른 asset 정답을 고정하고, 모든 범위에 공통으로 유효한 위치 사용 |
| 바꾸는 요소 | 고정 DINO feature에서 주변 정보를 모으는 범위만 48·96·160px로 변경 |
| 비교 방법 | 같은 feature 요약 방식·출력 차원·판별기 구조·학습 조건 사용 |
| 선택 기준 | 세 유형의 validation AUROC를 동일 가중 평균하여 가장 높은 범위를 선택 |
| 최종 확인 | Validation에서 선택과 판정 기준값을 고정한 뒤 test 쌍에서 AUROC와 두 종류 오류를 보고 |

새 비교의 분할과 평가 쌍은 test를 열기 전에 고정함. 판정 기준값은 validation에서 두 종류 오류율의 평균이 가장 작아지는 값으로 정함.

세 유형은 다음과 같이 구성할 계획임. 각 유형에 같은 물체 쌍과 다른 물체 쌍을 함께 두어 판별 성능을 비교함.

| 유형 | 같은 물체 쌍의 예 | 다른 물체 쌍의 예 |
|---|---|---|
| 외형 변화 | 한 책의 그림 부분과 글자 부분 | 서로 다른 두 책의 표지 부분 |
| 경계 근처 | 표지의 무늬 경계 양쪽 | 맞닿은 책 A·B의 경계 양쪽 |
| 분리된 가시 부분 | 가려져 떨어져 보이는 같은 책의 두 조각 | 서로 다른 두 책의 가시 조각 |

오류도 **다른 물체를 같다고 합치는 오류**와 **같은 물체를 다르다고 나누는 오류**로 나눠 보여줌. Depth·위치 단서와 범위별 평가 쌍의 구성이 결과를 대신 설명하지 않도록 맞추며, 기존의 순수 내부 patch만으로 평가를 제한하지 않음.

여기서 48·96·160px는 판별기에 추가로 모아주는 **주변 feature의 범위**임. ③처럼 그 영역 안의 정답 개수를 바꾸는 실험과 다름. DINO의 입력 영상과 16px patch는 고정하므로, DINO 자체가 그 작은 영역만 본다는 뜻도 아님.

이 비교로 고르는 것은 **물체 구분 과제에 도움이 되는 feature 집계 범위**임. 최종 Complexity의 모든 상황에 최적인 범위라고 확대하지 않음. 당장은 기존 세 count 출력을 유지하며, 이 구분에서 실제 실패가 확인되면 물체 묶음 모델이나 공간 표현이 그 오류를 줄이는지 같은 평가에서 비교함. 추가 모델 B/C와 최종 fusion은 아직 실행하지 않았음. **Phase 37은 어려운 위치쌍을 평가했지만 주변 feature 집계 범위는 바꾸지 않았으므로, 이 비교안을 실행한 결과가 아님.**

### ⑥ 다중 뷰의 일관성을 이용한 물체 구분 — 2026-09-18 검토안

**현재 상태:** 교수님이 제안한 다중 뷰 합의의 의미와 적용 조건을 검토한 단계임. Cross-view 대응·mask 통합·teacher 학습은 아직 구현하거나 평가하지 않았음. 현재 다섯 camera는 각각 단일-view sample로 처리하며, 다섯 영상을 한 번에 받는 Complexity 모델은 아님.

#### 제안의 목적과 ‘교집합’의 의미

출발 질문은 **“한 방향에서 불안정한 물체 구분을 다른 방향의 관측으로 보완할 수 있는가”**임. 같은 camera 방향에서도 장면·가림 상태에 따라 segmentation이 잘되거나 어려워질 수 있다는 문제의식임.

가상 예로, 정면에서는 책 표지의 글자·그림 때문에 한 권이 세 영역으로 나뉘고 옆에서는 책의 윤곽이 더 분명할 수 있음. 두 관측을 연결하면 정면의 조각들이 같은 책인지 판단할 단서가 늘어남. 이는 기존의 **“무엇을 하나의 물체로 묶을 것인가”**와 연결되며, segmentation 성공률이나 복잡도 개선을 새로 측정한 결과는 아님.

여기서 여러 영상의 mask를 그대로 겹쳐 교집합을 구할 수는 없음. 같은 물체도 camera마다 영상 위치·크기·보이는 면이 다름. Depth와 camera 보정값으로 영상의 관측 표면을 공통 3D 좌표에 대응시킨 뒤, 같은 표면에서 물체 구분이 일치하는지 확인해야 함.

| 구분 | 필요한 처리와 이유 |
|---|---|
| 영상 좌표의 차이 | Depth·camera 내부/외부 보정값으로 대응시킴. 각 영상의 같은 pixel 좌표가 같은 공간이라는 뜻은 아님 |
| 가림·화면 밖 영역 | 해당 view에서 관측되지 않는 표면은 합의 평가에서 제외함. 안 보인다는 사실을 물체가 없다는 반대 증거로 쓰지 않음 |
| 부분적으로만 보이는 물체 | 다섯 view 모두의 교집합을 요구하면 작은 물체·심하게 가려진 물체가 탈락할 수 있음. 실제로 함께 보이는 영역과 관측 coverage를 함께 봄 |
| 서로 다른 표면 | 정면과 뒷면은 같은 물체라도 관측 점이 겹치지 않을 수 있음. 공통 표면의 대응 확인과 물체 전체를 하나로 묶는 문제를 구분함 |
| 반복되는 같은 오류 | 같은 segmentation 모델이 여러 view에서 똑같이 잘못 나눌 수 있음. 합의 정도는 신뢰도 단서이며 정답 보장은 아님 |

따라서 검토할 대상은 **가시성이 확인된 대응 영역의 일관성**임. Depth·보정 오차, mask 품질도 함께 확인하고, 합의가 높은 영역만 남겼을 때 어려운 영역을 얼마나 버리는지도 보고해야 함. 합의 점수가 높다는 이유로 곧바로 Complexity가 높다고 정의하지 않음.

#### 학습의 다중 뷰와 추론 입력의 분리

사용자는 실제 사용 시 여러 camera가 필수인 구조를 우려함. 따라서 **단일 RGB-D 추론을 유지하면서 기존 다중 뷰 자료를 학습·검증에 활용하는 방향**을 우선 검토함. 아직 채택한 모델 구조는 아님.

| 사용 방식 | 실제 추론에 필요한 관측 | 적용 판단 |
|---|---|---|
| 여러 camera의 결과를 매번 통합 | 여러 camera의 RGB-D·보정값 | 추가 관측을 직접 활용하지만 설치·운용 조건이 늘어남 |
| 한 camera를 이동해 여러 방향 관측 | 여러 번 촬영·camera 위치 추정 | Camera 수는 하나여도 시간·이동 비용이 있으며 관측 사이의 물체 이동을 처리해야 함 |
| **학습 때 다중 뷰, 추론은 단일 뷰** | **한 시점의 RGB-D** | **현재 목표에 맞는 검토 방향. 추가 감독이 단일-view 성능에 전달되는지 비교 필요** |

학습 때만 추가 정보를 사용하는 구조는 다음과 같이 생각할 수 있음. **Teacher는 학습 목표를 제공하는 경로**, **student는 실제 사용 시 한 뷰를 입력받는 모델**을 뜻함.

```text
학습 — 미실행 개념안
  같은 더미의 여러 RGB-D view + 보정·검증된 물체 대응
      → 물체 단위 표현 / 대응 정보 집계 → 학습 목표(teacher)
                                              ↑ 예측과 비교하여 학습
  그중 한 view의 RGB-D → 단일-view 모델(student) → 예측 표현

추론 — 목표 입력 조건
  새로운 scene의 한 view RGB-D → student → 물체 구분에 사용할 표현
```

다른 방향에서만 보이는 정보를 단일 view에서 항상 복원할 수 있다는 뜻은 아님. 입력에서 구분할 단서가 없는 경우는 여전히 불확실할 수 있음. 또한 단일 camera 사용이 현재 pilot의 workspace·empty-depth reference 조건이나 camera 변화 문제까지 자동으로 해결하는 것은 아님.

관련 최근 연구는 다음 두 방향의 근거가 됨.

- **SAM2Object — CVPR 2025:** View별 mask 불일치와 저품질 mask를 동등하게 결합하는 문제를 다루며, SAM2의 segmentation·tracking, mask 품질 선별, 3D 기하를 함께 사용함. 다중 뷰 일관성이 유용한 단서라는 근거이며, 우리 다섯 고정 view에서 같은 효과를 확인한 것은 아님. [논문](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_SAM2Object_Consolidating_View_Consistency_via_SAM2_for_Zero-Shot_3D_Instance_CVPR_2025_paper.html)
- **3D Feature Distillation with Object-Centric Priors — 2024, 2025 개정:** 다중 뷰의 물체 단위 CLIP 표현을 학습 목표로 사용하고, 추론에서는 단일 RGB-D의 부분 point cloud로 표현을 계산하는 DROP-CLIP을 제시함. 다중 뷰 학습과 단일 뷰 추론을 분리할 수 있다는 사례임. 본 연구의 Complexity 정의·GT를 검증한 논문은 아님. [논문](https://arxiv.org/abs/2406.18742)

#### 현재 연구에서 확인할 순서

1. **단일-view 실패를 특정함.** 실제 책·과일·포장식품·장난감 더미에서 같은 물체를 여러 개로 나누는 오류와 다른 물체를 하나로 합치는 오류를 확인함. Phase 36의 순수 내부 patch 성공을 반복 평가하는 데 그치지 않음.
2. **추가 view가 해당 실패를 보완하는지 검사함.** 기준 view의 동일 영역·물체·정답을 고정하여 단일-view 결과와 다중-view 후보를 비교함. 시뮬레이션의 검증된 object ID·pose·기하를 준거로 사용하고, 색 충돌이 있는 기존 segmentation 색을 곧바로 물체 ID로 간주하지 않음. 기존 검증된 GT를 다중 뷰 합의 mask로 대체하지 않음.
3. **이득이 확인된 정보를 단일-view 모델에 전달하는지 비교함.** 같은 student 구조·단일 RGB-D 입력·기존 감독·학습 조건에서, 다중 뷰 감독의 추가 여부를 비교함. 다른 물체의 병합 오류, 같은 물체의 분할 오류, 평가 coverage를 함께 보고함. Multi-view teacher의 성능과 최종 single-view student의 성능을 따로 기록함.

다중 뷰 대응은 **같은 source pool·같은 실제 scene/layout**에서 구성하고, 여러 camera의 관측은 scene-key 단위로 같은 train/validation/test split에 둠. 다른 pool에서 scene key 문자열만 같다는 이유로 같은 더미로 연결하지 않음. Phase 35의 추가 asset capture와 Phase 36의 원본 데이터도 구분함.

이 검토의 첫 목표는 **물체 구분 표현을 안정화할 수 있는가**임. 그 표현을 이용해 어느 국소 물체 관계를 Complexity로 나타낼지는 후속 문제이며, 다중 뷰 일관성 자체를 새 Complexity GT나 최종 feature로 채택하지 않았음. 기존 density pilot·Phase 33–36 결과는 그대로 보존함.

### ⑦ 어려운 물체 대응과 가시 관계를 함께 확인함 — Phase 37

**기존 DINO는 경계 주변·외형 변화가 있는 순수 patch에서도 같은 물체를 구분할 정보를 유지했음. 동시에, 물체 ID를 정확히 알아도 16×16 영역마다 label 하나만 남기면 접경이 달라짐을 확인함.** 따라서 다음 구현에서는 encoder를 바로 바꾸기보다, 여러 물체가 섞인 영역과 경계 정보를 어떻게 유지하고 물체별로 묶을지 먼저 다룸.

Phase 36과 같은 validation 4 keys·320영상, test 8 keys·640영상을 사용함. 기존 16개 asset·5 camera 조건이며 이미 확인한 test 장면의 새 진단임. 새로운 blind test나 unseen-object 평가로 세지 않음. 같은 key의 pool·view를 묶어 집계함. 저장된 segmentation 색이 여러 asset에 대응하면 unknown으로 제외하며, 과거 density pilot의 GT·수치는 소급 변경하지 않음.

이번 실험은 서로 다른 세 계산을 분리함.

| 검사 | 입력과 계산 | 답한 질문 |
|---|---|---|
| 어려운 위치쌍 판별 | 기존 frozen DINO와 Phase 36 판별기·정규화를 그대로 사용함. GT는 위치·정답 선택에만 사용함 | 어려운 조건에서도 같은/다른 물체의 점수 순서를 읽을 수 있는가 |
| 가시 관계 계산 | 알려진 pixel GT 물체 ID와 depth에서 서로 다른 물체의 접경을 셈 | 국소 count가 같아도 관측 관계가 다른가 |
| GT block 근사 | Pixel GT를 16×16마다 다수 label 하나로 바꾸고 관계를 다시 계산함 | 한 영역에 물체 하나만 남기는 변환으로 어떤 정보가 바뀌는가 |

#### 어려운 위치쌍에서 확인한 기존 DINO의 대응 정보

두 위치 모두 한 물체의 면적이 patch의 90% 이상이며 workspace·유효 depth가 각각 95% 이상인 조건을 유지함. 경계 주변이라도 **혼합 patch를 억지로 한 물체의 정답으로 지정하지 않음.** 위치쌍 거리는 기존 학습 범위인 1–8 patches로 제한함. 같은/다른 물체 쌍의 signed XY offset, anchor category, depth 차이 구간과 음성 category 조건을 맞추고, 영상·유형마다 최대 64개의 양성·음성 묶음을 선택함.

| 유형 | 위치쌍 선정 기준 |
|---|---|
| 내부 대조 | 두 중심 모두 알려진 다른 물체와의 접경에서 16px보다 멂 |
| 경계 주변 | 한쪽 이상 중심이 접경에서 16px 이내임. 두 patch의 순수성 조건은 유지함 |
| 외형 변화 | 평균 RGB 차이의 L2 크기가 0.15 이상임(RGB 0–1). 책의 글자·무늬 의미를 판정한 기준은 아님 |
| 분리된 가시 조각 | 같은 asset의 상하좌우·대각선으로 연결한 pixel 묶음(8-connected)이 서로 다른 경우를 양성으로 삼음. 분리 원인을 가림으로 확정하지 않음 |

아래는 **같은 category 안에서의 구분 결과**임. DINO·depth·RGB-D 비교군은 모두 position을 포함하며, 기존 3-seed 판별기를 추가 학습 없이 사용함. View별 지표 → key 안의 pool/view 평균 → keys 동일 평균 → seeds 평균 순서로 집계함.

| 유형 | Pairs / views / keys | DINO AUROC | Depth AUROC | RGB-D AUROC | DINO 같은→다름 | DINO 다름→같음 |
|---|---:|---:|---:|---:|---:|---:|
| 내부 대조 | 17,108 / 438 / 8 | 0.999963 | 0.836209 | 0.999958 | 0.056% | 1.375% |
| 경계 주변 | 37,856 / 602 / 8 | **0.998789** | 0.747029 | 0.998715 | 0.539% | 3.161% |
| 외형 변화 | 30,870 / 584 / 8 | **0.998166** | 0.766375 | 0.998197 | 1.160% | 2.685% |
| 분리된 가시 조각 | 852 / 47 / 7 | **1.000000** | 0.751869 | 1.000000 | **4.274%** | 0.786% |

같은→다름은 실제 같은 물체 쌍을 다른 물체로 판정한 비율, 다름→같음은 그 반대임. **위치쌍의 판정 오류율이며 segmentation 결과의 물체 분할·병합 비율은 아님.** 모델별 판정 기준값은 validation에서 두 오류율 평균을 최소화하도록 정한 뒤 test 전에 고정함.

분리 조각은 AUROC 1이지만 같은→다름 오류 4.274%가 남았음. 같은 쌍의 점수가 다른 쌍보다 높아도, 모든 유형에 적용하는 기준값보다 낮을 수 있기 때문임. 표본도 47 views·7 keys이므로 전체 물체 연결을 완벽히 해결한 결과로 해석하지 않음. 서로 다른 유형에 같은 pair가 포함될 수 있어 표의 표본을 독립 관측으로 합산하지 않음.

RGB-D가 DINO보다 일관되게 높은 AUROC를 보이지는 않았음. 이 결과는 현재 대응 과제에서 기존 DINO를 계속 활용할 근거이며, depth가 물체 간 깊이·공간 관계를 표현하는 역할까지 부정하는 것은 아님.

![Frozen DINO and RGB-D anchor affinity](img/complexity/joint_anchor_affinity_20260918.png)

열은 RGB와 기준 patch / 실제 기준 물체의 patch별 면적 비율 / DINO score / RGB-D score임. 각 category의 첫 pool·첫 test key·center view를 고정하고, GT로 적격 patch가 가장 많은 물체의 내부 기준점을 선택함. 반경 8 patches 안에서 3-seed 평균 logit의 sigmoid를 표시함. **Score는 보정된 소속 확률이나 면적 비율이 아님.** 그림에는 정량 평가에서 제외한 mixed·background 위치도 포함되어 있으며, 자동 기준점 선택·전체 grouping 성능을 나타내지 않음.

#### 혼합 patch를 버리거나 한 label로 줄일 때 남는 문제

알려진 foreground를 포함한 128,380 patches 중 기존 평가 조건을 만족한 것은 54,429개(**42.40%**)였음. 서로 다른 두 물체가 각각 10% 이상 들어 있는 patch는 20,469개(**15.94%**)였음. 부적격 영역 전체가 두 물체 혼합인 것은 아니며 물체–배경 혼합·unknown·workspace/depth 조건도 포함됨.

예를 들어 한 patch에 책 A가 60%, 책 B가 40% 보이면, 그 patch를 A 하나로 바꾸는 순간 B의 부분은 없어짐. 가느다란 B의 조각이나 A–B 사이의 작은 틈이 사라지면 이웃 관계도 바뀔 수 있음. 이를 확인하기 위해 **모델 예측 대신 정확한 pixel GT 자체를 block 다수 label로 변환**함. Background·unknown도 다수결 후보에 포함함.

이는 **향후 grouping을 patch당 단일 label로 구성할 때의 정보 손실**을 확인한 검사임. 현재 density pilot은 RGB-D에서 count를 직접 예측하며, 중간에 이 단일-label 변환을 수행하지 않음. 따라서 이 결과로 기존 density 모델의 실패 원인을 찾아냈다고 해석하지 않음.

| Pixel GT와 비교한 16px GT block 근사 | 결과 |
|---|---:|
| 접경 pair precision | **0.804652** |
| 접경 pair recall | **0.845858** |
| 물체별 접경 이웃 수 MAE | 0.584765개 |
| 알려진 foreground pixel의 label 일치율 | 0.846709 |
| Block 근사에서 사라진 object-view | 606 / 8,446 = 7.175% |

Precision은 block 근사에서 생긴 접경 중 원래 GT에도 있던 비율, recall은 원래 접경 중 block 근사에도 남은 비율임. 앞 네 값은 view→key 동일 평균이며 누락 object-view는 전체 개수 비율임. **이는 DINO의 segmentation 성능도, DINO로 달성할 수 있는 성능의 상한도 아님.** 하나의 768-D feature에 여러 물체 정보를 유지하거나 더 세밀한 경계를 복원하는 방법은 별도로 평가할 수 있음.

#### 같은 count 안에서도 달라지는 가시 관계

관계는 수평·수직으로 이웃한 두 pixel이 서로 다른 알려진 물체 ID이며, 양쪽 depth가 유효하고 workspace 안에 있을 때 기록함. 책 표지의 글자처럼 **같은 물체 내부의 색 변화는 포함하지 않음.** 한 물체 쌍의 scene 전체 접경 support가 8 이상인 조건을 기본으로 사용함. Support는 A의 pixel 바로 옆에 B의 pixel이 있는 **이웃 pixel 쌍의 수**임. 그런 쌍이 10개면 support는 10이며, 물리적으로 접촉했다는 뜻은 아님.

각 window에서는 scene 전체에 32px 이상, 해당 window에 16px 이상 보이는 물체를 count에 포함함. 관계 수도 **이 count에 들어간 물체 집합 안에서만** 계산하고, 접경 양쪽 pixel이 window 안에 함께 들어온 물체 쌍을 한 번 셈. 따라서 count에서 제외한 작은 조각을 관계에만 추가해 차이를 만드는 비교가 아님. 예를 들어 A–B, B–C 접경과 A–B, B–C, A–C 접경은 모두 세 물체이지만 관측 관계 수는 2와 3임.

96px에서 같은 영상·같은 count≥2를 갖는 window가 5개 이상인 **3,225개 그룹 중 3,037개(94.17%)에서 관계 수가 달랐음.** 예를 들어 한 영상에서 count가 3인 위치가 5개 이상이면 그 위치들을 한 그룹으로 묶고, 그 안의 관계 수가 같은지 확인한 것임. Count와 접경 관계가 서로 다른 관측 정보를 담는다는 결과임. 서로 다른 window의 물체 종류·형상·면적까지 고정한 실험은 아니며, 관계 수가 클수록 반드시 더 복잡하거나 탐색에 유리하다는 결론은 아님.

| Window | 같은 count 그룹 / 관계 수가 달라진 그룹 | GT block count MAE | GT block 관계 수 MAE |
|---|---:|---:|---:|
| 48px | 2,151 / 2,110 | 0.452506 | 0.523765 |
| 96px | 3,225 / 3,037 | 0.279495 | 0.728926 |
| 160px | 2,744 / 2,552 | 0.464726 | 1.265518 |

오차는 full-pixel GT와 block 근사 사이의 차이임. Window별 정답과 공통 유효 위치가 달라 이 표로 최적 window를 고르지 않음. Count 평가에는 workspace≥95%와 workspace 내부 unknown 부재를 요구하며, 관계 평가에는 추가로 해당 window의 알려진 물체 depth가 모두 유효해야 함. Invalid 영역과 관계 수 0을 구분함.

![Observed scene relations and GT block approximation](img/complexity/joint_scene_relations_20260918.png)

열은 scene RGB / pixel GT 물체 ID / 16px GT block 근사 / 96px 가시 count / count와 같은 물체 집합의 접경 pair 수임. 위 기준점 그림과 같은 사전 고정 장면이며 빈 map 영역은 평가 조건을 만족하지 않는 위치임. **관계 map은 GT 계산 결과이며 RGB-D 추론 모델의 예측이 아님.**

접경 support를 4/8/16으로 바꾸면 전체 pair 수는 9,451/8,852/7,656개였음. 관측 접경에서 depth 차이가 ±5/10/20mm 밖인 비율도 따로 계산했으며 count·관계 수와 합산하지 않았음. 이 관계는 영상의 가시 인접과 깊이 순서이며, 물리적 접촉·지지·숨은 겹침을 확정하지 않음. 면적과 접경 이웃 수의 평균 Spearman도 0.405692여서 크기와 무관한 보편적 Complexity로 채택할 단계는 아님.

#### 이번 결과로 정한 다음 구현 범위

Frozen DINO를 유지하면서 한 patch에 여러 물체 정보가 남을 수 있는 **soft grouping**과 RGB-D 경계 보완을 우선 비교할 후보로 둠. 여기서 soft grouping은 patch 하나를 곧바로 물체 하나에 고정하지 않고, 여러 영역에 대한 소속 정도를 유지하는 표현을 뜻함. 아직 구현·학습하거나 효과를 확인한 방법은 아님. 물체 묶음을 명시적으로 만들지 않고 RGB-D에서 관계를 직접 예측하는 경로도 가능하므로 최종 architecture는 선택하지 않았음. 보완 후에는 순수 patch 대응뿐 아니라 혼합 영역 coverage·관계 보존을 함께 보고, 물체 묶음을 출력한다면 그 품질도 평가해야 함.

⑤의 집계 범위 비교, ⑥의 다중 뷰 teacher, 추가 모델 B/C는 이번에 실행하지 않았음. 단일 RGB-D 추론을 유지하며 어떤 보완이 실제 실패를 줄이는지 확인한 뒤 채택 여부를 판단함. Count·가시 관계는 분리된 감독 후보로 남기고 최종 Complexity scalar·fusion·DRL 효용은 이후 검증함.

실행 조건과 추가 수치는 `development_log.md` Phase 37에 정리함. 로컬 상세 근거는 `docs/complexity_results/joint_diagnostic_20260918.md`와 `outputs/complexity_joint_diagnostic_20260918_v1/`의 `summary.json`, `audit.json`임. 코드·checkpoint·dense cache·원시 NPZ는 로컬에 보존함. 독립 감사는 13개 score의 AUROC·validation threshold·표본 균형·offset·scene-key/seed 집계를 통과했으며, 원시 GT 관계 계산 전체를 재실행한 감사와 구분함.

### ⑧ RGB-D 접경 예측과 관측 관계의 효용 — Phase 38

이번에는 두 질문을 분리함. **관계를 정확히 알면 물체를 제거했을 때의 효과를 더 잘 설명하는가**, 그리고 **그 관계의 출발점인 접경을 RGB-D만으로 찾을 수 있는가**임. 첫 질문은 GT 물체 ID와 depth로, 두 번째는 전체 RGB-D 영상을 입력받는 학습 모델로 검사함.

#### 외부 입력과 내부 관계 표현

**실제 추론에서 제공하는 것은 RGB와 depth뿐임.** 관계는 사람이 정답 숫자로 넣는 추가 입력이 아니라, 이 관측을 처리한 모델이 내부에서 생성해야 하는 표현임.

```text
단일 RGB + depth
    ↓
물체·영역과 관계를 관측에서 추정하는 모듈 — 후속 구현 대상
    ↓
예측한 물체 소속·접경·관계 feature
    ↓
Complexity 표현과 평가 — 최종 구조·정의는 검증 중

GT segmentation·관계 정답 → 학습 loss / 출력 평가에만 사용
```

외부 pretrained 모듈을 보완하더라도 입력은 같은 관측이어야 함. GT로 후보 물체·기준점·crop을 고르거나, GT 관계로 예측 오류를 고친 뒤 실환경 성능이라고 보고하지 않음. 예측 mask를 사용하는 방법은 가능하며 그 mask 오류까지 최종 평가에 포함함. 현재 접경 모델은 RGB-D만 사용하는 부분 경로지만, 물체 소속과 관계 feature를 완성해 제거 효용까지 연결한 경로는 아직 없음.

**실행 경로 구분:** 현재 `BoundaryDecoder.forward(rgb, depth, dino)`와 예측 경계의 depth 순서 계산은 GT 없이 가능한 연산임. 다만 `run_boundary_fp32.py`의 평가 runner는 GT cache를 읽으므로 실환경용 독립 실행 경로와 구분함. 후속 검증은 GT 자료 없이 RGB-D를 로드하여 예측 파일을 완성한 뒤 별도 평가기가 GT를 읽는 방식으로 구성해야 함. 기존 GT utility는 관계뿐 아니라 면적·count baseline부터 GT 의존이므로 수치 일부만 교체해 실환경 경로로 간주하지 않음.

#### 원본 영상에서 물체 간 접경을 예측함

책 A 바로 옆에 책 B가 보이는 두 pixel은 접경의 양성 정답임. 책 A의 글자와 표지 사이처럼 같은 물체 안의 두 pixel은 음성임. 책과 바닥 사이도 이번 물체–물체 관계의 양성에는 포함하지 않음. 단순히 영상의 모든 edge를 찾는 문제와 구분하기 위한 정답임.

각 pixel에서 오른쪽 이웃과 아래쪽 이웃을 각각 검사하여 **2×480×640 score**를 출력함. 두 채널은 접경을 검사하는 영상 방향이며 앞/뒤 깊이 순서가 아님. GT asset ID로 정답을 만들고 unknown·workspace 밖은 loss·평가에서 제외함. **추론 입력에는 GT mask·ID·crop·기준점·workspace·빈 서랍 reference를 넣지 않음.**

```text
Scene RGB → frozen DINO 768×30×40 → 변환·확대 → 32×120×160 ─┐
Scene RGB 3×480×640 → 4×4 pixel을 채널로 재배열 → CNN32 ────┼─ concat96
Depth + 유효성 2×480×640 → 같은 재배열 → CNN32 ────────────┘
         concat96 → CNN 64 → 64 → 32 → 원본 위치로 재배열 → 2×480×640
```

DINO는 물체 외형·문맥을, 원본 RGB-D 경로는 세부 위치를 제공하려는 구성임. 4×4 재배열은 pixel을 평균내지 않고 채널로 옮기는 연산임. 마지막 32채널을 `2방향×4×4위치`로 풀어 원본 위치에 배치함. Frozen DINO를 제외한 학습 parameter는 140,672개임.

같은 구조에서 RGB(DINO+원본 RGB), Depth, RGB-D를 비교함. 사용하지 않는 입력만 0으로 두고 같은 seed의 초기값·영상 순서를 맞춤. 각 3 seeds, 12 epochs를 학습함. 기존 16개 asset pool의 scene-key별 train/val/test **640/320/640영상**을 사용하며 Phase 36–37의 기존 장면 재평가임.

실제 접경을 놓치지 않는 정도가 **recall**, 표시한 접경 중 정답인 비율이 **precision**임. **F1은 두 값을 함께 반영하며 높을수록 좋음.** 주 지표 exact F1은 방향과 원본 pixel 위치가 정확히 맞아야 인정함. 2px 위치 오차를 허용한 값은 별도 보조 지표임.

Flat은 이웃 depth 차이 ≤5mm인 물체 간 접경, depth-step은 ≥20mm 단차임. FPR은 같은 물체 안의 단차를 다른 물체의 접경으로 잘못 표시한 비율이며 낮을수록 좋음.

| 방법 | Exact F1 ↑ | 2px tolerant F1 ↑ | Flat 접경 recall ↑ | 같은 물체 depth-step FPR ↓ |
|---|---:|---:|---:|---:|
| RGB: DINO + 원본 RGB | 0.313310 | 0.706900 | 0.438207 | 0.082137 |
| Depth 학습 head | 0.019448 | 0.089586 | 0.339208 | 0.143049 |
| RGB-D 학습 head | 0.314580 | 0.687858 | 0.384019 | 0.065777 |
| 직접 depth 단차 기준선 | **0.351372** | 0.398220 | 0.000000 | 0.970060 |

학습 모델은 3-seed 평균, 직접 단차는 deterministic 단일 계산임. Image→scene-key 평균 규칙은 같음. 서로 다른 threshold를 validation에서 선택했으며 threshold는 test에서 조정하지 않음. 직접 단차 기준선의 판정 기준은 약 **20.303mm**임.

RGB-D−RGB exact-F1 차이는 **+0.001270**, 95% 구간 `[-0.000137,+0.002688]`임. Seed별 차이는 `+0.005801 / −0.000823 / −0.001168`이며 사전 gate를 통과하지 못함. Depth 학습 head 대비 개선은 있으나 RGB 대비 일관된 우위나 직접 depth 단차 대비 exact-F1 우위는 확인하지 못함. 따라서 이번 결합 decoder를 최종 Complexity 모델로 채택하지 않음.

RGB-D의 낮은 같은-물체 FPR에는 recall 감소가 동반됨. 전체 boundary recall은 RGB `0.462481`, RGB-D `0.408396`이고 mixed-patch recall도 `0.471848→0.416505`임. 동일 recall에 맞춘 비교가 아니므로 낮아진 FPR만으로 depth가 개선 원인이라고 단정하지 않음. Depth-only head의 낮은 성능은 이 입력 표현·작은 decoder·12 epoch 조건의 결과이며 depth의 정보 한계나 최적 depth 모델 성능이 아님.

직접 단차는 큰 depth-step 접경의 recall `0.993462`를 보이지만, ≤5mm 접경은 threshold 때문에 놓침. 같은 물체 내부의 ≥20mm 단차도 FPR `0.970060`으로 대부분 다른 물체처럼 표시함. 양쪽 endpoint가 GT 물체 표면인 곳으로 평가를 제한하면 direct-depth exact-F1은 `0.727052`지만, 이것은 GT가 평가 영역을 제한한 별도 진단이며 GT-free 추론 성능 향상으로 사용하지 않음. 물체–배경 외곽과 물체–물체 접경을 구분하는 것이 핵심 남은 문제임.

![Full-resolution boundary comparison](img/complexity/rgbd_boundary_fixed_scenes_20260921.png)

RGB/depth/GT/RGB head/Depth head/RGB-D head/직접 단차 순서임. 첫 test key·각 category 첫 pool·center view·seed0를 고정함. 노랑=GT, 초록=정확한 예측, 빨강=오검출, 파랑=누락임. 표시선만 1px 팽창하며 실제 평가는 원본 위치에서 수행함.

![Boundary metrics and failure types](img/complexity/rgbd_boundary_metrics_20260921.png)

예측 접경에서 원본 depth를 직접 비교하는 경로도 구현함. 한쪽이 2.90m, 다른 쪽이 2.94m면 전자가 camera에 더 가까움. 10mm 이하 차이는 불명확으로 두고 5/10/20mm 민감도를 저장함. 이는 **두 관측 표면의 깊이 순서**이며 물리적 접촉·숨은 가림·물체 ID가 연결된 완성된 그래프는 아님.

<details>
<summary>참고 기록: GT 숫자 입력의 제거 회귀 — 실환경 추론 경로에서 제외</summary>

#### 정확한 관계가 개수 이상의 제거 효과 정보를 주는지 검사함

이 실험에는 RGB·DINO feature 대신 `[log면적, count, 접경 이웃 수, 깊이 순서 요약, 불명확 비율]`의 5개 숫자를 Ridge 회귀에 입력했음. 물체 구분·집계에 GT ID를 사용했으므로 아래 수치를 RGB-D 관계 추론 성능으로 사용하지 않음.

Phase 35의 실제 더미 **701 object-view 조건·49 views·10 layouts**를 재사용함. 한 물체만 정적으로 제거하고 다른 물체를 고정했을 때 `새로 보인 다른 물체 면적 / 제거한 물체의 원래 가시 면적`을 예측함. 여기서는 GT ID로 물체를 정확히 구분한 관계를 사용하며 위 경계 모델의 출력은 사용하지 않음.

Count와 관계는 같은 96px 범위에서 계산함. **접경 이웃 수**는 해당 물체와 영상에서 맞닿은 다른 물체 수임. **깊이 순서**는 접경에서 어느 쪽 표면이 앞에 있는지, **불명확 비율**은 깊이 차이가 10mm 이하여서 순서를 구분하지 않은 비율임. 면적을 모든 비교에 포함하여 큰 물체의 효과를 함께 고려함.

| 제거 효과 예측 입력 | 실제 효과와 예측 순위의 상관 ↑ | 노출 비율 MAE ↓ |
|---|---:|---:|
| 면적 + 국소 count | 0.550088 | 0.275112 |
| 위 입력 + 접경 이웃 수 | 0.631521 | 0.253211 |
| 위 입력 + 깊이 순서·불명확 비율 | **0.801487** | **0.212034** |

순위 상관은 실제로 더 많은 다른 물체를 드러낸 후보를 예측에서도 높게 두는지 나타냄. 1에 가까울수록 순서가 잘 맞으며 성공 확률이나 Complexity 점수가 아님. 한 layout의 모든 camera를 평가용으로 빼고 나머지 9개로 학습하는 과정을 10번 반복함. 관계 전체를 추가한 개선은 10/10 layouts에서 양수였음. 깊이 순서와 불명확 비율은 함께 추가했으므로 방향 단독 효과는 분리하지 않음.

![Observed relations and static removal utility](img/complexity/observed_relations_utility_20260921.png)

**정확히 구분한 물체 사이의 관측 관계가 면적·개수 이상의 제거 효과 정보를 가진다**는 근거임. 경계 모델의 예측 관계, 실제 집기·낙하·target 탐색 성공까지 평가한 것은 아님. 10 layouts가 asset·capture batch를 공유하므로 독립된 대규모 일반화 결과로 확대하지 않음.

</details>

#### 현재 판단과 다음 구현 대상

후속 작업은 **관계 표현의 생성부터 최종 평가까지 단일 RGB-D에서 연결하는 것**임. GT 관계 입력의 회귀를 후속 모델로 사용하거나 그 숫자를 추론에서 요구하지 않음. Depth의 경계 위치를 유지하면서 물체–배경, 같은 물체의 내부 단차, 서로 다른 물체의 접경을 구분하는 방법을 비교함. Depth 차이가 작은 접경도 계속 평가함.

이번 RGB-D concat decoder는 채택하지 않음. 물체 소속 표현·경계 모듈·직접 관계 예측 중 어떤 구성이 오류를 줄이는지 판단한 뒤 **예측 관계에도 제거 효과의 추가 정보가 남는지** 검사해야 함. 최종 Complexity scalar·추가 encoder·다중 뷰 teacher·fusion은 아직 채택·실행하지 않음.

독립 수치 검산과 unit tests 18개를 통과함. 정밀도 수정·GT·학습·평가 상세는 `development_log.md` Phase 38과 공개 `agent.md` C8·C9에 보존함. 코드·checkpoint·원시 배열은 로컬에 둠.

## 4. 보존한 Density Pilot의 구현과 GT

아래는 **Phase 33에서 실제로 구현·평가한 모델**의 상세임. 연구 흐름과 분리해 보존하며, 이후 경계·관계 진단의 모델이나 최종 Complexity 구조와 구분함.

<details>
<summary>Pilot 목적·입출력 규격과 고정 camera reference</summary>

### 연구 목적과 현재 구현 범위

Complexity는 target identity와 무관하게 **물체 더미 내부의 국소 구성과 물체 간 관계 차이**를 표현하는 stream임. Similarity의 target–scene 외형·의미 관계, Occlusion의 target별 가림 가능성을 scene 자체의 구조 정보로 보완하는 것이 목적임.

현재 구현은 **가시 segmentation label-group 개수와 점유율을 예측하는 density pilot**임. 동일 조건의 RGB-D/depth-only 비교에서 RGB의 추가 정보를 확인했고, learned55와 direct geometry9를 합친 공통 feature 규격을 구현함. 후속 근접도·정적 제거·표현 진단을 통해 count 예측, 국소 표면 관계, 물체 선택 효용을 구분할 근거도 확보함. 이 결과를 바탕으로 최종 Complexity가 감독할 관계와 이를 평가할 기준을 정할 필요 있음.

Count·occupancy·depth 변화는 각각 가시 그룹 수, 피복 비율, 표면의 깊이 변화를 나타냄. 이 관측량의 예측은 구현됐으며, 구조적 Complexity로 확장하려면 같은 값에서도 달라지는 배치·가림 관계를 확인할 필요 있음. 넓은 책 한 권과 다물체 밀집은 모두 높은 occupancy를 가질 수 있고, 기울어진 단일 물체가 같은 높이의 여러 물체보다 큰 depth 변화를 만들 수도 있음.

다음 **가상 장면 비교**는 RGB와 depth가 제공하는 단서를 정리한 것임. 두 장면의 복잡도 순위를 정하려면 별도의 구조 기준이 필요함.

| 장면 | Depth 관측 | RGB의 보완 단서 | 추가로 구분할 관계 |
|---|---|---|---|
| 같은 높이의 책·상자·장난감 | 윗면 depth가 비슷하여 넓은 단일 표면처럼 보일 수 있음 | 색·무늬·윤곽·문맥을 통한 영역 구분 | 물체 구분 이후의 접촉·지지 관계 |
| 기울어진 큰 책 한 권 | 영상 위치에 따라 depth가 크게 변함 | 변화가 하나의 책 내부에서 이어진다는 단서 | 책 아래의 관측 불가능한 물체와 불확실성 |

```text
같은 높이의 여러 물체                  기울어진 책 한 권
RGB:   [책 A] [상자 B] [장난감 C]      RGB:   [        책 A        ]
depth:  2.90    2.90      2.90 m        depth:  2.85 → 2.90 → 2.95 m

낮은 depth 변화 ≠ 한 물체              높은 depth 변화 ≠ 여러 물체
```

### 입력·출력 규격

고정 five-camera rig의 각 시점을 독립 처리하는 입력·출력 경로를 구현·검증함. 현재 범위는 한 시점의 관측량 예측이며, 다중 시점 융합과 숨겨진 물체 복원은 이 모델의 평가 대상에 포함하지 않았음.

| 항목 | 규격 | 역할 |
|---|---|---|
| Scene RGB | `B×3×480×640` | Target image·category text 없이 scene 외형과 문맥 입력 |
| Scene depth | `B×1×480×640`, meter | Image-plane **axial-Z depth**; camera부터 표면까지의 유클리드 ray distance와 구분 |
| Camera별 고정 reference | Workspace mask와 empty-drawer depth, 각각 `480×640` | 서랍 관심 영역과 빈 배경을 정의 |
| Auxiliary prediction | `B×4×30×40` | 48/96/160px count score 3개와 patch occupancy 1개 |
| Stream feature `F_C` | `B×64×30×40` | 학습 feature 55개와 직접 depth cue 9개; 향후 fusion용 중간 표현 |

`B`는 batch의 영상 수이며 공간 격자는 `480/16 × 640/16 = 30×40`임. 영상 한 장의 출력은 `1×64×30×40`이며, 각 위치에 64개 feature 성분이 존재함. 학습한 55개 channel에 개수·접촉·가림 등의 의미를 개별 지정하지 않으며, 직접 cue 9개만 고정된 계산 의미를 가짐. **`F_C` 자체에는 sigmoid를 적용하지 않고 4개 auxiliary map에만 적용함.**

Workspace mask와 empty depth는 camera별 고정 배경 정보임. 매 scene의 물체 정답인 segmentation과 구분하며, rig를 이동하면 reference 정합을 다시 검증해야 함. Density 추론에는 scene segmentation·asset 이름·target reference를 입력하지 않음. 학습 정답 생성과 Phase 34–36의 GT 기반 진단 조건은 4·5절에 별도로 정리함.

</details>

<details>
<summary>Pilot 모델 구조·모듈 선택 이유·depth cue 계산</summary>

### Pilot 모델 구조

RGB는 frozen DINOv3에서 위치별 feature를 추출하고, depth는 고정 수식으로 9개 cue를 계산함. 두 branch의 학습 projection과 fusion으로 feature 55개를 만든 뒤 원래 depth cue 9개를 concat하여 `F_C64`를 구성함. Auxiliary head는 이 feature에서 네 감독값을 예측함.

```mermaid
flowchart LR
    I["Scene RGB<br/>B x 3 x 480 x 640"] --> P["RGB /255 + ImageNet normalization"]
    P --> D["Frozen DINOv3 ViT-B/16<br/>layer 11, normalized dense feature<br/>B x 768 x 30 x 40"]
    D --> R["1x1 Conv 768 to 64<br/>GroupNorm 8 + GELU<br/>B x 64 x 30 x 40"]
    Z["Scene axial-Z depth<br/>B x 1 x 480 x 640"] --> G["Deterministic depth cues<br/>B x 9 x 30 x 40"]
    E["Fixed camera workspace<br/>+ empty-drawer depth"] --> G
    G --> Q["3x3 Conv 9 to 64<br/>GroupNorm 8 + GELU<br/>B x 64 x 30 x 40"]
    R --> C["Channel concat<br/>B x 128 x 30 x 40"]
    Q --> C
    C --> L["3x3 Conv 128 to 64<br/>GroupNorm 8 + GELU<br/>1x1 Conv 64 to 55<br/>B x 55 x 30 x 40"]
    L --> F["Channel concat = F_C<br/>B x 64 x 30 x 40"]
    G --> F
    F --> H["1x1 Conv 64 to 4 + sigmoid<br/>B x 4 x 30 x 40"]
    H --> A["Count scores: 48 / 96 / 160 px<br/>+ patch occupancy"]
    F -.-> U["Future S + O + C fusion<br/>not implemented"]
```

### 위치별 차원 변화

```text
RGB:    DINO 768 → projection 64 ─┐
Depth:  cue 9   → projection 64 ─┴→ concat 128 → fusion 55 ─┐
        cue 9 ─────────────────────── direct path ──────────┴→ F_C64
                                                               ↓
                                                        auxiliary map 4

공간 격자: 모든 단계에서 30×40 = 1,200개 출력 위치 유지
```

Concat은 같은 위치의 channel을 이어 붙이는 연산임. `64+64→128`, `55+9→64`는 원소별 덧셈과 구분함. 이후 convolution이 channel과 이웃 위치의 정보를 학습 가중치로 혼합함.

### Pilot 내부 모듈

| 모듈 | 입력 → 출력 | 역할과 선택 이유 | 적용 범위와 추가 검증 |
|---|---|---|---|
| Frozen DINOv3 layer11 | RGB → `768×30×40` | 1,200개 patch의 외형·문맥 표현 제공. Backbone을 고정하여 작은 RGB-D head 비교에 사용 | Pilot과 A probe에서 사용한 고정 RGB 표현 |
| RGB projection | `1×1 Conv 768→64` | 위치를 유지하면서 RGB feature 폭을 압축 | Matched RGB-D/depth-only pilot의 공통 head 규격 |
| Depth projection | `3×3 Conv 9→64` | 직접 cue와 주변 격자 정보를 결합 | Image-space 이웃 연산을 구현함; 물체별 graph 연산과 구분 |
| RGB-D fusion | `Concat128 → 3×3 Conv64 → 1×1 Conv55` | 두 관측을 결합한 학습 feature 생성 | Count/occupancy 예측을 학습함; 관계 정보의 보존 여부는 별도 평가 필요 |
| GroupNorm 8 + GELU | 64-channel 중간 표현 | Group별 정규화와 비선형 변환 | RGB-D와 depth-only 비교에 동일하게 적용 |
| Direct geometry path | `learned55 + geometry9 → F_C64` | 학습 bottleneck 이후에도 원래 기하값을 제공 | 9개 직접 cue와 학습 feature를 함께 반환하는 경로까지 구현 |
| Auxiliary head | `1×1 Conv 64→4 + sigmoid` | 세 count score와 occupancy를 예측하여 학습 신호 제공 | 네 관측량의 예측을 평가함; 구조적 GT와 target 존재 확률은 별도 문제 |

DINO는 freeze하고 RGB/depth projection, fusion convolution, auxiliary head만 학습함. 세 stream을 결합할 `64×30×40` 규격까지 맞췄으며, 의미 정렬과 탐색 효용은 향후 fusion 평가에서 확인할 필요 있음. 모듈별 역할은 위 표에, 전체 pilot의 비교 결과와 후속 검증은 5절에 정리함.

### RGB feature 추출과 projection

RGB를 255로 나눈 뒤 ImageNet mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`로 channel별 정규화함. 사전학습 encoder의 입력 scale을 맞추는 전처리임.

DINOv3 ViT-B/16은 16×16 patch를 처리하며, index 11인 마지막 12번째 block의 768-D dense token을 사용함. `norm=True`는 backbone의 LayerNorm 적용을 뜻함. Density pilot에서는 별도 token L2 normalization을 하지 않으며, Phase 36 pair probe의 L2 normalization과 구분함.

원본 patch의 `16×16×3=768`개 RGB 값과 768-D token은 차원 수만 같음. Token은 학습 projection과 attention을 거쳐 주변·전체 영상 문맥이 반영된 표현으로, 원본 pixel의 단순 flatten이 아님. 동일한 색도 책·바닥 등 문맥에 따라 다른 feature를 만들 수 있으나 channel별 의미가 고정된 것은 아님.

`1×1 Conv 768→64`는 각 위치의 768개 feature에 학습 가중치를 적용하여 64개 가중합을 생성함. 공간 해상도는 유지하며 GroupNorm/GELU 이후 depth branch와 결합함.

### Depth reference와 유효 pixel

Depth 입력은 optical axis 방향의 axial-Z를 meter로 저장한 값임. 화면 가장자리까지의 비스듬한 ray 길이와 구분하며, 여기서 계산하는 image-space cue를 3D 곡률로 해석하지 않음.

Scene depth $d$와 동일 camera의 empty depth $d_0$가 모두 유한한 양수이고 workspace 안에 있는 pixel만 유효함. Empty reference 대비 변위는 다음과 같음.

$$
\Delta(u,v)=d_0(u,v)-d(u,v).
$$

양수는 빈 배경보다 camera 쪽으로 나온 표면을 뜻함. 음수는 reference 정합·관측 차이 등에서도 발생할 수 있으므로 특정 물체 상태로 단정하지 않음. Roughness 계산에서는 음수를 보존하여 0으로 clipping할 때 생기는 인위적 경계를 방지함.

| Empty `d₀` | Scene `d` | `Δ=d₀−d` | 처리 — 가상 계산 예 |
|---:|---:|---:|---|
| 3.00m | 2.90m | +0.10m = +100mm | `Δ>15mm`이므로 direct foreground에 포함 |
| 3.00m | 2.99m | +0.01m = +10mm | 앞쪽 표면이지만 foreground threshold 미달 |
| 3.00m | 3.02m | −0.02m = −20mm | Roughness에는 부호를 보존, foreground에서는 제외 |
| 3.00m | 0 | 제외 | Invalid를 3m 앞의 표면으로 해석하지 않음 |

### 직접 depth cue 9개

| Channel | 계산 | 역할 |
|---|---|---|
| 1. Normalized depth | 유효 patch depth 평균을 `(mean−2.5m)/(3.5m−2.5m)`로 변환 후 `[0,1]` 제한 | 절대 깊이 제공 |
| 2. Direct foreground occupancy | 유효 pixel 중 `Δ>15mm`인 비율 | Empty reference보다 충분히 앞선 표면 비율 |
| 3. Depth validity | Patch 256px 중 scene/empty depth·workspace가 모두 유효한 비율 | 관측 지원 영역 제공 |
| 4–6. Plane residual RMS | 48/96/160px window의 `Δ`에 affine plane 적합 → 잔차 RMS `/30mm` → `[0,1]` 제한 | 일정한 image-space 기울기 제거 후 남는 변화 |
| 7–9. Gradient variation | 각 window의 수평·수직 유효 인접 pixel `Δ` 차분 분산 합의 제곱근 `/20mm` → `[0,1]` 제한 | 1px first difference의 불균일성 |

첫 세 channel은 16×16 patch를 요약함. 평균 depth가 2.90m이면 normalized depth는 `(2.90−2.50)/(3.50−2.50)=0.40`임. 유효 pixel이 없으면 평균·occupancy·validity를 모두 0으로 둠.

| 가상 patch: 전체 256px, 유효 224px, foreground 168px | 계산 | 분모의 의미 |
|---|---:|---|
| Direct occupancy | `168/224=0.75` | 관측 가능한 영역 중 foreground 비율 |
| Validity | `224/256=0.875` | 전체 patch 중 관측 가능한 비율 |

Direct occupancy는 depth threshold의 표면 비율을 제공하고, segmentation occupancy GT는 알려진 물체 영역 비율을 제공함. 이처럼 각 cue의 계산 의미를 구분했으며, 깊이·roughness와 구조적 관계의 연결은 추가 평가가 필요함.

### Plane residual과 gradient variation

Plane residual은 window의 유효 pixel 집합 $V_s(p)$에서 affine plane을 최소제곱 적합한 뒤 RMS 오차를 계산함. $x,y$는 영상 크기로 나눈 image-space 좌표임.

$$
(a_s,b_s,c_s)=\arg\min_{a,b,c}\sum_{q\in V_s(p)}
\left[\Delta(q)-(ax_q+by_q+c)\right]^2,
$$

$$
r_s(p)=\sqrt{\frac{1}{|V_s(p)|}\sum_{q\in V_s(p)}
\left[\Delta(q)-(a_sx_q+b_sy_q+c_s)\right]^2}.
$$

$a_s,b_s,c_s$는 현재 window의 depth에서 직접 계산한 plane 계수이며 network의 학습 parameter가 아님. 잔차를 제곱 평균한 뒤 제곱근을 취하므로 양·음의 오차가 상쇄되지 않음.

Gradient variation은 유효 인접 쌍의 first difference $g_x,g_y$에 대해 $\sqrt{\mathrm{Var}(g_x)+\mathrm{Var}(g_y)}$로 계산함. 두 cue의 차이는 다음과 같음.

| 비교 | Plane residual | Gradient variation |
|---|---|---|
| 측정 대상 | Window 전체를 하나의 plane으로 설명한 잔차 | 인접 pixel 변화량의 불균일성 |
| `Δ=0,10,20,30mm`인 선형 변화 | Plane으로 설명 가능하여 잔차 0 | 차분이 모두 10mm이므로 분산 0 |
| 여러 면·단차·곡률 | 단일 plane으로 설명되지 않는 오차 발생 | 차분이 `0,30,0mm`처럼 달라지면 분산 발생 |
| 해석 한계 | 단일 곡면 물체도 잔차를 만들 수 있음 | Sensor noise와 물체 경계 모두 반응 가능 |

두 cue는 선형 기울기에 대한 반응을 줄이도록 구성함. 각각의 추가 정보는 독립 ablation으로 확인할 필요 있음. Plane 계산에는 전체 window 면적의 유효 비율 ≥25%, 유효 pixel ≥6, 비퇴화 좌표 조건이 필요함. 지원이 부족한 roughness는 0으로 두고 workspace 밖에는 cue를 생성하지 않음. 이때 0은 평탄함의 확정 판정이 아님.

정규화 예는 residual 6mm → `6/30=0.20`, residual 45mm → `45/30=1.5`를 1로 제한, gradient variation 4mm → `4/20=0.20`임. 30mm/20mm는 입력 scale을 맞추는 고정 상수이며 승인된 복잡도 기준이나 최적 threshold가 아님.

### Empty-reference 보정

V2에서는 동일한 scene/empty depth의 곡면·단차 배경에 대해 **여섯 roughness channel이 0이 되는 보정**을 확인함. V1 raw-depth cue가 빈 서랍의 벽·곡면까지 clutter처럼 반응한 오류를 수정한 결과임. Plane 제거만으로 남았던 고정 배경 반응을 줄이기 위해 **`roughness_reference="empty_difference"`**를 적용했으며, `scene_depth=empty_depth`이면 `Δ=0`이 됨. Normalized depth·occupancy·validity의 정의는 유지함.

함수의 기본값 `scene`은 V1 호환용이므로 실제 실행에는 checkpoint protocol의 V2 설정을 사용해야 함. 이 보정은 고정 camera와 empty scene의 정합을 전제로 하며, camera/FOV·drawer 위치·depth calibration 변경 및 실제 sensor의 강건성은 별도 검증 대상임.

### Depth projection과 RGB-D fusion

`3×3 Conv 9→64`는 현재 위치와 주변 8개 위치의 cue를 사용함. 일반적인 내부 위치에서 출력 channel 하나의 입력은 `3×3×9=81`개이며, padding 1로 `30×40` 크기를 유지함. 물체별 소속을 지정한 graph 연산은 아님.

RGB/depth projection을 concat한 128개 channel에 `3×3 Conv 128→64`, GroupNorm/GELU, `1×1 Conv 64→55`를 적용함. 동일 높이의 서로 다른 RGB 영역과 단일 물체 내부의 큰 depth 기울기를 함께 처리할 수 있도록 구성함. 이 결합의 count 예측 이득은 확인했으며, 어떤 물체 관계가 feature에 남는지는 관계별 진단이 필요함.

GroupNorm은 sample별 64 channels를 8개 group, group당 8 channels로 나누어 공간 위치와 함께 정규화함. 다른 영상의 batch 통계에 의존하지 않음. GELU는 선형 가중합 사이에 비선형 반응을 추가함.

### Direct path: 학습 feature 55개와 기하 cue 9개

학습 branch를 거치면서 기하 cue가 다른 feature와 혼합되더라도, 후속 head가 원래 값을 직접 사용할 수 있도록 우회 경로를 유지함. 예를 들어 학습 feature가 RGB 문맥을 강조해도 depth validity는 직접 참조 가능함.

```text
geometry9 ──→ depth projection ──→ RGB-D fusion ──→ learned55 ──┐
     └──────────────── direct9 유지 ──────────────────────────┴→ F_C64
```

Direct cue의 계산식을 유지하면서 이를 읽는 auxiliary head의 weight를 학습함. 기하값을 보존하되 출력 기여도는 조정할 수 있는 경로임. `55+9`를 적용한 pilot까지 평가됐으며, 다른 채널 배분에 대한 상대 이득은 별도 비교가 필요함.

### Auxiliary head: `F_C64 → 4 maps`

`F_C64`는 후속 fusion에 제공할 중간 표현이고, 네 map은 학습 정답과 비교하기 위한 readout임. 위치별 64개 feature의 가중합과 bias로 logit 4개를 만든 뒤 각각 sigmoid를 적용함.

$$
z_k(p)=\sum_{c=1}^{64}w_{kc}F_{C,c}(p)+b_k,
\qquad \hat y_k(p)=\frac{1}{1+e^{-z_k(p)}},\quad k=1,2,3,4.
$$

Logit 0의 sigmoid 값은 0.5임. Count channel에서는 `0.5×16=8` label-groups, occupancy에서는 알려진 영역의 절반에 해당함. 같은 값이라도 감독 대상에 따라 의미가 달라지며 target 존재 확률 50%를 뜻하지 않음.

Sigmoid는 bounded GT를 예측하는 마지막 head에만 적용함. `F_C` 전체에 적용하면 학습 feature 55개까지 0–1로 제한되므로 중간 표현은 그대로 유지함. 네 map과 `F_C`를 함께 반환하는 경로를 구현했으며, 최종 fusion에서 중간 feature의 추가 효과를 확인할 필요 있음.

</details>

<details>
<summary>Count·occupancy GT 생성, window·정규화, loss와 추론 출력</summary>

### 감독 경로와 추론 경로

저장된 scene segmentation·mapping으로 count/occupancy GT를 만들고, 이를 RGB-D 예측과 비교해 loss를 계산함. **Scene segmentation은 density 모델의 입력 feature로 전달하지 않음.** 추론에는 RGB-D와 고정 reference만 필요함.

```mermaid
flowchart LR
    subgraph TRAIN["학습"]
        TIN["Scene RGB-D<br/>+ fixed references"] --> NET["Trainable RGB-D pilot<br/>frozen DINO included"]
        NET --> PRED["4 predicted maps"]
        SEG["Stored scene segmentation<br/>+ mapping + workspace"] --> GT["3 count GT maps<br/>+ occupancy GT"]
        GT --> LOSS["Masked SmoothL1 loss"]
        PRED --> LOSS
    end
    subgraph INFER["추론"]
        NIN["New scene RGB-D<br/>+ matching fixed references"] --> SAVED["Saved model"]
        SAVED --> OUT["4 maps + F_C64<br/>geometry9 + validity"]
    end
```

Phase 34–35는 GT label을 계산 입력으로 사용한 teacher 진단이고, Phase 36은 GT로 순수 patch와 평가 pair를 선정한 조건부 평가임. Density pilot의 segmentation 없는 추론과 이 진단 조건을 구분함.

### 가시 segmentation label-group count

저장 mapping은 asset 이름에 색을 연결하며 영속적인 physical instance ID는 아님. 서로 다른 asset의 색 충돌이나 동일 asset의 복수 배치는 한 그룹으로 합쳐질 수 있음. Pilot은 **서로 다른 색 그룹을 한 번씩** 세며, 동일색 조각이 떨어져 있어도 중복으로 세지 않음.

Workspace를 $\Omega$, 알려진 색 그룹을 $g$, 그 그룹의 workspace 내 가시 pixel 집합을 $A_g$로 정의함. Patch 중심 $p$의 한 변 길이 $s$인 정사각 pixel window를 $W_s(p)$라 하면:

$$
n_s(p)=\sum_g
\mathbf{1}[|A_g|\geq32]\,
\mathbf{1}[|A_g\cap W_s(p)|\geq16],
\qquad s\in\{48,96,160\}.
$$

$$
y_s(p)=\min\left(1,\frac{n_s(p)}{16}\right).
$$

전체 workspace에서 ≥32px 보이고 해당 window에 ≥16px 들어온 그룹을 count에 포함함. 작은 흔적의 과도한 계수를 줄이려는 고정 threshold를 적용해 pilot을 평가했으며, 다른 물체 크기·camera에 적용할 때는 threshold 적합성을 확인할 필요 있음.

### Patch·count window·정규화의 구분

| 항목 | 값 | 역할 |
|---|---|---|
| 출력 sampling | `16×16` patch | `480×640 → 30×40` 출력 위치 결정 |
| Count 포함 조건 | Window 교집합 ≥`16 pixels` | 최소 가시 증거 면적; patch 한 변과 다른 단위 |
| Count 정규화 | `count ÷16` | 원본 library의 16개 asset을 기준으로 한 고정 출력 scale |
| Count window | `48/96/160px` | 위치 주변의 집계 범위; 3/6/10 patch 폭에 해당 |

Window는 원본 pixel에서 계산하며 출력은 동일 patch 위치에 저장함. 96px window가 출력 간격이나 정규화 분모를 96으로 바꾸는 것은 아님. 아래 좌표는 0-based이며 구간 끝은 포함하지 않음.

```text
원본: 높이480 × 너비640 → 16×16 patch → 30행×40열 =1,200개 출력 위치

patch (행12, 열20): y=[192,208), x=[320,336) → 256 pixels
window 중심: (y=200, x=328)

count48:  y=[176,224), x=[304,352) →  48×48= 2,304 pixels
count96:  y=[152,248), x=[280,376) →  96×96= 9,216 pixels
count160: y=[120,280), x=[248,408) →160×160=25,600 pixels

가상 count:                  1개      3개      5개 label-groups
정규화:                      1/16     3/16     5/16
같은 위치의 y_count[:,12,20]: [0.0625,  0.1875,  0.3125]
```

| 그룹별 가상 가시 면적 | Workspace 전체 | Window 내부 | 처리 |
|---|---:|---:|---|
| A | 100px | 20px | 두 조건을 만족하여 count에 1 추가 |
| B | 100px | 12px | Window 내부 ≥16px 조건 미달로 제외 |
| C | 20px | 20px | 전체 ≥32px 조건 미달로 count 제외; 알려진 색이면 occupancy에는 포함 |

세 그룹이 포함된 window의 GT는 `3/16=0.1875`임. Scene별 min–max 정규화는 하지 않고 count가 16을 초과할 때의 clipping 비율을 기록함. 실제 3개 물체가 색 충돌로 2개 그룹이면 GT는 2를 셈. **Phase 36의 충돌 검출로 과거 density GT·checkpoint·수치를 소급 수정하지 않았음.**

48/96/160px은 국소 영역부터 넓은 문맥까지 비교하는 **image-space window 가설**로 적용함. 이 세 scale의 예측까지 평가했으며, window 수·크기의 개별 이득은 독립 비교가 필요함. Metric 물체 크기·접촉 거리·grasp 범위를 뜻하지 않음. 큰 window에서 count가 커질 수 있으나 `count/16`은 scale 간 동일한 물리 면적당 density가 아님.

### Count validity

Count supervision은 다음 두 조건을 모두 만족한 window에 적용함.

| 조건 | 계산·의미 |
|---|---|
| Workspace coverage ≥95% | 분모는 **전체 정사각 window 면적**임. 영상 밖으로 잘린 부분도 포함 |
| Workspace 내부 unknown nonblack 없음 | Mapping에 없는 pixel을 빈 배경으로 취급하지 않음 |

96×96 window의 전체 면적은 9,216px임. Workspace가 9,000px이면 `9000/9216≈97.66%`로 통과하고, 8,600px이면 약 93.32%로 제외함. Coverage를 통과해도 unknown nonblack이 있으면 count mask는 0임. 낮은 count와 신뢰할 수 없는 GT의 제외를 구분하는 규칙임.

Density pilot은 알려진 동일색 alias를 한 그룹으로 유지함. 반면 Phase 36은 asset identity가 모호한 **충돌 색 자체를 unknown으로 제외**했으므로 label 처리와 metric을 혼용하지 않음.

### Patch occupancy와 loss weight

Occupancy는 count window가 아닌 **16×16 patch 자체**의 비율임. $F$는 알려진 foreground, $U$는 workspace 내 unknown nonblack, $P(p)$는 patch pixel 집합임.

$$
y_{\mathrm{occ}}(p)=
\frac{|P(p)\cap F|}{|P(p)\cap(\Omega\setminus U)|}.
$$

분모가 0이면 GT와 loss weight를 모두 0으로 둠. 그 외 occupancy loss weight는 `알려진 workspace pixel 수 /256`임. Count의 최소 면적 조건에서 제외된 작은 알려진 색 그룹도 occupancy foreground에는 포함함.

```text
가상 patch 전체: 256px
├─ Workspace 밖 16px                 → GT 분자·분모에서 제외
└─ Workspace 안 240px
   ├─ Unknown 40px                   → GT 분자·분모에서 제외
   └─ 알려진 영역 200px               → GT 분모
      ├─ Foreground 150px            → GT 분자
      └─ Background 50px

Occupancy GT =150/200=0.75
Loss weight  =200/256=0.78125
```

`150/256`을 GT로 사용하면 workspace 밖과 unknown을 배경으로 간주하여 occupancy를 낮추게 됨. 현재는 알려진 영역 내부 비율을 GT로 사용하고, 관측 지원 비율을 loss weight로 반영함. Count의 binary ≥95% validity와 다른 규칙임.

Occupancy 1은 알려진 workspace가 물체로 덮였다는 뜻임. 넓은 단일 물체와 다물체 밀집을 구분하는 감독값은 아님.

### Masked SmoothL1 loss

출력 순서는 `[count48/16, count96/16, count160/16, occupancy]`임. Count binary validity 3개와 occupancy의 알려진 workspace 비율 1개를 channel별 weight $M_c$로 사용함.

$$
\mathcal{L}=\frac{1}{4}\sum_{c=1}^{4}
\frac{\sum_{b,p}M_c(b,p)\,\ell_{0.05}(\hat y_c(b,p)-y_c(b,p))}
{\max\left(1,\sum_{b,p}M_c(b,p)\right)},
$$

$$
\ell_\beta(e)=
\begin{cases}
e^2/(2\beta),& |e|<\beta,\\
|e|-\beta/2,& |e|\geq\beta.
\end{cases}
$$

각 channel의 loss를 유효 weight 합으로 정규화한 뒤 네 channel을 동일 가중 평균함. Window별 유효 영역의 크기가 loss 비중을 결정하지 않도록 한 구성임.

| 조건 | 포함 범위 |
|---|---|
| 학습 loss | **Count=0을 포함한** 모든 유효 window |
| Primary metric·checkpoint 선택 | **GT count>0인** 유효 window의 count MAE |

SmoothL1은 작은 오차에는 제곱, 큰 오차에는 절댓값 형태로 반응함. GT `3/16=0.1875`, prediction `4/16=0.25`이면 오차 `0.0625≥β=0.05`이므로 weight 적용 전 loss는 `0.0625−0.05/2=0.0375`임. 오차가 `0.02`이면 `0.02²/(2×0.05)=0.004`임. 이후 validity와 channel별 정규화를 적용함.

### Auxiliary supervision과 gradient 경로

학습 feature 55개 각각에 GT를 지정하지 않음. 네 map의 loss가 head와 앞쪽 projection/fusion으로 역전파되어 count/occupancy 예측에 유용한 feature를 학습하도록 함.

```mermaid
flowchart RL
    LOSS["4 maps와 GT의 차이<br/>masked SmoothL1"] -. "gradient" .-> HEAD["1x1 prediction head<br/>학습 weight 갱신"]
    HEAD -. "learned55 경로" .-> FUSE["RGB-D fusion convolution<br/>학습 weight 갱신"]
    FUSE -.-> RGB["RGB projection<br/>학습 weight 갱신"]
    FUSE -.-> DEP["Depth projection<br/>학습 weight 갱신"]
    DINO["Cached frozen DINO feature<br/>gradient 계산 경계 밖"] -->|"고정 입력"| RGB
    GEO["Deterministic geometry9<br/>gradient 계산 경계 밖"] -->|"고정 입력"| DEP
    GEO -->|"direct9: 읽는 head weight만 학습"| HEAD
```

점선은 gradient 전달, 실선은 고정 feature 입력 경로임. Cached DINO와 직접 depth 수식은 gradient 계산 경계 밖에 있으며, direct9의 값은 유지하되 이를 읽는 head weight는 학습함.

네 auxiliary GT를 이용한 feature 학습과 예측 개선까지 확인함. Count/occupancy가 같은 두 구조에는 동일한 감독값이 주어지므로, 관계 차이가 내부 feature에 얼마나 남는지는 별도 평가가 필요함. 이 구분을 위해 Phase 36에서 frozen 표현의 대응 정보를 진단함.

### Matched 비교와 추론 결과

Frozen RGB feature를 cache하고 작은 head를 AdamW로 학습함. RGB-D/depth-only는 동일 architecture·초기값·sample 순서를 사용하며 depth-only의 DINO 입력만 0으로 설정함. Validation occupied-window count MAE로 checkpoint를 고정한 뒤 test를 평가함.

`inference_complexity.py`는 checkpoint protocol, DINO weight와 고정 reference의 hash를 검사하고 다음 결과를 반환함.

| NPZ 항목 | Shape | 의미 |
|---|---|---|
| `maps` | `(4,30,40)` | 세 count score와 occupancy |
| `features` | `(64,30,40)` | `F_C` |
| `geometry` | `(9,30,40)` | 직접 depth cue |
| `validity` | `(1,30,40)` | 관측 depth의 유효 비율 |

Count score에 16을 곱한 값은 연속적인 가시 label-group 개수 추정치임. 정수 물체 수나 hidden count가 아님. 추론 `validity`도 관측 지원 비율이며 segmentation 기반 GT count mask를 복원한 결과가 아님.

Hash 검사는 학습에 사용한 weight/reference와 다른 파일의 혼입을 확인하는 절차임. 동일 해상도 또는 동일 파일이라는 사실만으로 새 camera의 실제 정합을 보장하지 않음.

구현: `complexity_cues.py`, `complexity_model.py`, `run_complexity_pilot.py`, `inference_complexity.py`. 실행법은 density pilot 문서 (`docs/complexity_results/README.md`)에 정리함.

</details>

## 5. 상세 평가 조건과 결과

**Phase 34–35는 물체 구분 모델을 학습한 과정과 구분되는, 별도의 관계 점수 후보 실험임.** 당시에는 “물체를 이미 정확하게 구분할 수 있다고 가정했을 때, 다른 물체 표면이 가까운 정도를 유용한 국소 점수로 쓸 수 있는가”를 검토했음. 물체 구분은 GT로 주고 점수 자체를 점검한 것임.

| 단계 | 무엇을 확인하려 했는가 | 실제 수행 |
|---|---|---|
| Phase 34 | 서로 다른 물체가 가까운 부분에 점수가 생기는가 | GT로 A·B·C를 구분한 뒤 depth에서 관측 표면 간 거리를 계산 |
| Phase 35 | 그 국소 점수를 물체별로 평균하면 제거할 물체를 고르는 단서가 되는가 | 물체 평균 근접도와, 하나를 제거했을 때 새로 보인 다른 물체 면적 비율의 순위를 비교 |

Phase 34에서 가까운 접경에 반응하는 계산은 확인했음. 다만 물체 사이 간격의 효과를 따로 보려던 통제 실험에서, 영상에 보이는 전체 물체 면적도 3.1746% 변해 사전 허용치 3%를 넘었음. **간격만 바꾼 비교로 충분히 통제되지 않았다는 뜻**이며, 이 수치를 count 예측이나 segmentation의 실패율로 해석하지 않음.

Phase 35는 별도 capture의 실제 asset 더미 10 layouts에서 다른 물체는 고정하고 하나씩 제거한 검사임. 공통 701조건·49 views에서 “평균 근접도가 높은 물체 순서”와 “제거 후 새로 보이는 비율이 큰 물체 순서”를 비교했음. 순위가 함께 움직이는 정도인 Spearman 상관을 view별로 계산하고, layout 안의 view 평균 후 10 layouts에 동일 비중을 주어 평균한 값은 **0.078**로 약했음. 7.8%의 제거 성공률이라는 뜻은 아님.

따라서 해당 **물체 평균 근접도**를 제거 선택 점수나 Complexity GT로 채택하지 않았음. 이 결과가 국소 count 가설을 반박하거나, RGB-D에서 물체를 구분하는 문제를 해결한 것은 아님. 본문 ④의 DINO 실험은 별도로 **기존 feature에서 같은 물체인지 판별할 정보를 읽을 수 있는가**를 확인한 것임.

아래에는 Phase 33–36의 원래 비교 조건·수치·입력 구성과 추가 해석을 보존함. 개별 실행과 추가 사진은 [Development Log](development_log.md)의 해당 Phase에 있음.

<details>
<summary>Phase 33–36: Density 비교, 근접도·제거 진단, DINO pair 구성과 평가</summary>

고정 rig에서 RGB의 density 예측 기여를 확인한 뒤, 국소 관계 후보와 제거 효과를 대조하고 frozen 표현의 대응 정보를 진단함. 각 단계에서 확인한 결과와 이어서 검증할 질문은 다음과 같음.

| Phase·평가 조건 | 확인 결과 | 판단과 다음 검증 |
|---|---|---|
| **33 — RGB-D/depth-only matched 비교**: all16 pools·5 views, train/val/test `3,840/960/960`, 3 seeds | Count MAE **0.8327 → 0.6414**, **22.97% 감소** | 이 조건에서 RGB의 추가 정보를 확인함. 새로운 물체·rig와 관계 표현으로의 확장 효과는 추가 평가 필요 |
| **34 — GT label+depth의 표면 근접도** | 가까운 접경 반응과 단독 물체의 0 반응을 확인함. 투영 면적 편차 **3.1746%>사전 3%**로 면적 통제에는 실패함 | 관측 근접도의 국소 반응과 통제 실패를 구분해 보존함. 실제 asset의 관계를 독립 기준으로 확인할 필요 있음; GT 미승인 |
| **35 — 실제 asset 10 layouts의 정적 제거**, 공통 **701조건/49 views** | 새 노출 비율 Spearman은 근접도 **0.078**, 면적 **0.387**, count **0.291**임 | 30mm 근접도 평균을 제거 순위·GT로 채택할 근거가 부족하다고 판단함. 물체 소속·국소 관계 표현에서 보완할 정보를 먼저 특정할 필요 있음 |
| **36 — GT가 고른 순수 patch의 same-category asset 대응** | AUROC DINO+position **0.998953**, depth+position **0.773882**, RGB-D+position **0.998908**, raw cosine **0.925424** | 기존 feature에서 대응 정보를 잘 읽을 수 있음을 확인함. 다음은 경계·조각 소속·다중 관계의 누락 능력 진단임 |

### Phase 33: Density 예측과 해석 범위

All16 source pools의 고정 five-camera 평가에서 RGB-D는 depth-only보다 count 오차가 작았고 **5/5 camera에서 개선**됨. Primary count MAE는 유효 occupied window의 오차를 영상·세 scale·seed에 걸쳐 평균한 **가시 segmentation label-group 개수 단위**임. 12개 test scene-key cluster의 paired 재표집에 따른 absolute 개선량 95% 구간은 `[0.1790,0.2054]` 그룹이었음.

Occupancy도 RGB-D `0.00854`로 depth-only learned head `0.01524`, 직접 depth occupancy `0.00981`보다 낮은 MAE를 보였음. 가시 개수·점유율의 예측까지 확인됐으며, 동일 occupancy에서 달라지는 단일 물체·다물체 구성을 구분하려면 별도의 관계 평가가 필요함.

③에 제시한 density 그림의 열은 scene RGB / 96px count GT / RGB-D prediction / 절대 오차 / occupancy GT / 직접 depth occupancy / depth plane residual임. Count는 GT-valid window만 표시하므로 표시 밖의 0을 물체 부재로 해석하지 않음. Empty-reference 수정과 평가 조건은 Phase 33에 정리함.

### Phase 34–35: 근접도 teacher와 정적 제거

Phase 34에서는 **GT label로 물체를 구분하고 depth에서 복원한 관측 표면 간 거리**를 계산하여 가까운 접경의 국소 반응을 확인함. 단일 물체 내부 무늬에 반응하는 RGB edge와 구분되는 특성임. 다만 동일 면적 통제는 **3.1746%>3%로 실패**했고, 연속 silhouette도 원근·옆면 노출로 1.6396% 변했음. 이 실패를 단순 raster 오차로 처리하거나 전체 Complexity GT 승인으로 해석하지 않음.

Phase 35에서는 pose가 보존된 추가 capture를 사용해 **50/50 views의 원본 label/depth 정합과 정적 제거 비교 경로**를 확인함. 원본 16개에 `World1`을 더한 17-asset 데이터로 `260714_data`와 별도임. 다른 물체를 고정하고 하나만 제거하여 새로 보이는 다른 물체 면적을 측정했음.

이 비교로 **관측 접경의 근접도와 물체 전체의 정적 노출 효과를 구분할 필요**가 드러났음. 물체 평균 근접도의 상관이 약하여 해당 평균을 선택 점수로 채택하지 않았음. 다음은 물체 소속과 국소·다중 관계에서 표현이 놓치는 정보를 특정하는 것임. 숨겨진 접촉·전체 적층, 실제 grasp·재정착·target 발견·행동 성공률은 각각 추가 검증이 필요함. 현재 결과는 모든 근접 feature의 무용함을 뜻하지 않음. 사진·통제 실패·공통 표본 집계는 Phase 34, Phase 35에 정리함.

### Phase 36 A probe: 평가 과제와 pair 구성

All16 seen-assets의 순수 patch 조건에서 **기존 frozen DINO로 가시 asset 대응 정보를 읽을 수 있음**을 A probe로 확인함. Probe는 고정된 feature에서 특정 정보를 읽는 작은 예측기이며, 이번 과제는 **두 patch의 가시 asset label이 같은가**임. Density CNN의 4-map head와 별도로 readout을 학습하고 DINO는 고정함. 다음 관계 과제를 정하기 전에 기존 표현의 대응 정보부터 확인한 진단임.

| Pair | 예 | 정답·평가 목적 |
|---|---|---|
| Positive | 책 A 표지와 책 A의 다른 내부 patch | 같은 asset label |
| Same-category negative | 책 A와 책 B의 patch | 같은 book category 안에서 다른 asset 구분; primary |
| Different-category negative | 책 A와 사과의 patch | Book과 fruit 등 다른 category 구분 |

Same-category 조건으로 같은 category 안의 서로 다른 asset을 구분하는 능력까지 확인함. 동일 asset을 복제한 physical instance의 구분은 별도 label과 평가가 필요함.

Positive/negative의 정확한 XY offset·anchor category·depth 차이 구간을 맞추고, position-only와 depth-only 비교군으로 위치·깊이 단서의 성능을 함께 확인함. 이 조건을 넘어 남을 수 있는 shortcut은 추가 진단이 필요함.

### Pair feature와 readout

| 구성 | 연산 | 차원 |
|---|---|---:|
| DINO | Patch별 768-D를 L2 정규화 → 두 vector의 절댓값 차이·원소별 곱 concat | `768+768=1536` |
| Depth | Cue9 + 4×4 subblock 평균16 + 유효 비율16 = patch당41 → pair 평균·절댓값 차이 | `41×2=82` |
| Position | 두 patch의 평균 위치와 절댓값 위치 차이 | 4 |
| 전체 | 세 입력 concat → MLP `1622→64→16→1` | **1,622** |

GT label 번호와 category 이름은 feature에 넣지 않음. GT는 순수 patch·pair의 위치 선정과 정답 지정에 사용함.

```text
GT segmentation → 순수 patch·pair 선택 → 위치 p, q ─────────┐
전체 scene RGB-D → frozen feature·depth descriptor ─────────┴→ pair feature 1,622
                                                                  ↓
                                                            MLP same-label score
                                                                  ↓
GT segmentation → 선택한 pair의 정답 1/0 ──────────────────────→ loss / 평가
```

학습 비교군은 동일 MLP 크기·초기값·sample 순서를 유지하고, train-only 정규화 후 제외할 branch를 0으로 설정함. DINO 비교에도 위치 정보가 있으므로 `DINO+position`으로 표기함. Raw cosine은 MLP 학습 없이 정규화 DINO vector의 내적을 사용하는 baseline임.

### 평가 조건과 coverage

| 항목 | 조건·집계 |
|---|---|
| 데이터 | All16 seen assets, train/val/test `8/4/8 scene keys ×16 pools ×5 views` |
| Patch | 동일한 알려진 GT label ≥90%, workspace·valid depth 각각 ≥95% |
| Primary | Same-category **80,024 pairs/604 views/8 scene keys** |
| AUROC 집계 | View별 → key별 pool/view 평균 → 8 keys 동일 평균 → 3 seeds 평균 |
| 전체 test | 150,690 pairs/640 views; pair·view를 독립 scene 수로 세지 않음 |
| 적격 coverage | 알려진 foreground 128,380 patches 중 54,429개, **42.40%** |

Purity ≥90%는 patch 전체 256px 중 최소 231px이 같은 알려진 label인 조건임. 이 조건의 내부 patch 대응을 평가했으며, 경계·mixed patch를 포함한 전체 영상 grouping은 추가 평가가 필요함. 두 물체가 섞인 경계 patch는 상당 부분 제외됐고 unknown/색 충돌은 알려진 foreground 분모에서도 제외함. 따라서 보고된 coverage와 AUROC는 GT가 지정한 순수 내부 patch의 조건부 결과임.

AUROC는 positive가 negative보다 높은 score를 받는 순위 판별 지표임. 동점은 절반 기여, 이상적 순위는 1, 무작위 순위는 대략 0.5임. Score 0.5로 정·오답을 나눈 accuracy나 확률 calibration과 다르므로 **0.998953을 pixel segmentation 정확도 99.8953%로 해석하지 않음.**

순수 patch 대응의 높은 AUROC로 **새 encoder를 추가하기 전에 기존 feature의 활용 가능성을 확인**함. 이 이진 과제는 거의 포화되어 B/C의 추가 효과를 구분하기 어려우므로 현재 도입을 보류함. 다음은 경계·분리된 조각·다중 관계에서 남은 실패와 관측 가능한 label을 특정하는 단계임. 작은 물체·심한 가림·동일 asset 복제·unseen asset 일반화도 별도 검증이 필요함. 자료와 큰 오차 pair 그림은 Phase 36에 정리함.

</details>

## 6. 질문과 답변

### Q1. 왜 Complexity를 count/occupancy로 바로 정의하지 않는가?

고정 rig의 density pilot에서 **가시 label-group 수와 피복 비율의 예측 개선**까지 확인함. Count와 occupancy는 이 두 관측량을 감독하며, 구조적 Complexity로 확장하려면 같은 관측량에서도 달라지는 배치·분리·가림 관계를 구분할 필요 있음.

```text
알려진 영역 200px의 occupancy=1:
[       책 A       ]       또는       [ A ][ B ][ C ][ D ]

Count=3:
[A] [B] [C]                또는       서로 겹친 A/B/C
```

첫 비교에서는 물체 구성, 두 번째 비교에서는 배치 관계가 달라져도 해당 값은 같음. Pilot을 관측량 예측의 기준으로 보존하고, 다음에는 이러한 관계 차이를 어떤 label로 관측·평가할지 정할 필요 있음. 최종 구조적 GT는 그 타당성을 확인한 뒤 결정함.

### Q2. Depth의 거칠기를 그대로 복잡도라고 하면 안 되는가?

동일한 scene/empty depth의 곡면·단차 배경에서 **고정 서랍 구조에 대한 거칠기 반응을 제거**한 결과까지 확인함. Plane/gradient 처리는 일정한 기울기에 대한 반응도 줄이도록 구성함. 다음은 남은 잔차가 단일 물체 곡면·sensor 오차·물체 간 구조 중 무엇을 나타내는지 구분하는 평가임.

Raw depth variance는 기울어진 책 한 권에서도 커질 수 있음. Plane residual은 일정한 기울기를 제거하지만 단일 곡면·noise·다물체 경계는 모두 잔차를 만들 수 있음. 반대로 같은 높이의 여러 물체는 depth 변화가 작아 RGB 외형·문맥 단서가 필요함.

현재 depth cue를 RGB-D 입력으로 유지하면서, RGB가 이러한 원인 구분에 추가로 기여하는지를 관계 과제에서 확인할 필요 있음. Cue의 30mm 정규화 상수와 근접도 진단의 30mm 탐색 반경은 서로 다른 역할이며, 보정된 roughness 자체를 Complexity 정답으로 채택한 것은 아님.

Phase 38에서는 depth를 거칠기 scalar 대신 원본 접경 예측의 입력·비교 기준으로 사용함. 직접 단차는 큰 깊이 변화의 위치를 잘 찾지만 평평한 접경과 같은 물체 내부 단차를 구분하지 못했음. 접경 위치와 양쪽 물체 소속을 함께 표현하는 이유임.

### Q3. 다른 물체와 가까운 가장자리만 찾아서 그 물체를 선택하면 되지 않는가?

GT로 물체 소속을 제공한 조건에서는 **가까운 관측 표면의 국소 반응**까지 확인함. 제거할 물체를 선택하려면 RGB-D에서 소속을 추정하고, 국소값을 물체별로 집계한 뒤 선택 결과와의 관계를 검증할 필요 있음.

```text
국소 관계 map → 물체 A/B/C 소속 → 평균·합·최댓값 등 집계 → 물체별 점수 → 선택 효용 검증
```

Phase 34–35는 GT label로 물체별 소속을 고정하여 관계 후보를 점검함. 분리된 가시 조각의 소속을 RGB-D로 추정하는 단계에서는 잘못된 grouping이 선택 단위를 바꿀 수 있으므로 이 오류를 별도로 평가해야 함.

집계 방식에도 편향이 있음. 가까운 가장자리 10%만 score1, 내부90%가 0인 물체의 평균은 0.1임. 넓은 내부는 평균을 낮추고, 최댓값은 작은 noisy 접경 하나에 민감함. 또한 넓은 책 아래 가려진 면적이 커도 현재 관측된 다른 표면까지의 거리는 클 수 있음.

Phase 35에서 실제로 비교한 **30mm 근접도 물체 평균**은 정적 새 노출 비율과 Spearman 0.078로 약한 상관을 보였음. 이를 통해 해당 평균을 선택 점수로 채택할 근거가 부족하다고 판단함. 다음은 소속·집계 과정에서 사라지는 관계 정보와 보완할 표현을 특정하는 것임. 이 결과를 모든 edge·관계 feature의 무용함으로 확대하지 않음.

### Q4. 물체 대응을 거의 완벽하게 구분했으면 Complexity도 해결된 것인가?

Phase 36–37에서 **GT가 선택한 순수 patch의 가시 asset 대응을 기존 DINO에서 읽을 수 있음**을 확인함. Phase 37은 경계 주변·외형 변화까지 조건을 나눴으며, 분리된 가시 조각도 작은 표본에서 높은 순위 판별력을 보였음. 다음은 이 정보를 혼합 patch와 전체 물체 묶음에 사용하는 방법을 확인하는 것임.

책 A의 두 patch를 비교할 때 정확한 점수를 내는 것과, 영상 전체에서 책 A의 모든 영역을 찾아 하나로 묶는 것은 다른 과제임. 한 patch에 책 A·B가 함께 들어 있으면 어느 한쪽만 정답으로 고르는 방식도 불충분함. Phase 37의 GT block 근사는 이러한 단일-label 변환이 접경을 바꿀 수 있음을 보였지만, DINO의 segmentation 성능을 측정한 결과는 아님.

대응 평가의 범위는 purity ≥90%, 적격 coverage가 알려진 foreground patch의 42.40%인 조건임. 경계의 세밀한 위치·mixed patch·전체 grouping·동일 asset 복제·unseen asset의 평가는 남아 있음. 가시 접경과 깊이 순서가 실제 target 탐색에 얼마나 도움이 되는지도 별도 질문이므로 높은 AUROC를 Complexity 완성 지표로 사용하지 않음.

Phase 38 전체 영상 접경의 exact F1은 RGB 0.313310, RGB-D 0.314580임. GT가 고른 순수 patch의 높은 AUROC와 원본 영상 전체의 경계 복원이 다른 과제임을 확인함.

### Q5. Similarity의 SigLIP처럼 별도 모델을 추가하는가?

A 진단으로 기존 feature의 순수 patch 대응 정보가 충분히 읽힌다는 결과를 먼저 확보했고, Phase 37에서 기존 판별기를 고정해 어려운 조건의 대응과 GT 기반 관계를 점검함. 아래 추가 모델 비교 계획에서는 **A만 완료했고 B/C는 미실행**임. Phase 37이 B/C 모델을 실행한 것은 아니며, 현재 순수 patch 과제에서 새 encoder를 바로 채택할 근거는 없음.

| 단계 | 구성 | 분리할 효과 |
|---|---|---|
| A | 기존 DINO+depth | 기존 표현에서 읽을 수 있는 정보 |
| B | A + 예측한 물체/영역 묶음 | Grouping 자체의 효과 |
| C | B + 사전학습 공간 관계 feature | 동일 영역에서 새 표현이 제공하는 추가 정보 |

Similarity의 SigLIP은 외형만으로 부족한 의미 관계를 보완하려는 설계였음. Complexity에서도 먼저 어떤 경계·소속·다물체 관계를 기존 표현에서 읽지 못하는지 특정해야 함. B/C가 같은 region과 평가 조건을 공유해야 grouping 개선과 사전학습 표현의 효과를 분리할 수 있음. GT region을 제공하면 별도 oracle 결과로 취급함.

추가 모델에서는 `30×40` 위치별 feature에 필요한 token–위치/물체 대응, 관측되지 않은 관계의 추정 범위, 계산량 대비 평가 이득을 확인할 필요 있음. VLM의 문장 응답만으로 이 효과를 대신 평가하지 않으며, 문장이나 scalar를 Complexity GT로 사용하는 계획도 아님. SAM/VLM의 도입 여부는 해당 비교 결과로 판단할 단계임.

### Q6. 학습에서 GT segmentation을 쓰면서 추론은 RGB-D만 쓴다는 것이 모순인가?

Density pilot에서는 **GT segmentation으로 학습하고 RGB-D와 고정 reference로 추론하는 경로**를 구현·검증함. GT는 네 예측 map의 loss 계산에 사용하고 모델 입력 feature와 분리함. 저장된 weight로 추론할 때는 GT와의 비교가 없어 scene segmentation이 필요하지 않음.

| 실험 | GT 역할 | 해석 범위 |
|---|---|---|
| Density inference | 추론에는 미사용 | RGB-D + fixed reference → prediction |
| Phase 34–35 proximity teacher | 물체 label을 거리 계산에 직접 사용 | GT 조건의 관계 후보 진단 |
| Phase 36 A probe | 순수 patch·pair 선택과 정답 지정 | GT가 고른 평가 위치의 대응 판별 |
| Phase 37 고정 readout 평가 | 어려운 pair·기준점 선택과 정답 지정 | 기존 RGB-D feature에서 위치쌍 점수를 계산한 조건부 진단 |
| Phase 37 가시 관계·block 근사 | 물체 ID를 계산에 직접 사용 | GT의 관계 정보와 단일-label 변환을 비교한 진단 |
| Phase 38 접경 학습 | GT는 loss·평가에 사용, 추론에는 미사용 | 전체 RGB-D → 원본 해상도 접경 score |
| Phase 38 제거 utility | GT ID·depth로 관계를 계산 | 관계가 정확히 주어진 경우의 추가 예측 정보 |

GT 활용을 위와 같이 구분하여 density 추론, 관계 후보 계산, 표현 진단을 각각 확인함. 다음 관계 추론을 구현할 때는 teacher·probe가 사용한 GT 소속과 위치 선택을 RGB-D에서 추정했을 때의 오류까지 평가할 필요 있음. Phase 38 접경 모델도 추론에 GT를 입력하지 않음. 관계 효용 진단은 GT를 사용한 별도 실험임.

### Q7. 64개 feature가 있는데 왜 출력은 4개이며, 0–1이면 확률이 아닌가?

Learned55와 direct9를 합친 `F_C64` 생성과, 여기서 세 count·occupancy를 예측하는 4-map head까지 구현·평가함. 64는 중간 표현의 channel 수이고 4는 감독 목표 수임. Feature 64개 각각에 GT를 지정하는 방식과 구분함.

Count channel의 0.25는 `0.25×16=4` label-groups의 연속 추정치이고, occupancy 0.25는 알려진 영역의 4분의 1임. Sigmoid가 0–1 범위를 보장해도 target 존재 확률 25%를 감독한 것은 아님.

현재 확인된 감독 효과는 count/occupancy 예측임. 다음에는 최종 fusion이 필요로 하는 관계 정보가 `F_C`에 얼마나 남는지 별도로 확인할 필요 있음.

### Q8. 다음 Step은 무엇인가?

**RGB-D만 받아 물체·관계 표현을 생성하는 전체 추론 경로를 구현·검증하는 것임.** GT 숫자 입력의 회귀는 이 경로에서 제외함. Phase 38의 접경 decoder는 GT 없는 부분 경로까지 구현됐지만 RGB 대비 일관된 개선을 보이지 못했음.

같은 물체의 내부 단차를 나누는 오류, 물체–배경을 물체–물체 접경으로 표시하는 오류, depth 변화가 작은 접경을 놓치는 오류를 구분하여 보완함. 기존 DINO는 유지하고 소속 표현·경계 보완·직접 관계 예측을 후보로 둠. 전체 segmentation을 필수 단계나 확정된 구조로 두지는 않음.

⑤의 48/96/160px feature 집계 비교, ⑥의 다중 뷰 teacher, 추가 encoder B/C는 미실행임. 구체적 실패를 개선하는 근거가 있을 때 비교하며 단일 RGB-D 추론을 유지함. 최종 Complexity GT·fusion·탐색 효용은 예측 관계 검증 이후에 연결함.

### Q9. 이 feature로 최종 2D-PDM을 만들 수 있는가?

세 stream의 feature 규격을 맞춰 `Concat(F_S,F_O,F_C): B×192×30×40`의 결합 입력까지 정리함. 다음은 fusion network·GT·loss·DRL 통합을 구현하고 탐색 효용을 평가하는 단계임. 이 결합과 최종 정책 실험은 **현재 미구현·미실행** 상태임.

행·열·channel 규격은 결합의 형식 조건을 제공함. 구조적 Complexity의 감독·평가를 정한 뒤, 각 stream을 어느 조건에서 활용하고 가려진 target 탐색에 어떻게 연결할지 학습·검증할 필요 있음. 세 map의 단순 합이나 밝기 기반 Complexity 확률을 확정한 단계는 아님.

정적 제거로 다른 물체가 보이는 효과와 실제 target 발견 효율을 구분해야 함. 최종 필요성은 동일 탐색 조건에서 **S+O 대비 S+O+C**의 추가 효용으로 검증함.

### Q10. Camera나 서랍이 달라져도 같은 숫자 기준을 사용하면 되는가?

원본 asset library와 고정 five-camera rig에서는 **다섯 camera 모두에서 density 예측 개선**을 확인함. Workspace·empty reference는 camera별 고정값이고 48/96/160px window는 영상 단위이므로, 다른 camera/FOV로 확장할 때는 reference 정합과 물리적 관측 범위를 추가 확인할 필요 있음.

Camera가 가까워지면 동일 물체의 pixel 크기가 커져 같은 96px window의 물리 면적이 달라짐. Camera 이동 후 기존 empty depth를 빼면 배경 정합 오차가 `Δ`에 남을 수 있음. Hash는 파일 동일성 검사이며 새 camera와 reference의 실제 정합 검사가 아님.

다음 rig에서는 15mm foreground threshold와 30/20mm 정규화가 sensor noise·물체 크기에 적합한지 평가해야 함. 현재 고정 rig 결과를 기준으로 실제 RGB-D sensor, 새로운 물체, camera/FOV, reference 오차에 대한 강건성을 확인할 단계임.

---

<!-- navigation:start -->
[전체 개요](README.md) · [Similarity](similarity_stream.md) · [Occlusion](occlusion_stream.md) · **Complexity** · [Development Log](development_log.md) · [연구 문맥](agent.md)
<!-- navigation:end -->
