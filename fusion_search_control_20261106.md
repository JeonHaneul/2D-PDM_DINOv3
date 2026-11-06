# S/SO/SOC 소규모 제거 비교 — 2026-11-06

<!-- navigation:start -->
[전체 개요](README.md) · [Fusion](fusion_stream.md) · [Development Log](development_log.md) · [연구 문맥](agent.md#current-handoff-20261106)
<!-- navigation:end -->

> 공개 갱신: 2026-11-06. 기존 실행을 설명하는 보고서이며 새 실험이 아님. 최신 상태는 연구 문맥의 같은 날짜 인계를 우선함. 제거 우선순위 head와 새 행동 GT는 미실행 제안임.

사용자의 빠른 후속 확인 요청으로 기존32개 선정배치에서 책·과일·포장식품·장난감의 첫배치를 결과와 무관하게 고정함. book_1_scene00013_env0051, fruit_1_scene00003_env0046, packaged_food_1_scene00027_env0008, toy_1_scene00019_env0033의4물리장면×5독립camera×3head=60회 완료함. 한 모델의 입력은 해당camera의 단일RGB-D임.

| 모델 | 성공 | 평균 방해물 제거 | 중앙값 | 처음부터target선택 |
|---|---:|---:|---:|---:|
| S 특징 + 최종GT head(best18) | 20/20 | 2.35회 | 2회 | 8/20 |
| S+O 특징 + 최종GT head(best20) | 20/20 | 2.45회 | 2회 | 8/20 |
| S+O+C 특징 + 최종GT head(best10) | 20/20 | 2.25회 | 2회 | 8/20 |

평균은 성공조건부이며 이번에는60회 모두 성공함. Target회수1행동은 제거횟수에서 제외함. 처음target을 선택하면0회임. SOC는S보다20조건 중3개에서 적게 제거하고16개에서 같고1개에서 더 많이 제거함. 평균차0.10회(총2개), 상대감소4.26%지만 작은표본의기술통계이며 통계적우월성 주장으로 쓰지 않음. SOC는SO보다4개에서적고16개에서같으며 평균0.20회차임.

| Target 장면 | S | SO | SOC |
|---|---:|---:|---:|
| book_1 | 3.8 | 4.2 | 3.6 |
| fruit_1 | 5.2 | 5.2 | 5.0 |
| packaged_food_1 | 0 | 0 | 0 |
| toy_1 | 0.4 | 0.4 | 0.4 |

각행은 같은물리배치의5camera 평균임. 포장식품5뷰와장난감3뷰는세모델모두초기성공이어서제거차이를보일여지가없었음. 이장면을 결과에따라제외/교체하지않음. 20조건이20독립물리장면인것은아니며 전체16target 성능이나외부물체성능으로확대하지않음.

## 판단

동일최종GT를학습한S단독head도이번제거과제에서SOC에가까운결과를보임. 앞선test24,000장의amodal최고점S70.88%/SOC85.06%차이가 이번소규모제거횟수에서는큰차이로나타나지않았음. 가려진target위치적중과50%이상보이는물체를실제로선택해제거하는효용은다른지표임. 현재결과만으로추가stream이무용하거나CNN이이를무시한다고확정하지않음. 반대로기존standaloneSimilarity와Fusion의45.90%제거차이를추가stream만의효과로설명하지않음. 그이전S는최종GT head가아닌원래similarity지도정책이었음.

이번확인은CNN/attention교체나GT균등혼합채택의근거가아님. 각stream의예측의미를직접결합하는구조를비교하려면예측1채널지도3장만입력하는fusion을별도대조할수있으나, 이 소규모 실행 당시에는 미실행이었으며 이후 지도 비교를 완료함. SO를 포함한 전체32배치의 S/SO/SOC 비교는 미실행임. 특징 S/SOC와 지도 S/SOC의 네 head를 사용한 전체32배치 비교는 이후 완료했으며 fusion_map_control_20261106.md를 따름.

## 실행·검산

기존시뮬레이터의정책이름과환경수20→15만조정하고원래reset순서·실제floor검사·GPU pair capacity1048576·PhysX오류중단·가시율50%·제거후600actualphysicssteps(5초)를유지함. 현재세head를동일실행에서다시평가해renderer/physics조건을맞춤. 기존SOC결과를현재결과로대체사용하지않음. 매camera초기RGB는사전지정SOC환경의것을공유하고depth/physicallabels exact일치를확인함. 원래개별RGB도보존함. 이후행동이달라지면각각독립물리상태로진행함.

관측201회·가시율후보검사340회·방해물제거141회,60episode모두완료함. raw실행129.07초(모델초기화/시뮬레이션/종료검사포함),최종CPU검산·그림생성시간은별도임. 매step rawprediction으로순위재계산·native가시/단독mask면적·50%조건·처음적격후보선택·종료·600physicsstep을검산함. 기존55source·3head체크포인트와frozen parameter/buffer 유지,PhysX오류0임. 기존저장test한장의세예측과최대차S4.92e-6/SO2.06e-6/SOC1.37e-6으로원래허용오차를통과함.

코드: `experiments/fusion_search_control_20261106/`. 조건: output의`selection.json`·`protocol.json`,추론검사`runtime_check/equivalence.json`. 완료:`full/{raw_completion,completion,audit,physics_log_audit}.json`. 수치:`full/report/{summary.json,episodes.csv,paired_scene_camera.csv,index.html}`. 모든step의RGB-D/ID/pose/실제rawprediction/가시율/선택물체·제거기록과201비교그림·60trajectory를`full/episodes/`에보존함. 모델입력에는sceneGT를전달하지않으며환경ID/가시율은시뮬레이터행동연결·가능조건에만사용함. 실제로봇grasp성공률이아님.

이 탐색 실행 당시 새학습·GT/기존데이터수정·GitHub게시는 없었음. 이번 사용자 요청으로 보고서와 수치를 공개함. 이전32장면4정책결과는그대로보존함.


<!-- navigation:start -->
[전체 개요](README.md) · [Fusion](fusion_stream.md) · [Development Log](development_log.md) · [연구 문맥](agent.md#current-handoff-20261106)
<!-- navigation:end -->
