"""category별 train/held-out target을 GT 생성 전에 미리 고정한다 -- 결과를 보고 유리하게
hold-out을 고르는 post-hoc cherry-picking을 막기 위해, 이 파일을 커밋한 시점 이후로는
바꾸지 않는다. unseen-instance/seen-category LOTO에 쓸 목적.

toy_1은 asset이 아니라 mesh loader의 composed-stage scale 재현 문제로 제외됨(2026-08-11,
실측 깊이로 확인: 실제 크기는 정상이지만 standalone mesh extraction이 1000배 작게 나옴 --
production 대상에서 제외, 원인은 별도 후속 과제로 남김).
toy_4는 GT pilot 검증(mesh-vs-real depth/seg 대조)에 이미 쓰였지만 학습에는 아직 쓰이지
않았으므로 held-out target으로 지정 가능 -- 이 사실을 기록해둔다."""
import os

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")

# (usd 상대경로,) -- base_z는 generate_occlusion_gt_batched_v2.BASE_Z_TABLE에서 조회.
TARGET_USD = {
    "book_1": "asset/Book/book_1/Book_02.usd",
    "book_2": "asset/Book/book_2/Book_GetKnowPPU.usd",
    "book_3": "asset/Book/book_3/Book_Greener.usd",
    "book_4": "asset/Book/book_4/OmniConnect2015.usd",
    "fruit_1": "asset/Fruit/fruit_1/Apple.usd",
    "fruit_2": "asset/Fruit/fruit_2/Avocado01.usd",
    "fruit_3": "asset/Fruit/fruit_3/Lime01.usd",
    "fruit_4": "asset/Fruit/fruit_4/Orange_03.usd",
    "toy_2": "asset/Toy/toy_2/Ball_Walnut.usd",
    "toy_3": "asset/Toy/toy_3/Shield_Controller.usd",
    "toy_4": "asset/Toy/toy_4/RubixCube.usd",
    "packaged_food_2": "asset/Packaged_food/packaged_food_2/010_potted_meat_can.usd",
    "packaged_food_3": "asset/Packaged_food/packaged_food_3/006_mustard_bottle.usd",
    "packaged_food_4": "asset/Packaged_food/packaged_food_4/008_pudding_box.usd",
    # toy_1: 제외(사유는 위 docstring). packaged_food_1: clutter 소스 전용, target으로 안 씀.
    # packaged_food_5: 260714_data에 target 참조 사진 없어서 제외.
}

CATEGORY_OF = {
    "book_1": "book", "book_2": "book", "book_3": "book", "book_4": "book",
    "fruit_1": "fruit", "fruit_2": "fruit", "fruit_3": "fruit", "fruit_4": "fruit",
    "toy_2": "toy", "toy_3": "toy", "toy_4": "toy",
    "packaged_food_2": "packaged_food", "packaged_food_3": "packaged_food", "packaged_food_4": "packaged_food",
}

# category당 1개씩 고정 hold-out -- unseen-instance/seen-category 검증용. 이 지정 이후 변경 금지.
HELD_OUT_TARGETS = ["book_4", "fruit_4", "toy_4", "packaged_food_4"]
TRAIN_TARGETS = [t for t in TARGET_USD if t not in HELD_OUT_TARGETS]

ALL_TARGETS = list(TARGET_USD.keys())


def usd_path(target: str) -> str:
    return os.path.join(SRC_DIR, TARGET_USD[target])
