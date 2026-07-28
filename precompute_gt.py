"""target별로 유사도 GT를 전체 scene 이미지에 대해 미리 렌더링해서 파일로 저장.

260714_data/GT_data/<target>/<scene 파일명과 동일>.png  (grayscale, distribution_map 관례와 동일:
uint8, 0=유사도 없음 .. 255=동일 물체)

CLIP 이미지 임베딩(gt_similarity_clip.py, image-image cosine) / CLIP 텍스트 임베딩(usd_name을
문장으로 바꿔 text-text cosine) 둘 다 시도했지만, 둘 다 우리가 원하는 "카테고리 유사성"과
상관관계가 없었다 (예: toy_4 target 기준 packaged_food가 toy보다 더 유사하다고 나오는 등
순서가 뒤죽박죽). 그래서 CLIP 블렌드를 포기하고 순수 카테고리 규칙
(gt_similarity.py의 SIMILARITY_MAP: 동일 카테고리=0.8, 유사=0.5, 나머지=0.2, target 자신=1.0,
260707_code/similarity_map_generator.py와 동일한 방식)만 사용한다."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from gt_similarity import build_color_to_score, discover_assets, load_scene_mapping, render_gt_map, resolve_target_from_data_folder
from paths_config import ASSET_DIR, GT_DIR, SCENE_DIR
from train_common import discover_scene_ids

# 260714_data에 데이터가 준비된 15개 target 전체.
# 참고: book_2는 target/mapping.json이 없어서 GT 계산은 되지만, 나중에 학습 시 target 사진을
# 못 불러와서 별도 조치가 필요함 (일단 GT부터 만들어둠).
TARGETS_TO_PROCESS = [
    "book_1", "book_2", "book_3", "book_4",
    "fruit_1", "fruit_2", "fruit_3", "fruit_4",
    "packaged_food_1", "packaged_food_2", "packaged_food_3", "packaged_food_4",
    "toy_1", "toy_2", "toy_3", "toy_4",
]

CAMS = ["center", "top", "left", "right", "bottom"]
ENV_RANGE = range(0, 300)   # 학습 시 subset을 쓰든 안 쓰든, 캐시 자체는 항상 전체를 만들어둠


def precompute_for_target(target_name: str, usd_to_category: dict):
    scene_dir = os.path.join(SCENE_DIR, target_name)
    gt_out_dir = os.path.join(GT_DIR, target_name)
    os.makedirs(gt_out_dir, exist_ok=True)

    target_usd_name, _ = resolve_target_from_data_folder(ASSET_DIR, target_name)
    scene_ids = discover_scene_ids(scene_dir)
    print(f"[{target_name}] target usd={target_usd_name}, {len(scene_ids)} scenes found")

    total_saved = 0
    t0 = time.time()
    for sid in scene_ids:
        prefix = f"scene{sid:05d}"
        mapping = load_scene_mapping(os.path.join(scene_dir, "seg", f"{prefix}_mapping.json"))
        color_to_score = build_color_to_score(mapping, target_usd_name, usd_to_category)

        # 이 scene에 실제로 존재하는 (env,cam) 파일 목록을 만듦
        fnames = []
        for env in ENV_RANGE:
            for cam in CAMS:
                fname = f"{prefix}_env{env:04d}_{cam}"
                if os.path.isfile(os.path.join(scene_dir, "seg", f"{fname}.png")):
                    fnames.append(fname)

        def render_and_save(fname):
            seg = cv2.imread(os.path.join(scene_dir, "seg", f"{fname}.png"))
            gt = render_gt_map(seg, color_to_score)
            cv2.imwrite(os.path.join(gt_out_dir, f"{fname}.png"), (gt * 255).astype(np.uint8))

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(render_and_save, fnames))
        total_saved += len(fnames)
        print(f"    {prefix}: {len(fnames)}장 저장 (누적 {total_saved})")

    print(f"[{target_name}] 완료 -- {total_saved}장을 {gt_out_dir} 에 저장, {time.time() - t0:.1f}s 소요")


def main():
    usd_to_category = discover_assets(ASSET_DIR)
    for target_name in TARGETS_TO_PROCESS:
        precompute_for_target(target_name, usd_to_category)


if __name__ == "__main__":
    main()
