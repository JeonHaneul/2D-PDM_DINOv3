"""Category-based similarity-map ground truth, adapted from 260707_code/similarity_map_generator.py
to the current asset/data layout (asset/<Category>/<instance>/*.usd, no manual pairwise table needed
beyond the 4 macro-categories)."""
import json
import os

import numpy as np

SIMILARITY_MAP = {
    "book":          {"book": 0.8, "toy": 0.5, "fruit": 0.2, "packaged_food": 0.2},
    "toy":           {"book": 0.5, "toy": 0.8, "fruit": 0.2, "packaged_food": 0.2},
    "fruit":         {"book": 0.2, "toy": 0.2, "fruit": 0.8, "packaged_food": 0.5},
    "packaged_food": {"book": 0.2, "toy": 0.2, "fruit": 0.5, "packaged_food": 0.8},
}


def discover_assets(asset_dir: str) -> dict:
    """usd_name -> lowercased macro-category, from asset/<Category>/<instance>/*.usd(c)."""
    usd_to_category = {}
    for category in sorted(os.listdir(asset_dir)):
        cat_dir = os.path.join(asset_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for subdir in sorted(os.listdir(cat_dir)):
            subdir_path = os.path.join(cat_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().endswith((".usd", ".usdc")):
                    usd_name = os.path.splitext(f)[0]
                    usd_to_category[usd_name] = category.lower()
                    break
    return usd_to_category


def discover_asset_instances(asset_dir: str) -> dict:
    """instance folder name (lowercased, e.g. 'toy_4') -> (usd_name, category), from
    asset/<Category>/<instance>/*.usd(c). Used to resolve which physical object a data
    folder like `260708_data/toy_4/` refers to -- the folder name IS the target selection
    made when the scene data was generated (same convention as 260707_code's output/<target_name>/)."""
    instance_to_info = {}
    for category in sorted(os.listdir(asset_dir)):
        cat_dir = os.path.join(asset_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for subdir in sorted(os.listdir(cat_dir)):
            subdir_path = os.path.join(cat_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().endswith((".usd", ".usdc")):
                    usd_name = os.path.splitext(f)[0]
                    instance_to_info[subdir.lower()] = (usd_name, category.lower())
                    break
    return instance_to_info


def resolve_target_from_data_folder(asset_dir: str, data_folder_name: str) -> tuple:
    """e.g. data_folder_name='toy_4' -> ('RubixCube', 'toy'). Raises if the data folder name
    doesn't match any known asset instance -- forces an explicit, traceable target choice
    instead of a silently hardcoded usd name."""
    instance_to_info = discover_asset_instances(asset_dir)
    key = data_folder_name.lower()
    if key not in instance_to_info:
        raise ValueError(
            f"data folder '{data_folder_name}' doesn't match any asset instance under {asset_dir}. "
            f"known instances: {sorted(instance_to_info)}"
        )
    return instance_to_info[key]


def load_scene_mapping(mapping_json_path: str) -> dict:
    with open(mapping_json_path) as f:
        m = json.load(f)
    return {k: tuple(v["color_rgb"]) for k, v in m.items()}  # actually BGR


def build_color_to_score(scene_mapping: dict, target_usd_name: str, usd_to_category: dict) -> dict:
    target_category = usd_to_category.get(target_usd_name, "")
    target_scores = SIMILARITY_MAP.get(target_category, {})
    color_to_score = {}
    for usd_name, bgr in scene_mapping.items():
        if usd_name == target_usd_name:
            color_to_score[bgr] = 1.0
        else:
            cat = usd_to_category.get(usd_name, "")
            color_to_score[bgr] = target_scores.get(cat, 0.0)
    return color_to_score


def render_gt_map(seg_bgr: np.ndarray, color_to_score: dict) -> np.ndarray:
    h, w = seg_bgr.shape[:2]
    gt = np.zeros((h, w), dtype=np.float32)
    for (b, g, r), score in color_to_score.items():
        mask = (seg_bgr[:, :, 0] == b) & (seg_bgr[:, :, 1] == g) & (seg_bgr[:, :, 2] == r)
        gt[mask] = score
    return gt
