"""inference_zeroshot.py — 학습된 v2(SigLIP) 체크포인트로, 학습 때 한 번도 안 본 target에
대해 zero-shot 유사도맵을 뽑아보는 검증 스크립트.

train_similarity_v2.py와 다른 점:
  - GT가 전혀 필요 없다 (정답을 비교하는 게 아니라 예측만 뽑아서 눈으로 확인하는 용도)
  - target_dir/scene_image가 260714_data 구조 밖의 임의 경로여도 된다 -- target_capture.py로
    새로 촬영한, 학습 TARGETS 리스트에 아예 없는 물체를 그대로 넣어서 진짜 zero-shot을 테스트한다.

target_dir는 target_capture.py 출력과 동일한 구조를 기대한다: <target_dir>/rgb/, /seg/, /mapping.json

사용 예:
    python inference_zeroshot.py \\
        --checkpoint outputs/multi_target_xxx_siglip/similarity_head_best.pt \\
        --target_dir /home/haneul/isaacsim/src/scene_generator/output/fruit_5/target \\
        --scene_image /home/haneul/isaacsim/src/scene_generator/output/fruit_5/scene/rgb/scene00001_env0000_center.png \\
        --label banana \\
        --out zeroshot_result.png
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, SiglipModel

from backbone import DINOv3Backbone, PATCH_SIZE
from similarity_model import SimilarityMapModel
from target_utils import bgr_to_tensor, crop_with_mask, discover_target_frame_id, load_target_reference
from train_common import encode_target as encode_target_appearance
from train_similarity_v2 import (
    LAYERS, SIGLIP_DIM, SIGLIP_MODEL_ID, SemanticProjection, encode_target_semantic, encode_text_siglip,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def label_to_prompt(label: str) -> str:
    """자유 라벨(단어/짧은 구) -> 항상 같은 짧은 템플릿으로 감싸기 (학습 때와 동일한 원칙:
    길고 장황한 문장이 아니라 일관된 짧은 형태로 정규화)."""
    return f"a photo of a {label}"


@torch.no_grad()
def encode_target(backbone, siglip_model, siglip_tokenizer, target_dir: str, cam: str, label: str | None):
    """target_dir(=target_capture.py 출력 폴더)에서 target 하나를 appearance+semantic 융합 벡터로 인코딩."""
    target_rgb_dir = os.path.join(target_dir, "rgb")
    target_seg_dir = os.path.join(target_dir, "seg")
    target_mapping_path = os.path.join(target_dir, "mapping.json")
    frame_id = discover_target_frame_id(target_rgb_dir)

    appearance = encode_target_appearance(backbone, DEVICE, target_rgb_dir, target_seg_dir,
                                          target_mapping_path, frame_id, cam)

    text_embed = None
    if label is not None:
        text_embed = encode_text_siglip(siglip_model, siglip_tokenizer, label_to_prompt(label), DEVICE)
    semantic_raw = encode_target_semantic(siglip_model, DEVICE, target_rgb_dir, target_seg_dir,
                                          target_mapping_path, frame_id, cam, text_embed=text_embed)

    # target crop도 같이 반환 -- 결과 패널에 같이 보여주기 위함
    tgt_rgb, tgt_mask, _ = load_target_reference(target_rgb_dir, target_seg_dir, target_mapping_path, frame_id, cam)
    crop_rgb, _crop_mask, _ = crop_with_mask(tgt_rgb, tgt_mask, pad_ratio=0.25)

    return appearance, semantic_raw, crop_rgb


def build_panel(crop_rgb, scene_rgb, pred_full, overlay_alpha: float = 0.5) -> np.ndarray:
    """target crop | scene | 예측 heatmap overlay 3-패널 시각화."""
    H, W = scene_rgb.shape[:2]

    def label_bar(img, text, bar_h=32):
        bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (8, bar_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([bar, img], axis=0)

    target_vis = cv2.resize(crop_rgb, (H, H))

    heat_u8 = (np.clip(pred_full, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(scene_rgb, 1 - overlay_alpha, heat_color, overlay_alpha, 0)

    gray_bgr = cv2.cvtColor(heat_u8, cv2.COLOR_GRAY2BGR)

    return np.concatenate([
        label_bar(target_vis, "TARGET (zero-shot, unseen)"),
        label_bar(scene_rgb, "SCENE"),
        label_bar(gray_bgr, "PRED (grayscale)"),
        label_bar(overlay, "PRED (heatmap overlay)"),
    ], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot 유사도맵 검증 (v2/SigLIP 체크포인트 사용)")
    parser.add_argument("--checkpoint", required=True, help="similarity_head_best.pt 경로")
    parser.add_argument("--target_dir", required=True, help="target_capture.py 출력 형식 폴더 (rgb/seg/mapping.json)")
    parser.add_argument("--target_cam", default="center", choices=["center", "top", "left", "right", "bottom"])
    parser.add_argument("--scene_image", required=True, help="테스트할 scene RGB 이미지 경로")
    parser.add_argument("--label", default=None, help="선택: 텍스트 힌트 (예: 'banana'). 안 주면 image-only")
    parser.add_argument("--out", default="zeroshot_result.png")
    args = parser.parse_args()

    print(f"loading DINOv3 backbone (vitb16, layers={LAYERS}, device={DEVICE})")
    backbone = DINOv3Backbone(variant="vitb16", layers=LAYERS, device=DEVICE)

    print(f"loading SigLIP so400m ({SIGLIP_MODEL_ID}, frozen) ...")
    siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model.eval()
    for p in siglip_model.parameters():
        p.requires_grad_(False)
    siglip_model.to(DEVICE)
    siglip_tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)

    print(f"loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model = SimilarityMapModel(embed_dim=backbone.embed_dim, num_layers=len(LAYERS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    semantic_proj = SemanticProjection(SIGLIP_DIM, backbone.embed_dim, len(LAYERS)).to(DEVICE)
    semantic_proj.load_state_dict(ckpt["semantic_proj_state"])
    semantic_proj.eval()

    print(f"encoding target: {args.target_dir} (cam={args.target_cam}, label={args.label!r})")
    appearance, semantic_raw, crop_rgb = encode_target(
        backbone, siglip_model, siglip_tokenizer, args.target_dir, args.target_cam, args.label
    )

    scene_rgb = cv2.imread(args.scene_image)
    if scene_rgb is None:
        raise FileNotFoundError(f"scene 이미지를 읽을 수 없음: {args.scene_image}")
    H, W = scene_rgb.shape[:2]

    with torch.no_grad():
        sem_proj = semantic_proj(semantic_raw.unsqueeze(0))
        target_vecs = [a.unsqueeze(0) + s for a, s in zip(appearance, sem_proj)]
        scene_feats = backbone(bgr_to_tensor(scene_rgb, device=DEVICE))
        out = model(scene_feats, target_vecs, out_size=(H, W))
    pred_full = out["prob_full_res"][0, 0].cpu().numpy()

    panel = build_panel(crop_rgb, scene_rgb, pred_full)
    cv2.imwrite(args.out, panel)
    print(f"저장 완료: {args.out}")
    print(f"예측 유사도 범위: min={pred_full.min():.3f} max={pred_full.max():.3f} mean={pred_full.mean():.3f}")


if __name__ == "__main__":
    main()
