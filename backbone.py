"""Frozen DINOv3 backbone wrapper — multi-layer patch grid + CLS token extraction."""
import os
from typing import Sequence

import torch
import torch.nn as nn

from paths_config import MODEL_DIR

# variant 이름 -> (torch.hub 진입점 함수명, 로컬 가중치 파일명, 임베딩 차원(C), 레이어(block) 수)
# DINOv3는 이 6개 크기로 사전학습된 체크포인트를 공개했고, 전부 로컬에 받아둔 상태.
# 지금은 vitb16(768차원, 12층)을 기본으로 쓰고 있고, 나중에 variant 문자열만 바꾸면
# 더 가벼운(vits16) 또는 더 큰(vitl16, vith16plus, vit7b16) 모델로 교체 가능.
VARIANTS = {
    "vits16":      ("dinov3_vits16",      "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",      384,  12),
    "vits16plus":  ("dinov3_vits16plus",  "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",  384,  12),
    "vitb16":      ("dinov3_vitb16",      "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",      768,  12),
    "vitl16":      ("dinov3_vitl16",      "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",      1024, 24),
    "vith16plus":  ("dinov3_vith16plus",  "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",  1280, 32),
    "vit7b16":     ("dinov3_vit7b16",     "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",     4096, 40),
}

# ViT가 이미지를 몇 픽셀 단위 패치로 쪼개는지 (16x16). 이 값 때문에 입력 이미지의
# H, W는 항상 16의 배수여야 patch grid가 딱 떨어짐 (예: 640x480 -> 40x30 grid).
PATCH_SIZE = 16


class DINOv3Backbone(nn.Module):
    """Frozen (eval-only) DINOv3. forward() returns per requested layer:
    (patch_grid: (B,C,H',W'), cls_token: (B,C))
    """

    def __init__(self, variant: str = "vitb16", layers: Sequence[int] = (2, 5, 8, 11), device: str = "cuda"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}, choose from {list(VARIANTS)}")
        hub_name, weight_file, embed_dim, num_layers = VARIANTS[variant]
        weight_path = os.path.join(MODEL_DIR, weight_file)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(weight_path)
        for l in layers:
            if l >= num_layers:
                raise ValueError(f"layer {l} out of range for {variant} (depth={num_layers})")

        # torch.hub.load: 실제 가중치(숫자)는 이미 로컬 .pth로 갖고 있으므로 재다운로드하지 않고,
        # 그 가중치를 어떤 모양의 신경망에 채워 넣어야 하는지에 대한 "모델 설계도 코드"만
        # facebookresearch/dinov3 공식 레포에서 받아온다 (weights=로컬경로로 지정했기 때문).
        # 이 설계도 코드는 최초 1회만 다운로드되고 이후엔 ~/.cache/torch/hub에 캐시되어 재사용됨.
        self.model = torch.hub.load("facebookresearch/dinov3", hub_name, weights=weight_path, trust_repo=True)
        self.model.eval()
        # DINOv3는 학습 대상이 아니라 "고정된 특징 추출기"로만 사용 -- 모든 파라미터의
        # gradient 계산을 꺼서 backprop이 여기까지 흘러들어오지 않게 만든다 (frozen backbone).
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.variant = variant
        self.layers = list(layers)
        self.embed_dim = embed_dim
        self.patch_size = PATCH_SIZE
        self.device = device
        self.to(device)

    @torch.no_grad()  # frozen backbone이므로 여기서 계산 그래프(gradient 기록)를 만들 필요가 없음 -> 메모리/속도 이득
    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        # get_intermediate_layers: DINOv3가 제공하는 표준 API.
        #   n=self.layers        -> 몇 번째 block(층)들의 출력을 뽑을지 (예: 2,5,8,11번째, multi-layer 사용)
        #   reshape=True         -> 패치 토큰들을 1차원 시퀀스가 아니라 (B,C,H',W') 공간 형태 그리드로 복원
        #   return_class_token=True -> 패치 토큰들과 별개로, 이미지 전체를 요약하는 CLS 토큰도 같이 반환
        #   norm=True            -> 각 층 출력에 최종 LayerNorm까지 적용된 값을 반환 (그대로 쓰기 좋은 상태)
        outs = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=True, return_class_token=True, norm=True
        )
        return list(outs)  # [(patch, cls), ...] one per layer, in the order of self.layers
