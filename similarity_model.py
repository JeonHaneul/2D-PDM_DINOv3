"""Trainable head on top of frozen DINOv3 features: per-layer cosine-sim + raw-feature interaction,
a small matching conv per layer, multi-layer fusion, and an auxiliary map head producing the
final similarity probability map."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingBlock(nn.Module):
    """=== 여기가 CNN입니다 (MLP 아님) ===
    conv 3x3 -> GroupNorm -> ReLU -> conv 1x1 -> GroupNorm -> ReLU 로 이루어진 작은 합성곱 블록.
    입력/출력이 계속 (채널, H', W') 공간 형태를 유지한다 -- 만약 MLP(nn.Linear)를 썼다면
    입력을 1차원으로 flatten해야 해서 "이 위치가 scene의 어디인지"라는 공간 정보가 깨진다.
    지금 목표가 위치별로 다른 값을 갖는 "유사도 지도(map)"를 만드는 것이므로, 공간 구조를
    보존하는 CNN이 구조적으로 맞는 선택. Matching block은 DINO layer마다 하나씩 독립적으로 둠."""
    def __init__(self, in_ch: int, hidden_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimilarityMapModel(nn.Module):
    """
    scene_feats: list (one per DINO layer) of (patch (B,C,H',W'), cls (B,C)) from the scene encoder
    target_vecs: list (one per DINO layer) of masked-pooled target vectors (B,C), L2-normalized
    """

    def __init__(self, embed_dim: int, num_layers: int, hidden_ch: int = 64, category_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.category_dim = category_dim
        # interaction(=matching block에 들어갈 입력)의 채널 수 구성:
        #   scene patch feature (embed_dim) : "이 위치에 뭐가 있는지"에 대한 원본 정보
        #   target을 그대로 공간에 broadcast한 것 (embed_dim) : "우리가 찾는 게 뭔지"에 대한 원본 정보
        #   cosine similarity (1) : 위 둘을 직접 비교한 유사도 스칼라값 -- head에게 강한 힌트를 주는 채널
        #   category 확률 분포 broadcast (category_dim, 옵션) : "target이 book/toy/fruit/packaged_food
        #     중 뭘로 보이는가"를 DINOv3 CLS 토큰 프로토타입(train_common.py)으로 미리 계산해 넣어주는
        #     채널. cosine/raw feature만으로는 "형상은 다르지만 같은 카테고리"라는 관계를 순전히
        #     학습에만 의존해서 유추해야 하는데, 이 채널이 그 판단의 명시적 사전 정보를 줘서
        #     학습 때 못 본 카테고리 조합에 대해서도 head가 근거 있는 예측을 하도록 돕는다.
        # raw feature 두 개 + cosine 스칼라 하나를 다 같이 넣는 이유: cosine 값만 주면 head가
        # "형상이 비슷한가"라는 얕은 신호만 보게 되는데, raw feature까지 주면 head가 학습을 통해
        # "형상은 달라도 같은 카테고리다" 같은 더 깊은 관계도 배울 여지가 생김.
        interaction_ch = embed_dim + embed_dim + 1 + category_dim
        self.matching_blocks = nn.ModuleList(
            [MatchingBlock(interaction_ch, hidden_ch) for _ in range(num_layers)]
        )
        # 여러 DINO layer(예: 2,5,8,11번째 -> 저수준 형태 정보 ~ 고수준 의미 정보)의 결과를
        # 하나로 합치는 fusion 층. 1x1 conv라 위치별 채널 혼합만 하고 공간 크기는 그대로 유지.
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_ch * num_layers, hidden_ch, kernel_size=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
        )
        # 최종적으로 채널을 1개(=유사도 값 하나)로 압축하는 head. 이것도 1x1 conv (MLP 아님).
        self.aux_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)

    def forward(self, scene_feats, target_vecs, category_probs=None, out_size=None):
        if self.category_dim > 0 and category_probs is None:
            raise ValueError("category_dim > 0인 모델인데 category_probs가 안 넘어옴")
        # zip()은 길이가 안 맞으면 조용히 짧은 쪽에 맞춰버려서, 이 체크가 없으면 layer 개수가
        # 잘못 넘어와도 에러 없이 일부 layer만 쓰고 지나감 -- 미리 크게 실패하게 함.
        if len(scene_feats) != self.num_layers or len(target_vecs) != self.num_layers:
            raise ValueError(f"num_layers={self.num_layers}인데 scene_feats={len(scene_feats)}개, "
                              f"target_vecs={len(target_vecs)}개가 넘어옴")

        layer_outs = []
        # DINO layer마다 (scene patch, target vector) 쌍을 하나씩 순회하며 interaction을 만든다
        for (patch, _cls), target_vec in zip(scene_feats, target_vecs):
            B, C, Hp, Wp = patch.shape
            # cosine similarity 계산 준비: 두 벡터를 먼저 L2 정규화하면 내적(dot product)이
            # 곧 코사인 유사도가 됨 (cos = dot(a_hat, b_hat), a_hat=a/|a|, b_hat=b/|b|)
            patch_n = F.normalize(patch, dim=1)
            target_n = F.normalize(target_vec, dim=1)  # (B,C)
            # scene의 모든 patch 위치(Hp,Wp)에 대해 target과의 코사인 유사도를 한 번에 계산
            # (채널 축으로 내적 -> 결과는 위치마다 하나의 스칼라)
            cos = (patch_n * target_n.view(B, C, 1, 1)).sum(dim=1, keepdim=True)  # (B,1,Hp,Wp)
            cos = (cos + 1.0) / 2.0  # cosine 값 범위 [-1,1] -> [0,1]로 재조정 (GT와 같은 스케일로 맞춤)
            # target 벡터(원래는 위치 정보가 없는 단일 벡터)를 scene의 모든 위치에 똑같이 복제해서 붙임
            target_bcast = target_vec.view(B, C, 1, 1).expand(-1, -1, Hp, Wp)
            interaction_parts = [patch, target_bcast, cos]
            if self.category_dim > 0:
                # category_probs도 target_vec과 마찬가지로 위치 정보가 없는 (B,K) 벡터이므로
                # 동일한 방식으로 모든 위치에 복제해서 붙인다.
                cat_bcast = category_probs.view(B, self.category_dim, 1, 1).expand(-1, -1, Hp, Wp)
                interaction_parts.append(cat_bcast)
            # 채널 방향으로 이어붙임(concat) -> 이게 이번 layer의 interaction feature
            interaction = torch.cat(interaction_parts, dim=1)
            layer_outs.append(interaction)

        # layer별 interaction을 각자의 MatchingBlock(CNN)에 통과시킴
        matched = [block(x) for block, x in zip(self.matching_blocks, layer_outs)]
        # 모든 layer의 결과를 채널 방향으로 합친 뒤 fuse로 하나의 feature map으로 통합
        fused = self.fuse(torch.cat(matched, dim=1))
        logits = self.aux_head(fused)
        # sigmoid로 [0,1] 확률(유사도) 값으로 변환 -- patch 해상도(Hp,Wp) 그대로의 결과
        prob_patch_res = torch.sigmoid(logits)  # (B,1,Hp,Wp)

        prob_full_res = None
        if out_size is not None:
            # patch 해상도(예: 30x40)는 원본 이미지(예: 480x640)보다 훨씬 작으므로,
            # 시각화/최종 출력을 위해 bilinear 보간으로 원본 해상도까지 업샘플링
            prob_full_res = F.interpolate(prob_patch_res, size=out_size, mode="bilinear", align_corners=False)

        return {"prob_patch_res": prob_patch_res, "prob_full_res": prob_full_res, "fused": fused}
