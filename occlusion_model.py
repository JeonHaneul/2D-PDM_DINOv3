"""Occlusion map 모델 (2D PDM_ver2.pdf의 Occlusion map 스트림 설계를 반영, 실험 논의 결과 반영).

PDF 원안과 다른 점 (대화에서 논의된 이유):
  - target depth 인코더(ResNet18)를 없앴다. 실제 배포 시 target은 RGB 사진(+텍스트)만 받을 수
    있고 depth 사진은 못 받는다는 제약 때문 -- depth를 별도로 예측/hallucination하는 것도
    검토했지만, 학습 데이터가 target 15개뿐이라 새 네트워크가 안정적으로 일반화될지 불확실해서
    포기했다.
  - target appearance(DINO feature)는 crop/mask 없이 원본 프레임 그대로 average pooling한
    벡터를 쓴다(마스크 기반 산수는 zero-shot 새 물체에서 부정확한 마스크로 깨지기 쉬워서 피함).
    다만 이건 "matching interaction"에만 쓰고(아래), 크기/형태 조건은 별도로
    target_utils.extract_target_geometry()의 결정적(mask 기반, 학습 불필요) 벡터로 명시적으로
    넣는다 -- depth feature에 FiLM으로 적용(target_geometry_condition, 첫 smoke test 이후
    크리틱으로 추가). "물체가 클수록 평균 벡터가 배경에서 멀어진다"는 암묵적 크기 신호에만
    의존하던 이전 설계는 실제로 검증되지 않았고, 명시적 geometry 벡터가 있는데 안 쓸 이유가 없음.
  - output에 target-scene cosine similarity를 직접 더하는 shortcut(초기 버전에 있었음, 첫
    smoke test 이후 제거)을 없앴다 -- cosine은 "이 patch가 시각적으로 target과 닮았는가"이지
    "여기서 target이 가려졌는가"가 아니라서, logits에 직접 더하면 모델이 depth 기반 occlusion
    추론을 우회하고 appearance 유사도만으로 답을 낼 위험이 있음(어느 쪽이 기여했는지도 해석
    불가능해짐). cosine 자체는 matching interaction의 한 채널로는 남겨둠(위치별 대응 단서로는
    여전히 유용, 다만 output에 직접 더하지는 않음).

MatchingBlock은 similarity_model.py 것을 그대로 재사용(중복 정의 방지)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from similarity_model import MatchingBlock
from target_utils import bgr_to_tensor, extract_target_geometry, load_target_reference

GEOMETRY_DIM = 4 + 8 * 8  # extract_target_geometry() 기본 silhouette_size=8과 맞춤


class SceneDepthEncoder(nn.Module):
    """Scene depth 이미지(depth_norm + valid_mask, 2채널)를 ResNet18로 인코딩해서 DINOv3와
    같은 형식(레이어별 patch grid)으로 반환. ImageNet 사전학습 가중치는 3채널 RGB 기준이라
    depth엔 안 맞으므로, 처음부터 학습(from scratch)한다 -- depth map은 RGB보다 통계가 훨씬
    단순해서(질감/색 없음, 형태 위주) 굳이 큰 사전학습 없이도 이 정도 태스크에서 학습 가능하다고
    판단. valid_mask를 depth와 별도 채널로 넣는 이유: depth 값 0(레이더 무반사/측정 불가)과
    "실제로 깊이가 0인 표면"이 depth 채널 하나만으로는 구분이 안 돼서, 모델이 유효하지 않은
    픽셀을 depth=0으로 오인하지 않도록 명시적으로 알려준다.
    layer1~4 중 3개(공간 해상도가 크게 차이 나는 지점)를 뽑아 DINOv3 patch grid 해상도에
    맞춰 보간(resize)한다."""

    def __init__(self, out_channels: int = 256):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # depth_norm+valid_mask 2채널 입력
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # stride 4,  64ch
        self.layer2 = resnet.layer2  # stride 8,  128ch
        self.layer3 = resnet.layer3  # stride 16, 256ch (DINOv3 patch_size=16과 해상도가 같아 기준으로 삼음)
        self.layer4 = resnet.layer4  # stride 32, 512ch
        # 서로 다른 채널 수를 동일한 out_channels로 맞추는 1x1 projection (레이어별 독립)
        self.proj2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, depth: torch.Tensor, target_hw: tuple) -> list:
        """depth: (B,1,H,W). target_hw: (Hp,Wp) -- DINOv3 patch grid 해상도에 맞춰 보간.
        반환: [(patch (B,out_channels,Hp,Wp), cls (B,out_channels)), ...] x 3 (layer2,3,4 순)
        -- SimilarityMapModel의 scene_feats와 동일한 (patch, cls) 튜플 리스트 형식."""
        x = self.stem(depth)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        outs = []
        for feat, proj in ((x2, self.proj2), (x3, self.proj3), (x4, self.proj4)):
            f = proj(feat)
            f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            cls = f.mean(dim=(2, 3))  # 전역 평균 -- DINOv3의 cls token 자리를 대신하는 요약 벡터
            outs.append((f, cls))
        return outs


@torch.no_grad()
def encode_target_occlusion_frame(backbone, device: str, target_rgb_dir: str, target_seg_dir: str,
                                   target_mapping_path: str, target_frame_id: str, cam: str) -> tuple:
    """target을 crop/mask 없이 원본 프레임 그대로 DINOv3에 넣고, 레이어마다 **일반**(마스크 없는)
    average pooling으로 appearance 벡터 하나씩 뽑는다(matching interaction용). 동시에 같은
    프레임의 mask에서 geometry 벡터(FiLM 조건용, extract_target_geometry)도 뽑아서 같이 반환
    -- 어차피 같은 load_target_reference 호출 결과에서 나오므로 따로 다시 읽지 않음.
    반환: ([(C,), ...] per layer, geometry (GEOMETRY_DIM,) np.ndarray)"""
    tgt_rgb, tgt_mask, _ = load_target_reference(target_rgb_dir, target_seg_dir, target_mapping_path,
                                                  target_frame_id, cam)
    tgt_tensor = bgr_to_tensor(tgt_rgb, device=device)
    target_feats = backbone(tgt_tensor)  # [(patch (1,C,Hp,Wp), cls), ...] per layer
    target_vecs = [patch[0].mean(dim=(1, 2)) for patch, _cls in target_feats]  # [(C,), ...] per layer
    geometry = extract_target_geometry(tgt_mask)
    return target_vecs, geometry


class GeometryFiLM(nn.Module):
    """target_geometry 벡터(크기/aspect/실루엣, mask에서 결정적으로 계산됨) -> 레이어별
    depth feature에 적용할 FiLM (gamma, beta). "target 크기/형태에 따라 depth 공간 해석을
    바꾼다"는 설계 의도를 명시적인 conditioning 경로로 구현 -- 이전엔 이 경로 자체가 없었음.
    gamma init=1, beta init=0(항등 변환에서 시작)으로 초기화해서 학습 초반에 depth feature를
    갑자기 왜곡하지 않도록 함."""

    def __init__(self, geom_dim: int, depth_ch: int, num_layers: int, hidden: int = 64):
        super().__init__()
        self.num_layers = num_layers
        self.depth_ch = depth_ch
        self.net = nn.Sequential(nn.Linear(geom_dim, hidden), nn.ReLU(inplace=True))
        self.out = nn.Linear(hidden, num_layers * depth_ch * 2)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            bias = self.out.bias.view(num_layers, 2, depth_ch)
            bias[:, 0, :] = 1.0  # gamma
            bias[:, 1, :] = 0.0  # beta

    def forward(self, geometry: torch.Tensor) -> list:
        """geometry: (B,geom_dim) -> [(gamma (B,C,1,1), beta (B,C,1,1)), ...] per layer."""
        out = self.out(self.net(geometry)).view(-1, self.num_layers, 2, self.depth_ch)
        return [(out[:, li, 0].unsqueeze(-1).unsqueeze(-1), out[:, li, 1].unsqueeze(-1).unsqueeze(-1))
                for li in range(self.num_layers)]


class OcclusionMapModel(nn.Module):
    """
    scene_rgb_feats : DINOv3Backbone(scene_rgb)의 출력 -- [(patch (B,C,Hp,Wp), cls), ...] per layer
    scene_depth     : (B,2,H,W) scene depth 텐서(채널 0=고정범위 정규화 depth, 채널 1=valid mask,
                      SceneDepthEncoder에 그대로 넣음)
    target_vecs     : encode_target_occlusion_frame 출력의 첫 번째 원소 -- [(C,), ...] per layer
                      (배치 시 (B,C)) -- matching interaction에만 쓰임(appearance 단서)
    target_geometry : encode_target_occlusion_frame 출력의 두 번째 원소를 배치로 쌓은 것
                      (B,GEOMETRY_DIM) -- GeometryFiLM을 통해 depth feature에만 적용(크기/형태
                      조건). appearance와 geometry를 분리한 이유는 위 모듈 docstring 참고.

    구조는 SimilarityMapModel과 동일한 패턴(레이어별 MatchingBlock -> fuse -> aux_head)이되,
    interaction에 scene depth feature(FiLM 적용됨)까지 한 채널 그룹 더 들어간다는 점만 다르다.
    """

    def __init__(self, dino_embed_dim: int, num_layers: int, depth_ch: int = 256, hidden_ch: int = 64,
                 geometry_dim: int = GEOMETRY_DIM):
        super().__init__()
        self.num_layers = num_layers
        self.depth_encoder = SceneDepthEncoder(out_channels=depth_ch)
        self.geometry_film = GeometryFiLM(geometry_dim, depth_ch, num_layers)
        # interaction 채널 구성: scene_rgb(dino_embed_dim) + scene_depth(depth_ch)
        #   + target_broadcast(dino_embed_dim) + cosine(1)
        interaction_ch = dino_embed_dim + depth_ch + dino_embed_dim + 1
        self.matching_blocks = nn.ModuleList(
            [MatchingBlock(interaction_ch, hidden_ch) for _ in range(num_layers)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_ch * num_layers, hidden_ch, kernel_size=1),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.aux_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        # 주의: output에 cosine을 직접 더하는 shortcut은 의도적으로 없음(모듈 docstring 참고).
        # target 신호는 matching_blocks의 interaction(=학습되는 conv)을 통해서만 logits에 영향을
        # 준다 -- appearance shortcut으로 depth 기반 추론을 우회할 경로를 원천적으로 막기 위함.

    def forward(self, scene_rgb_feats: list, scene_depth: torch.Tensor, target_vecs: list,
                target_geometry: torch.Tensor, out_size: tuple = None) -> dict:
        layer_outs = []
        depth_feats = None  # scene_depth의 (Hp,Wp)는 레이어마다 다를 수 있어 매번 새로 보간
        film_params = self.geometry_film(target_geometry)  # [(gamma,beta), ...] per layer

        for li, ((patch, _cls), target_vec) in enumerate(zip(scene_rgb_feats, target_vecs)):
            B, C, Hp, Wp = patch.shape

            patch_n = F.normalize(patch, dim=1)
            target_n = F.normalize(target_vec, dim=1)
            cos = (patch_n * target_n.view(B, C, 1, 1)).sum(dim=1, keepdim=True)
            cos = (cos + 1.0) / 2.0  # matching interaction 채널로만 사용(output에 직접 더하지 않음)

            target_bcast = target_vec.view(B, C, 1, 1).expand(-1, -1, Hp, Wp)

            if depth_feats is None:
                depth_feats = self.depth_encoder(scene_depth, target_hw=(Hp, Wp))
            depth_patch, _depth_cls = depth_feats[min(li, len(depth_feats) - 1)]
            gamma, beta = film_params[min(li, len(film_params) - 1)]
            depth_patch = gamma * depth_patch + beta  # target geometry 조건부 변조

            interaction = torch.cat([patch, depth_patch, target_bcast, cos], dim=1)
            layer_outs.append(interaction)

        matched = [block(x) for block, x in zip(self.matching_blocks, layer_outs)]
        fused = self.fuse(torch.cat(matched, dim=1))
        logits = self.aux_head(fused)
        prob_patch_res = torch.sigmoid(logits)

        prob_full_res = None
        if out_size is not None:
            prob_full_res = F.interpolate(prob_patch_res, size=out_size, mode="bilinear", align_corners=False)

        return {"prob_patch_res": prob_patch_res, "prob_full_res": prob_full_res, "fused": fused}
