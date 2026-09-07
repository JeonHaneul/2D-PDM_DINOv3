"""Run the complexity pilot using scene RGB-D and fixed rig calibration only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from complexity_cues import depth_geometry
from complexity_model import ComplexityModel, load_frozen_dino, dino_layer11
from run_complexity_pilot import rgb_tensor, sha, validate_checkpoint


class ComplexityPredictor:
    def __init__(self, run_dir, seed=0, mode="rgbd", device="cuda", dino_repo=None, dino_weights=None):
        run = Path(run_dir)
        self.protocol = json.loads((run / "protocol.json").read_text())
        checkpoint = torch.load(run / f"{mode}_seed{seed}" / "best.pth", map_location=device, weights_only=True)
        validate_checkpoint(checkpoint, run / "protocol.json", mode, seed)
        self.device, self.mode = device, mode
        self.model = ComplexityModel().to(device).eval()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        weights = dino_weights or self.protocol["dino_weights"]
        if mode == "rgbd" and sha(weights) != self.protocol["dino_weights_sha256"]:
            raise ValueError("DINO weights differ from training")
        self.backbone = load_frozen_dino(dino_repo or self.protocol["dino_repo"], weights, device) if mode == "rgbd" else None

    @torch.inference_mode()
    def predict(self, rgb, depth, camera):
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError("pilot calibration requires RGB H480 x W640 x3 and depth H480 x W640")
        ws_path = Path(self.protocol["workspace_root"]) / f"workspace_{camera}.npy"
        empty_path = Path(self.protocol["data_root"]) / "target" / "empty_scene" / "depth" / f"{camera}.npy"
        for key, p in ((f"workspace/{camera}", ws_path), (f"empty_depth/{camera}", empty_path)):
            if sha(p) != self.protocol["fixed_asset_sha256"][key]:
                raise ValueError(f"calibration changed: {key}")
        geometry = depth_geometry(depth, np.load(empty_path).squeeze(), np.load(ws_path),
                                  windows=self.protocol["windows_pixels"], **self.protocol["depth_options"])
        # Match the immutable float16 training-cache representation.
        geo = torch.from_numpy(geometry["features"])[None].to(self.device).to(torch.float16).float()
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            if self.backbone is not None:
                # Match the contiguous NCHW tensors produced by the training DataLoader.
                image = torch.from_numpy(rgb_tensor(rgb))[None].contiguous().to(self.device)
                encoded = dino_layer11(self.backbone, image).to(torch.float16).float()
            else:
                encoded = torch.zeros((1, 768, 30, 40), device=self.device)
            maps, features = self.model(encoded, geo, self.mode)
        return {"maps": maps[0].float().cpu().numpy(), "features": features[0].float().cpu().numpy(),
                "geometry": geometry["features"], "validity": geometry["validity"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--rgb", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--camera", required=True, choices=("center", "top", "left", "right", "bottom"))
    p.add_argument("--out", required=True, help="new .npz file containing 4 maps and F_C64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("rgbd", "depth"), default="rgbd")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() or out.suffix != ".npz":
        raise ValueError("out must be a new .npz path")
    bgr = cv2.imread(args.rgb)
    if bgr is None:
        raise OSError(args.rgb)
    predictor = ComplexityPredictor(args.run_dir, args.seed, args.mode, args.device)
    result = predictor.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.load(args.depth).squeeze(), args.camera)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(json.dumps({k: list(v.shape) for k, v in result.items()}))


if __name__ == "__main__":
    main()
