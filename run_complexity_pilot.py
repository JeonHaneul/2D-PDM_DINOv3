"""Locked all16 RGB-D/depth comparison. Segmentation is supervision only.

Example: python run_complexity_pilot.py --run-dir outputs/complexity_pilot_YYYYMMDD
Preparation caches one frozen DINO layer once; three paired seeds share this cache.
The test split is opened for model evaluation only after all validation selections.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from complexity_cues import compute_supervision, depth_geometry, depth_channel_names
from complexity_model import ComplexityModel, load_frozen_dino, dino_layer11

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
TARGETS = tuple(f"{cat}_{i}" for cat in ("book", "fruit", "packaged_food", "toy") for i in range(1, 5))
CAMERAS = ("center", "top", "left", "right", "bottom")
WINDOWS = (48, 96, 160)
CAPACITY = 16
SCHEMA = "complexity_visible_density_pilot_v1"
MODES = ("rgbd", "depth")


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_paths(data_root, row):
    folder, key, camera, split = row
    base = Path(data_root) / "scene" / folder
    return {"rgb": base / "rgb" / f"{key}_{camera}.png",
            "depth": base / "depth" / f"{key}_{camera}.npy",
            "seg": base / "seg" / f"{key}_{camera}.png",
            "mapping": base / "seg" / f"{key.split('_env')[0]}_mapping.json"}


@lru_cache(maxsize=256)
def read_mapping(path):
    return json.loads(Path(path).read_text())


def rgb_tensor(rgb):
    a = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (a - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array(
        [.229, .224, .225], np.float32)[:, None, None]


class PreparationDataset(Dataset):
    def __init__(self, rows, data_root, workspace_root, depth_options):
        self.rows, self.data_root, self.depth_options = rows, data_root, depth_options
        self.workspace = {c: np.load(Path(workspace_root) / f"workspace_{c}.npy") for c in CAMERAS}
        self.empty = {c: np.load(Path(data_root) / "target" / "empty_scene" / "depth" / f"{c}.npy").squeeze()
                      for c in CAMERAS}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        cv2.setNumThreads(1)
        row = self.rows[idx]
        p = sample_paths(self.data_root, row)
        bgr, seg = cv2.imread(str(p["rgb"])), cv2.imread(str(p["seg"]))
        if bgr is None or seg is None:
            raise OSError(f"unreadable scene: {row}")
        depth = np.load(p["depth"]).squeeze()
        ws = self.workspace[row[2]]
        gt = compute_supervision(seg, read_mapping(str(p["mapping"])), ws,
                                 windows=WINDOWS, capacity=CAPACITY)
        geo = depth_geometry(depth, self.empty[row[2]], ws, windows=WINDOWS, **self.depth_options)
        y = np.concatenate((gt["density"], gt["occupancy"]), axis=0)
        valid = np.concatenate((gt["density_valid"], gt["occupancy_valid"]), axis=0)
        return idx, rgb_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), geo["features"], y, valid, json.dumps(
            {"gt": gt["summary"], "depth": geo["summary"]}, allow_nan=False)


def create_protocol(args):
    old = json.loads(Path(args.split_manifest).read_text())
    excluded_test = set()
    for path in args.exclude_test_manifest:
        excluded_test.update(json.loads(Path(path).read_text())["split"]["test"])
    selected = {}
    for offset, (split, count) in enumerate(zip(("train", "val", "test"),
                                               (args.train_keys, args.val_keys, args.test_keys))):
        pool = list(old["full"][split])
        if split == "test":
            pool = [k for k in pool if k not in excluded_test]
        if not 1 <= count <= len(pool):
            raise ValueError(f"invalid {split} count")
        random.Random(3300 + offset).shuffle(pool)
        selected[split] = sorted(pool[:count])
    if any(set(selected[a]) & set(selected[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("scene split leakage")
    rows = [(target, key, camera, split) for split in selected for target in TARGETS
            for key in selected[split] for camera in CAMERAS]
    inventory = {}
    for row in rows:
        for p in sample_paths(args.data_root, row).values():
            st = p.stat()
            inventory[str(p.relative_to(args.data_root))] = [st.st_size, st.st_mtime_ns]
    fixed_assets = {f"workspace/{c}": sha(Path(args.workspace_root) / f"workspace_{c}.npy") for c in CAMERAS}
    fixed_assets.update({f"empty_depth/{c}": sha(Path(args.data_root) / "target" / "empty_scene" / "depth" / f"{c}.npy")
                         for c in CAMERAS})
    code_hashes = {f: sha(ROOT / f) for f in ("complexity_cues.py", "complexity_model.py", "run_complexity_pilot.py")}
    return {"schema": SCHEMA, "rows": rows, "split": selected,
            "method_revision": "empty_reference_corrected_geometry_v2",
            "excluded_previously_inspected_test_keys": sorted(excluded_test),
            "revision_reason": "raw-depth roughness responds to empty drawer walls; use empty-depth displacement before slope removal",
            "sample_counts": {s: len(v) * len(TARGETS) * len(CAMERAS) for s, v in selected.items()},
            "source_inventory": inventory, "fixed_asset_sha256": fixed_assets, "code_sha256": code_hashes,
            "data_root": str(Path(args.data_root).resolve()), "workspace_root": str(Path(args.workspace_root).resolve()),
            "dino_repo": str(Path(args.dino_repo).expanduser().resolve()), "dino_weights": str(Path(args.dino_weights).resolve()),
            "dino_weights_sha256": sha(args.dino_weights), "dino_layers": [11],
            "windows_pixels": WINDOWS, "count_normalizer": CAPACITY,
            "geometry_channels": depth_channel_names(WINDOWS),
            "coordinate_scope": "image-space windows, fixed five-camera rig; not metric area or view invariant density",
            "supervision": "visible mapped asset-label count and occupancy; no hidden count or overlap GT",
            "gt_options": {"min_visible_area_pixels": 32, "min_window_pixels": 16, "min_workspace_fraction": .95},
            "depth_options": {"foreground_threshold_m": .015, "roughness_scale_m": .03,
                              "gradient_scale_m": .02, "depth_range_m": [2.5, 3.5], "min_window_valid_fraction": .25,
                              "roughness_reference": "empty_difference"},
            "inference_inputs": "RGB, depth, fixed workspace and empty-depth calibration; no segmentation or target",
            "feature_contract": "B x 64 x 30 x 40: 55 learned channels + 9 deterministic depth channels",
            "training": {"seeds": args.seeds, "modes": MODES, "max_epochs": args.epochs, "patience": 5,
                         "batch_size": args.batch_size, "lr": .001, "weight_decay": .0001,
                         "loss": "equal channel masked SmoothL1(beta=.05), each channel normalized by its mask mass",
                         "selection": "validation occupied-neighborhood count MAE, equally averaged over three scales",
                         "paired": "identical initialization and permutation for each seed, same full architecture; depth mode zeros DINO features"},
            "metrics": {"density_domain": "density_valid and GT count>0, same mask for every model",
                        "count_mae": "sample-wise masked MAE x16, mean over samples, then equally over three scales and seeds",
                        "all_workspace_count_mae": "same metric including zero-count neighborhoods",
                        "occupancy_mae": "sample-wise fractional-workspace weighted absolute area-fraction error",
                        "bootstrap": "2000 resamples, seed3300, paired entire scene-key clusters including all pools/views/seeds"},
            "adoption_gate": {"relative_count_mae_gain_over_depth_at_least": .10,
                              "paired_scene_bootstrap_95_lower_gain_gt": 0,
                              "improved_cameras_at_least": 3,
                              "must_beat_train_camera_position_mean": True},
            "scientific_limits": ["seen asset library; no unseen-scene-object claim", "not target existence probability",
                                  "no fusion/DRL utility validated", "three seeds do not establish broad robustness"],
            "prepared": False}


def prepare(args):
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    protocol = create_protocol(args)
    reuse_features, reuse_lookup = None, {}
    if args.reuse_rgb_cache:
        previous = Path(args.reuse_rgb_cache)
        source = json.loads((previous / "protocol.json").read_text())
        if (not source.get("prepared") or source["dino_weights_sha256"] != protocol["dino_weights_sha256"]
                or source["dino_layers"] != protocol["dino_layers"]):
            raise ValueError("incompatible frozen RGB feature cache")
        reuse_features = np.load(previous / "rgb_features.npy", mmap_mode="r")
        for i, row in enumerate(source["rows"]):
            rgb_key = str(sample_paths(args.data_root, row)["rgb"].relative_to(args.data_root))
            if source["source_inventory"].get(rgb_key) == protocol["source_inventory"].get(rgb_key):
                reuse_lookup[tuple(row)] = i
        protocol["rgb_cache_reuse"] = {"source": str(previous.resolve()), "protocol_sha256": sha(previous / "protocol.json"),
                                       "reused_samples": sum(tuple(r) in reuse_lookup for r in protocol["rows"]),
                                       "scope": "frozen layer11 RGB only; all GT and geometry recomputed"}
    write_json(run / "protocol.json", protocol)  # Freeze before looking at predictions.
    n = len(protocol["rows"])
    cache = {}
    for name, shape in {"rgb_features": (n, 768, 30, 40), "geometry": (n, 9, 30, 40),
                        "labels": (n, 4, 30, 40), "valid": (n, 4, 30, 40)}.items():
        cache[name] = np.lib.format.open_memmap(run / f"{name}.npy", mode="w+", dtype=np.float16, shape=shape)
    ds = PreparationDataset(protocol["rows"], args.data_root, args.workspace_root, protocol["depth_options"])
    loader = DataLoader(ds, batch_size=args.encode_batch, num_workers=args.workers, shuffle=False)
    backbone = load_frozen_dino(args.dino_repo, args.dino_weights, args.device)
    t0, done, summaries = time.monotonic(), 0, []
    with torch.inference_mode():
        for idx, rgb, geometry, y, valid, summary in loader:
            ix = idx.numpy()
            missing = []
            for slot, index in enumerate(ix):
                old_index = reuse_lookup.get(tuple(protocol["rows"][index]))
                if old_index is None:
                    missing.append(slot)
                else:
                    cache["rgb_features"][index] = reuse_features[old_index]
            if missing:
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    features = dino_layer11(backbone, rgb[missing].to(args.device))
                cache["rgb_features"][ix[missing]] = features.to(torch.float16).cpu().numpy()
            cache["geometry"][ix], cache["labels"][ix], cache["valid"][ix] = geometry.numpy(), y.numpy(), valid.numpy()
            summaries.extend(json.loads(s) for s in summary)
            done += len(ix)
            if done == args.encode_batch or done % 512 == 0 or done == n:
                elapsed = time.monotonic() - t0
                print(f"prepare {done}/{n}; elapsed={elapsed:.1f}s; estimated_remaining={(n-done)*elapsed/done:.1f}s", flush=True)
    for v in cache.values():
        v.flush()
    protocol.update(prepared=True, prepare_seconds=time.monotonic() - t0,
                    cache_shapes={k: list(v.shape) for k, v in cache.items()})
    write_json(run / "sample_diagnostics.json", summaries)
    write_json(run / "protocol.json", protocol)
    del backbone, cache
    if args.device == "cuda":
        torch.cuda.empty_cache()


def load_cache(run, device):
    protocol = json.loads((run / "protocol.json").read_text())
    if not protocol.get("prepared"):
        raise RuntimeError("cache not complete")
    result = {}
    for name in ("rgb_features", "geometry", "labels", "valid"):
        a = np.load(run / f"{name}.npy")
        if list(a.shape) != protocol["cache_shapes"][name] or not np.isfinite(a).all():
            raise ValueError(f"invalid cache: {name}")
        result[name] = torch.from_numpy(a).to(device)
    return result


def validate_checkpoint(checkpoint, protocol_path, mode, seed):
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("mode") != mode
            or checkpoint.get("seed") != seed or checkpoint.get("protocol_sha256") != sha(protocol_path)):
        raise ValueError("checkpoint schema/mode/seed/protocol contract mismatch")


def sample_metrics(pred, y, valid):
    masks = valid[:, :3].float()
    occupied = masks * (y[:, :3] > 0)
    delta = (pred[:, :3] - y[:, :3]).abs().float() * CAPACITY
    dims = (-2, -1)
    mass = occupied.sum(dims)
    density = (delta * occupied).sum(dims) / mass.clamp_min(1)
    all_mass = masks.sum(dims)
    all_density = (delta * masks).sum(dims) / all_mass.clamp_min(1)
    occupancy_mass = valid[:, 3].float().sum(dims)
    occupancy = ((pred[:, 3] - y[:, 3]).abs().float() * valid[:, 3]).sum(dims) / occupancy_mass.clamp_min(1)
    return torch.cat((density, all_density, occupancy[:, None], mass, all_mass, occupancy_mass[:, None]), dim=1)


def metric_means(array):
    # Missing occupied windows are excluded, never counted as perfect predictions.
    values = []
    for j in range(7):
        m = array[:, 7 + j] > 0 if j < 6 else array[:, 13] > 0
        if not m.any():
            raise ValueError("no valid evaluation region")
        values.append(float(array[m, j].mean()))
    return {"count_mae": float(np.mean(values[:3])), "count_mae_by_scale": values[:3],
            "all_workspace_count_mae": float(np.mean(values[3:6])),
            "occupancy_mae": values[6]}


@torch.inference_mode()
def predict_metrics(model, cache, indices, mode, batch=64, intervention=None):
    model.eval()
    results = []
    for start in range(0, len(indices), batch):
        ix = indices[start:start+batch]
        rgb = cache["rgb_features"][ix].float()
        if intervention == "zero_rgb":
            rgb = torch.zeros_like(rgb)
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=rgb.is_cuda):
            pred, _ = model(rgb, cache["geometry"][ix].float(), mode)
        results.append(sample_metrics(pred.float(), cache["labels"][ix].float(), cache["valid"][ix].float()).cpu().numpy())
    return np.concatenate(results)


def train(args):
    run = Path(args.run_dir)
    protocol = json.loads((run / "protocol.json").read_text())
    if not protocol["prepared"]:
        raise RuntimeError("cache not complete")
    cache = load_cache(run, args.device)
    indices = {s: np.array([i for i, row in enumerate(protocol["rows"]) if row[3] == s]) for s in ("train", "val", "test")}
    t0 = time.monotonic()
    config = protocol["training"]
    for seed in config["seeds"]:
        for mode in config["modes"]:
            directory = run / f"{mode}_seed{seed}"
            directory.mkdir(exist_ok=False)
            torch.manual_seed(seed)
            model = ComplexityModel().to(args.device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            generator = torch.Generator().manual_seed(seed)
            best, stale, history = float("inf"), 0, []
            for epoch in range(1, config["max_epochs"] + 1):
                te = time.monotonic()
                model.train()
                order = indices["train"][torch.randperm(len(indices["train"]), generator=generator).numpy()]
                losses = []
                for start in range(0, len(order), config["batch_size"]):
                    ix = order[start:start+config["batch_size"]]
                    y, valid = cache["labels"][ix].float(), cache["valid"][ix].float()
                    with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
                        pred, _ = model(cache["rgb_features"][ix].float(), cache["geometry"][ix].float(), mode)
                    error = F.smooth_l1_loss(pred.float(), y, reduction="none", beta=.05)
                    loss = ((error * valid).sum((0, 2, 3)) / valid.sum((0, 2, 3)).clamp_min(1)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    losses.append(float(loss.detach()))
                val = metric_means(predict_metrics(model, cache, indices["val"], mode))
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": val,
                                "seconds": time.monotonic() - te})
                if val["count_mae"] < best:
                    best, stale = val["count_mae"], 0
                    torch.save({"schema": SCHEMA, "model": model.state_dict(), "mode": mode, "seed": seed,
                                "best_epoch": epoch, "validation": val, "protocol_sha256": sha(run / "protocol.json")}, directory / "best.pth")
                else:
                    stale += 1
                write_json(directory / "history.json", history)
                print(f"{mode} seed{seed} epoch{epoch}: val_count_mae={val['count_mae']:.4f}; {history[-1]['seconds']:.1f}s", flush=True)
                if stale >= config["patience"]:
                    break
            write_json(directory / "completion.json", {"best_count_mae": best, "epochs": len(history), "checkpoint_sha256": sha(directory / "best.pth")})
    write_json(run / "training_complete.json", {"seconds": time.monotonic() - t0, "seeds": config["seeds"], "modes": config["modes"]})
    del cache
    if args.device == "cuda":
        torch.cuda.empty_cache()


def bootstrap_gain(rgbd, depth, rows):
    keys = sorted({r[1] for r in rows})
    clusters = [np.array([i for i, r in enumerate(rows) if r[1] == k]) for k in keys]
    rng, deltas = np.random.default_rng(3300), []
    for _ in range(2000):
        ix = np.concatenate([clusters[j] for j in rng.integers(len(keys), size=len(keys))])
        deltas.append(metric_means(depth[ix])["count_mae"] - metric_means(rgbd[ix])["count_mae"])
    return {"unit": "scene key with all source pools, five views, and seed-averaged paired errors",
            "n_scene_keys": len(keys), "count_mae_gain_ci95": np.percentile(deltas, [2.5, 97.5]).tolist()}


def evaluate(args):
    run = Path(args.run_dir)
    if not (run / "training_complete.json").is_file():
        raise RuntimeError("finish every validation selection before test evaluation")
    protocol = json.loads((run / "protocol.json").read_text())
    cache = load_cache(run, args.device)
    rows = protocol["rows"]
    test = np.array([i for i, r in enumerate(rows) if r[3] == "test"])
    tr = np.array([i for i, r in enumerate(rows) if r[3] == "train"])
    test_rows = [rows[i] for i in test]
    arrays, results = {}, {}
    for mode in MODES:
        all_seeds = []
        for seed in protocol["training"]["seeds"]:
            ckpt = torch.load(run / f"{mode}_seed{seed}" / "best.pth", map_location=args.device, weights_only=True)
            validate_checkpoint(ckpt, run / "protocol.json", mode, seed)
            model = ComplexityModel().to(args.device)
            model.load_state_dict(ckpt["model"], strict=True)
            a = predict_metrics(model, cache, test, mode)
            all_seeds.append(a)
            results[f"{mode}_seed{seed}"] = {**metric_means(a), "best_epoch": ckpt["best_epoch"]}
            np.save(run / f"test_metrics_{mode}_seed{seed}.npy", a)
            if mode == "rgbd" and seed == protocol["training"]["seeds"][0]:
                results["rgbd_zero_rgb_intervention"] = metric_means(predict_metrics(model, cache, test, mode, intervention="zero_rgb"))
                save_panels(run, model, cache, test, rows, protocol)
        arrays[mode] = np.mean(all_seeds, axis=0)
        results[mode] = metric_means(arrays[mode])
    # Camera-position baseline fitted on training labels only (including zero neighborhoods).
    baseline = torch.zeros_like(cache["labels"][test]).float()
    for cam in CAMERAS:
        train_cam = tr[[rows[i][2] == cam for i in tr]]
        slots = np.flatnonzero([r[2] == cam for r in test_rows])
        weights = cache["valid"][train_cam].float()
        mean = (cache["labels"][train_cam].float() * weights).sum(0) / weights.sum(0).clamp_min(1)
        baseline[slots] = mean
    ba = sample_metrics(baseline, cache["labels"][test].float(), cache["valid"][test].float()).cpu().numpy()
    results["train_camera_position_mean"] = metric_means(ba)
    direct = cache["labels"][test].float().clone()
    direct[:, 3] = cache["geometry"][test, 1].float()
    results["direct_depth_occupancy_mae"] = metric_means(sample_metrics(direct, cache["labels"][test].float(), cache["valid"][test].float()).cpu().numpy())["occupancy_mae"]
    by_camera = {cam: {mode: metric_means(a[np.array([r[2] == cam for r in test_rows])]) for mode, a in arrays.items()} for cam in CAMERAS}
    by_pool = {pool: {mode: metric_means(a[np.array([r[0] == pool for r in test_rows])]) for mode, a in arrays.items()} for pool in TARGETS}
    ci = bootstrap_gain(arrays["rgbd"], arrays["depth"], test_rows)
    gain = 1 - results["rgbd"]["count_mae"] / results["depth"]["count_mae"]
    improved = sum(v["rgbd"]["count_mae"] < v["depth"]["count_mae"] for v in by_camera.values())
    gates = {"relative_gain_at_least_10pct": gain >= .10, "paired_ci_lower_positive": ci["count_mae_gain_ci95"][0] > 0,
             "at_least_three_cameras_improved": improved >= 3,
             "beats_train_position_mean": results["rgbd"]["count_mae"] < results["train_camera_position_mean"]["count_mae"]}
    report = {"schema": SCHEMA, "scope": protocol["scientific_limits"], "sample_counts": protocol["sample_counts"],
              "metrics": protocol["metrics"], "results": results, "by_camera": by_camera, "by_source_pool": by_pool,
              "bootstrap": ci, "relative_gain": gain, "improved_camera_count": improved,
              "adoption_gate_results": gates, "rgbd_pilot_gate_passed": all(gates.values())}
    write_json(run / "summary.json", report)
    print(json.dumps({"rgbd": results["rgbd"], "depth": results["depth"], "gain": gain,
                      "ci": ci, "gates": gates}, indent=2), flush=True)


@torch.inference_mode()
def save_panels(run, model, cache, test, rows, protocol):
    """Predetermined first test scene, one source pool/category, all five views."""
    out = run / "prediction_panels"
    out.mkdir(exist_ok=True)
    for pool in ("book_1", "fruit_1", "packaged_food_1", "toy_1"):
        strips = []
        key = protocol["split"]["test"][0]
        for camera in CAMERAS:
            idx = next(i for i in test if rows[i][:3] == [pool, key, camera])
            with torch.autocast(device_type=cache["geometry"].device.type, dtype=torch.bfloat16, enabled=cache["geometry"].is_cuda):
                pred, _ = model(cache["rgb_features"][idx:idx+1].float(), cache["geometry"][idx:idx+1].float())
            p, y = pred[0].float().cpu().numpy(), cache["labels"][idx].float().cpu().numpy()
            g, v = cache["geometry"][idx].float().cpu().numpy(), cache["valid"][idx].float().cpu().numpy()
            scene = cv2.imread(str(sample_paths(protocol["data_root"], rows[idx])["rgb"]))
            cols = [(scene, f"{pool} {camera}"), (y[1] * v[1], "GT count, 96px /16"),
                    (p[1] * v[1], "RGB-D count, same scale"), (np.abs(y[1]-p[1])*v[1], "Count abs error /16"),
                    (y[3], "GT occupancy"), (g[1], "Direct depth occupancy"), (g[4], "Direct plane residual,96px")]
            tiles = []
            for arr, title in cols:
                if arr.ndim == 2:
                    arr = cv2.applyColorMap(np.uint8(np.clip(arr, 0, 1) * 255), cv2.COLORMAP_VIRIDIS)
                arr = cv2.resize(arr, (320, 240), interpolation=cv2.INTER_NEAREST)
                tile = np.zeros((268, 320, 3), np.uint8)
                tile[28:] = arr
                cv2.putText(tile, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255), 1, cv2.LINE_AA)
                tiles.append(tile)
            strips.append(np.concatenate(tiles, axis=1))
        cv2.imwrite(str(out / f"{pool}_five_views.png"), np.concatenate(strips, axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "train", "evaluate"), default="all")
    parser.add_argument("--data-root", type=Path, default=SRC / "260714_data")
    parser.add_argument("--workspace-root", type=Path, default=ROOT / "experiments/occlusion_gt_pilot/workspace_masks_v4")
    parser.add_argument("--split-manifest", default=str(ROOT / "outputs/occlusion_full16_20260828_114243/split_manifest.json"))
    parser.add_argument("--exclude-test-manifest", nargs="*", default=[], help="earlier pilot protocol files whose test keys must remain excluded")
    parser.add_argument("--reuse-rgb-cache", help="reuse unchanged frozen RGB inputs only; recompute all supervision and geometry")
    parser.add_argument("--dino-repo", default=str(Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"))
    parser.add_argument("--dino-weights", default=str(SRC / "model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
    parser.add_argument("--train-keys", type=int, default=48)
    parser.add_argument("--val-keys", type=int, default=12)
    parser.add_argument("--test-keys", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in GPU-enabled environment")
    if args.stage in ("all", "prepare"):
        prepare(args)
    if args.stage in ("all", "train"):
        train(args)
    if args.stage in ("all", "evaluate"):
        evaluate(args)


if __name__ == "__main__":
    main()
