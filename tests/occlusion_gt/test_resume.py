"""checkpoint resume이 처음부터 실행한 것과 완전히 같은 결과를 내는지 회귀 검증
(500 pose 실행 -> 같은 설정으로 1000까지 resume vs 처음부터 1000 실행). production인
V2만 검증(V1은 reference로만 유지, resume 로직 자체는 V1/V2가 공유하는 구조이지만 이
테스트는 V2 경로만 실행함). 개발 중 batch_size 배수로 안 맞춰서 자르면 max_poses 확장
시 일부 pose가 조용히 누락되는 버그를 여기서 잡았음(build_pose_grid의 batch_size 올림
처리로 수정됨)."""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")
DINOV3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOOK1_USD = os.path.join(SRC_DIR, "asset", "Book", "book_1", "Book_02.usd")  # 절대경로로 SRC_DIR 실제 사용


def _run(script, target, max_poses, resume=False):
    cmd = [sys.executable, os.path.join(DINOV3_DIR, script),
           "--target", target, "--usd", BOOK1_USD, "--clutter_target", "packaged_food_2",
           "--max_scenes", "1", "--max_poses", str(max_poses), "--batch_size", "32",
           "--checkpoint_every_batches", "4"]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, cwd=DINOV3_DIR, check=True, capture_output=True)


def _compare_target_dirs(dir_a, dir_b):
    scene_dirs_a = sorted(glob.glob(f"{dir_a}/scene*"))
    assert len(scene_dirs_a) > 0
    for sd in scene_dirs_a:
        tag = os.path.basename(sd)
        for r in ["legacy", "corrected"]:
            a = np.load(f"{sd}/map_{r}.npy")
            b = np.load(f"{dir_b}/{tag}/map_{r}.npy")
            assert np.array_equal(a, b), f"resume vs fresh map 불일치: {tag}/{r}"


def test_resume_v2_matches_fresh():
    out_root = os.path.join(DINOV3_DIR, "occlusion_gt_output_batched_v2")
    resume_dir = os.path.join(out_root, "_test_resume", "scale_1.0")
    fresh_dir = os.path.join(out_root, "_test_fresh", "scale_1.0")
    for d in [os.path.dirname(resume_dir), os.path.dirname(fresh_dir)]:
        shutil.rmtree(d, ignore_errors=True)

    _run("generate_occlusion_gt_batched_v2.py", "_test_resume", 500)
    _run("generate_occlusion_gt_batched_v2.py", "_test_resume", 1000, resume=True)
    _run("generate_occlusion_gt_batched_v2.py", "_test_fresh", 1000)

    _compare_target_dirs(resume_dir, fresh_dir)

    for d in [os.path.dirname(resume_dir), os.path.dirname(fresh_dir)]:
        shutil.rmtree(d, ignore_errors=True)
    print("PASS: resume(500->1000) == fresh(1000) 완전 일치 (V2)")


if __name__ == "__main__":
    test_resume_v2_matches_fresh()
