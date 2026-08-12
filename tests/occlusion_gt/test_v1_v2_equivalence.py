"""generate_occlusion_gt_batched.py(V1, pose만 GPU batch)와
generate_occlusion_gt_batched_v2.py(V2, scene도 GPU 벡터화)가 같은 입력에서 완전히 같은
결과(N_all, scene별 map)를 내는지 회귀 검증. V2 채택 근거였던 "요청 1000 pose(batch_size=32라
실제로는 1024 pose로 올림 처리됨) x 20 scene에서 200개 map 전부 byte-identical" 결과를
재현 가능하게 유지."""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

SRC_DIR = os.environ.get("PDM_SRC_ROOT", "/home/haneul/isaacsim/src")
DINOV3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOY3_USD = os.path.join(SRC_DIR, "asset", "Toy", "toy_3", "Shield_Controller.usd")  # 절대경로로 SRC_DIR 실제 사용
REQUESTED_MAX_POSES = 1000
BATCH_SIZE = 32
EFFECTIVE_POSES = ((REQUESTED_MAX_POSES + BATCH_SIZE - 1) // BATCH_SIZE) * BATCH_SIZE  # batch_size 배수로 올림

V1_OUT = os.path.join(DINOV3_DIR, "occlusion_gt_output_batched", "_test_v1v2_v1", "scale_1.0")
V2_OUT = os.path.join(DINOV3_DIR, "occlusion_gt_output_batched_v2", "_test_v1v2_v2", "scale_1.0")


def _run(script, target):
    # target에 실제 등록된 이름이 아닌 테스트 전용 가짜 이름을 쓰므로(출력 디렉토리 격리 목적),
    # base_z를 BASE_Z_TABLE/자동산출 fallback에 맡기면 V1과 V2가 서로 다른(혹은 서로 다르게
    # 수정된) fallback 로직을 탈 위험이 있다 -- toy_3의 실측값(0.01)을 양쪽에 명시적으로
    # 강제해서 이 테스트가 "fallback이 우연히 같은가"가 아니라 "핵심 알고리즘이 같은가"만
    # 검증하게 한다.
    cmd = [sys.executable, os.path.join(DINOV3_DIR, script),
           "--target", target, "--usd", TOY3_USD, "--clutter_target", "toy_3",
           "--base_z", "0.01",
           "--max_poses", str(REQUESTED_MAX_POSES), "--batch_size", str(BATCH_SIZE),
           "--checkpoint_every_batches", "500"]
    subprocess.run(cmd, cwd=DINOV3_DIR, check=True, capture_output=True)


def test_v1_v2_identical():
    for d in [os.path.dirname(V1_OUT), os.path.dirname(V2_OUT)]:
        shutil.rmtree(d, ignore_errors=True)

    _run("generate_occlusion_gt_batched.py", "_test_v1v2_v1")
    _run("generate_occlusion_gt_batched_v2.py", "_test_v1v2_v2")

    for cam in ["center", "left", "right", "top", "bottom"]:
        for r in ["legacy", "corrected"]:
            a = np.load(f"{V1_OUT}/_coverage/N_all_{r}_{cam}.npy")
            b = np.load(f"{V2_OUT}/_coverage/N_all_{r}_{cam}.npy")
            assert np.array_equal(a, b), f"N_all 불일치: {r}/{cam}"

    scene_dirs_v1 = sorted(glob.glob(f"{V1_OUT}/scene*"))
    assert len(scene_dirs_v1) > 0, "V1 출력이 비어있음"
    n_checked = 0
    for sd in scene_dirs_v1:
        tag = os.path.basename(sd)
        bd = f"{V2_OUT}/{tag}"
        for r in ["legacy", "corrected"]:
            a = np.load(f"{sd}/map_{r}.npy")
            b = np.load(f"{bd}/map_{r}.npy")
            assert np.array_equal(a, b), f"map 불일치: {tag}/{r}"
            n_checked += 1

    for d in [os.path.dirname(V1_OUT), os.path.dirname(V2_OUT)]:
        shutil.rmtree(d, ignore_errors=True)

    print(f"PASS: N_all 10개 조합 + map {n_checked}개 조합 전부 완전 일치 "
          f"(요청 {REQUESTED_MAX_POSES} pose -> 실제 처리 {EFFECTIVE_POSES} pose, batch_size={BATCH_SIZE} 배수로 올림)")


if __name__ == "__main__":
    test_v1_v2_identical()
