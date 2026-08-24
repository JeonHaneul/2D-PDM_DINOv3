"""Small deterministic checks for the target-mask geometry used by Occlusion FiLM."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from target_utils import extract_target_geometry


def main() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:6, 5:15] = True
    geometry = extract_target_geometry(mask)

    assert geometry.shape == (68,)
    assert geometry.dtype == np.float32
    assert np.allclose(geometry[:4], [0.2, 0.4, 0.5, np.log(2.5)])
    assert np.isfinite(geometry).all()
    assert np.all((geometry[4:] >= 0.0) & (geometry[4:] <= 1.0))

    # Common uint8 masks must mean the same thing as boolean masks (255 is not 255 pixels).
    geometry_u8 = extract_target_geometry(mask.astype(np.uint8) * 255)
    assert np.array_equal(geometry_u8, geometry)

    empty = extract_target_geometry(np.zeros((10, 20), dtype=bool))
    assert np.array_equal(empty, np.zeros(68, dtype=np.float32))

    print("target geometry PASS: 68-D layout, scale fields, silhouette, empty mask")


if __name__ == "__main__":
    main()
