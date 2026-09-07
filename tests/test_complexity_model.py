import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from complexity_model import ComplexityModel
from run_complexity_pilot import SCHEMA, sample_metrics, metric_means, bootstrap_gain, sha, validate_checkpoint


class ComplexityModelTests(unittest.TestCase):
    def test_checkpoint_contract_rejects_same_shape_wrong_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text('{"test": 1}')
            checkpoint = {"schema": SCHEMA, "mode": "rgbd", "seed": 0, "protocol_sha256": sha(path)}
            validate_checkpoint(checkpoint, path, "rgbd", 0)
            for key, wrong in (("mode", "depth"), ("seed", 1), ("schema", "old"), ("protocol_sha256", "bad")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_checkpoint({**checkpoint, key: wrong}, path, "rgbd", 0)

    def test_depth_mode_ignores_rgb_and_preserves_geometry(self):
        torch.manual_seed(0)
        model = ComplexityModel().eval()
        geo = torch.rand(2, 9, 5, 7)
        with torch.no_grad():
            a, features = model(torch.randn(2, 768, 5, 7), geo, "depth")
            b, _ = model(torch.randn(2, 768, 5, 7), geo, "depth")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(features[:, -9:], geo, rtol=0, atol=0)
        self.assertEqual(features.shape, (2, 64, 5, 7))

    def test_native_count_units_and_empty_regions(self):
        y = torch.zeros(2, 4, 2, 2)
        y[0, :3] = .25
        pred = y + .125
        v = torch.ones_like(y)
        a = sample_metrics(pred, y, v).numpy()
        result = metric_means(a)
        self.assertAlmostEqual(result["count_mae"], 2.)
        self.assertAlmostEqual(result["occupancy_mae"], .125)
        self.assertTrue((a[1, 7:10] == 0).all())

    def test_invalid_predictions_do_not_change_metrics(self):
        y = torch.ones(1, 4, 2, 2) * .25
        v = torch.ones_like(y)
        v[..., 0, 0] = 0
        p = y.clone()
        p[..., 0, 0] = 99
        self.assertEqual(metric_means(sample_metrics(p, y, v).numpy())["count_mae"], 0.)

    def test_cluster_bootstrap_preserves_constant_paired_difference(self):
        y = torch.ones(4, 4, 1, 1) * .25
        v = torch.ones_like(y)
        a = sample_metrics(y, y, v).numpy()
        b = sample_metrics(y + .125, y, v).numpy()
        rows = [["book_1", f"scene{i//2}", "center", "test"] for i in range(4)]
        result = bootstrap_gain(a, b, rows)
        np.testing.assert_allclose(result["count_mae_gain_ci95"], [2, 2])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
