"""Small scientific counterexamples, runnable with Python's unittest."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from complexity_cues import (  # noqa: E402
    COUNT_CAPACITY, DEPTH_FEATURE_CHANNELS, compute_supervision, depth_geometry,
)


class VisibleCountTests(unittest.TestCase):
    def setUp(self):
        self.seg = np.zeros((192, 192, 3), np.uint8)
        self.workspace = np.ones((192, 192), bool)
        self.mapping = {"one": {"color_rgb": [30, 90, 200]},
                        "two": {"color_bgr": [100, 10, 50]}}

    def call(self, **kwargs):
        return compute_supervision(self.seg, self.mapping, self.workspace, **kwargs)

    def test_empty_scene_has_zero_count_and_occupancy(self):
        out = self.call()
        self.assertFalse(out["density"].any())
        self.assertFalse(out["occupancy"].any())
        self.assertEqual(out["density_valid"][2, 0, 0], 0)
        self.assertEqual(out["density_valid"][2, 6, 6], 1)

    def test_two_visible_labels_and_disconnected_fragments_are_not_extra_instances(self):
        self.seg[85:91, 85:91] = [30, 90, 200]
        self.seg[100:106, 100:106] = [30, 90, 200]
        self.seg[95:101, 95:101] = [100, 10, 50]
        out = self.call()
        np.testing.assert_allclose(out["density"][:, 6, 6], 2 / COUNT_CAPACITY)
        self.assertEqual(len(out["summary"]["eligible_label_groups"]), 2)
        # Two disconnected copies of one saved colour cannot be distinguished.
        self.seg[self.seg[:, :, 0] == 100] = 0
        np.testing.assert_allclose(self.call()["density"][:, 6, 6], 1 / COUNT_CAPACITY)

    def test_colour_alias_and_name_permutation_do_not_change_gt(self):
        self.seg[80:120, 80:120] = [30, 90, 200]
        baseline = self.call()
        self.mapping = {"arbitrary new name": {"color_bgr": [30, 90, 200]},
                        "alias": {"color_rgb": [30, 90, 200]}}
        aliased = self.call()
        np.testing.assert_array_equal(baseline["density"], aliased["density"])
        self.assertEqual(aliased["summary"]["mapping_colour_groups"], 1)

    def test_minimum_visible_and_intersection_area_are_separate(self):
        self.seg[100:104, 100:104] = [30, 90, 200]  # 16 pixels, below visible minimum.
        self.assertEqual(float(self.call()["density"].max()), 0)
        self.assertGreater(float(self.call()["occupancy"].max()), 0)
        self.seg[90:94, 90:94] = [30, 90, 200]  # total 32, both in central window.
        self.assertEqual(float(self.call()["density"][0, 6, 6]), 1 / COUNT_CAPACITY)
        self.assertEqual(float(self.call(min_window_pixels=33)["density"].max()), 0)

    def test_unknown_colour_invalidates_labels_without_becoming_negative_occupancy(self):
        self.seg[96:112, 96:112] = [1, 2, 3]
        out = self.call()
        self.assertEqual(out["summary"]["unknown_nonblack_pixels"], 256)
        self.assertEqual(float(out["occupancy_valid"][0, 6, 6]), 0)
        self.assertFalse(out["density_valid"][:, 6, 6].any())

    def test_clipping_reported_and_empty_workspace_safe(self):
        self.seg[80:95, 80:95] = [30, 90, 200]
        self.seg[100:115, 100:115] = [100, 10, 50]
        out = self.call(capacity=1)
        self.assertEqual(float(out["density"].max()), 1)
        self.assertGreater(out["summary"]["clipping_fraction_by_window"][0], 0)
        self.workspace[:] = False
        out = self.call()
        for key in ("density", "occupancy", "workspace", "density_valid"):
            self.assertFalse(out[key].any())

    def test_colour_bgr_precedes_true_rgb_in_new_mapping(self):
        self.seg[80:120, 80:120] = [30, 90, 200]
        self.mapping = {"new": {"color_bgr": [30, 90, 200], "color_rgb": [200, 90, 30]}}
        self.assertEqual(self.call()["summary"]["unknown_nonblack_pixels"], 0)


class DepthCueTests(unittest.TestCase):
    def setUp(self):
        self.workspace = np.ones((192, 192), bool)
        self.empty = np.full((192, 192), 3.2, np.float64)

    def test_sloped_plane_and_missing_depth_do_not_look_rough(self):
        y, x = np.indices(self.empty.shape)
        slope = 3.1 + 0.0004*x + 0.0002*y
        baseline = depth_geometry(slope, slope, self.workspace)
        self.assertLess(float(baseline["features"][3:].max()), 1e-5)
        self.assertFalse(baseline["direct_occupancy"].any())
        sparse = slope.copy()
        sparse[::7, ::9] = 0
        sparse[40:50, 40:50] = np.nan
        out = depth_geometry(sparse, slope, self.workspace)
        self.assertLess(float(out["features"][3:].max()), 1e-5)
        self.assertTrue(np.isfinite(out["features"]).all())
        self.assertFalse(out["direct_occupancy"].any())

    def test_real_depth_step_triggers_cues_and_occupancy(self):
        depth = self.empty.copy()
        depth[64:128, 64:128] -= 0.1
        out = depth_geometry(depth, self.empty, self.workspace)
        self.assertEqual(out["features"].shape, (DEPTH_FEATURE_CHANNELS, 12, 12))
        self.assertEqual(float(out["direct_occupancy"][0, 6, 6]), 1)
        self.assertGreater(float(out["features"][3:6, 4, 4].max()), 0.1)
        self.assertGreater(float(out["features"][6:, 4, 4].max()), 0.1)

    def test_slope_addition_preserves_roughness(self):
        depth = self.empty.copy()
        depth[64:128, 64:128] -= 0.1
        y, x = np.indices(depth.shape)
        slope = 0.0002*x + 0.0001*y
        out = depth_geometry(depth, self.empty, self.workspace)
        tilted = depth_geometry(depth+slope, self.empty+slope, self.workspace)
        np.testing.assert_allclose(out["features"][3:], tilted["features"][3:], atol=2e-6)
        np.testing.assert_array_equal(out["direct_occupancy"], tilted["direct_occupancy"])

    def test_invalid_empty_reference_cannot_create_foreground(self):
        depth = self.empty.copy() - 0.1
        self.empty[:] = 0
        out = depth_geometry(depth, self.empty, self.workspace)
        self.assertFalse(out["validity"].any())
        self.assertFalse(out["direct_occupancy"].any())
        self.assertTrue(np.isfinite(out["features"]).all())

    def test_no_support_outside_workspace(self):
        self.workspace[:96] = False
        out = depth_geometry(self.empty, self.empty, self.workspace)
        self.assertFalse(out["features"][:, :6].any())
        self.assertFalse(out["validity"][:, :6].any())

    def test_reference_mode_preserves_old_default_and_first_three_channels(self):
        depth = self.empty.copy()
        depth[64:128, 64:128] -= 0.1
        default = depth_geometry(depth, self.empty, self.workspace)
        explicit = depth_geometry(depth, self.empty, self.workspace, roughness_reference="scene")
        difference = depth_geometry(depth, self.empty, self.workspace,
                                    roughness_reference="empty_difference")
        np.testing.assert_array_equal(default["features"], explicit["features"])
        for key in ("direct_occupancy", "validity"):
            np.testing.assert_array_equal(default[key], difference[key])
        np.testing.assert_array_equal(default["features"][:3], difference["features"][:3])
        self.assertEqual(default["summary"]["roughness_reference"], "scene")
        self.assertEqual(difference["summary"]["roughness_reference"], "empty_difference")

    def test_curved_and_step_empty_background_has_exactly_zero_difference_roughness(self):
        y, x = np.indices(self.empty.shape)
        curved = 3.0 + 0.00001*((x-96)**2 + (y-96)**2)
        stepped = self.empty.copy()
        stepped[:, :64] -= 0.3
        for empty in (curved, stepped):
            with self.subTest(background="curved" if empty is curved else "step"):
                old = depth_geometry(empty, empty, self.workspace)
                out = depth_geometry(empty, empty, self.workspace,
                                     roughness_reference="empty_difference")
                self.assertGreater(float(old["features"][3:].max()), 0.1)
                np.testing.assert_array_equal(out["features"][3:], 0)
                self.assertFalse(out["direct_occupancy"].any())
                # Missing scene/empty pixels cannot turn fixed geometry into clutter.
                depth = empty.copy()
                depth[::7, ::9] = np.nan
                reference = empty.copy()
                reference[20:30, 60:70] = 0
                sparse = depth_geometry(depth, reference, self.workspace,
                                        roughness_reference="empty_difference")
                np.testing.assert_array_equal(sparse["features"][3:], 0)
                self.assertTrue(np.isfinite(sparse["features"]).all())

    def test_clutter_displacement_survives_curved_background_removal(self):
        y, x = np.indices(self.empty.shape)
        curved = 3.0 + 0.00001*((x-96)**2 + (y-96)**2)
        displacement = np.zeros_like(curved)
        displacement[64:128, 64:128] = 0.1
        out = depth_geometry(curved-displacement, curved, self.workspace,
                             roughness_reference="empty_difference")
        flat = depth_geometry(self.empty-displacement, self.empty, self.workspace,
                              roughness_reference="empty_difference")
        self.assertGreater(float(out["features"][3:6].max()), 0.1)
        self.assertGreater(float(out["features"][6:].max()), 0.1)
        np.testing.assert_allclose(out["features"][3:], flat["features"][3:], atol=1e-6)
        np.testing.assert_array_equal(out["direct_occupancy"], flat["direct_occupancy"])

    def test_negative_displacement_is_not_clipped_before_geometry(self):
        displacement = np.full_like(self.empty, -0.1)
        displacement[64:128, 64:128] = -0.2
        out = depth_geometry(self.empty-displacement, self.empty, self.workspace,
                             roughness_reference="empty_difference")
        self.assertFalse(out["direct_occupancy"].any())
        self.assertGreater(float(out["features"][3:6].max()), 0.1)
        self.assertGreater(float(out["features"][6:].max()), 0.1)

    def test_invalid_reference_mode_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "roughness_reference"):
            depth_geometry(self.empty, self.empty, self.workspace, roughness_reference="unknown")


if __name__ == "__main__":
    unittest.main()
