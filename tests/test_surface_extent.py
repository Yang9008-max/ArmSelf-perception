import unittest
from unittest.mock import patch

import numpy as np

from embodiment.model import basis
from embodiment.surface_extent import SurfaceAtlas, discover_surface, ridge_fit


class SurfaceExtentTests(unittest.TestCase):
    def atlas(self):
        w = np.zeros((2, 36, 3), np.float32)
        w[:, 0] = [[0., 0., .8], [.1, 0., .8]]
        w[:, 1, 0] = .1
        return SurfaceAtlas(dict(center=np.zeros(7), scale=np.ones(7), weights=w,
            training_q=np.zeros((1, 7)), gray=np.zeros((1, 40, 40), np.uint8),
            uv=np.array([[20., 20.], [25., 20.]], np.float32), chart=np.zeros(2, int),
            reference_xyz=w[:, 0], focal_px=np.array(100.), baseline_m=np.array(.06),
            cx=np.array(20.), cy=np.array(20.)))

    def test_unseen_space_is_unknown_not_filled_bounds(self):
        atlas = self.atlas()
        result = atlas.query(np.zeros(7), np.array([[0., 0., .8], [.05, 0., .8], [0., 0., .4]]))
        self.assertEqual(result.tolist(), ['predicted_surface', 'unknown', 'unknown'])

    def test_no_image_never_means_current_observation(self):
        result = self.atlas().evidence(np.zeros(7))
        np.testing.assert_array_equal(result['status'], [1, 1])
        np.testing.assert_array_equal(result['observed_xyz'], 0)

    def test_out_of_support_preserves_coordinates_and_admits_unknown(self):
        q = np.full(7, 3.)
        atlas = self.atlas()
        result = atlas.evidence(q)
        self.assertTrue(np.isfinite(result['xyz']).all())
        np.testing.assert_array_equal(result['status'], 0)
        self.assertEqual(atlas.query(q, result['xyz']).tolist(), ['unknown', 'unknown'])

    def test_same_state_moves_all_registered_samples_consistently(self):
        atlas = self.atlas()
        q = np.zeros(7)
        first = atlas.predict(q)
        q[0] = .5
        moved = atlas.predict(q)
        np.testing.assert_allclose(moved[:, 0]-first[:, 0], .05, atol=1e-7)
        np.testing.assert_allclose(moved[1]-moved[0], first[1]-first[0], atol=1e-7)

    def test_missing_nan_targets_cannot_pollute_ridge(self):
        rng = np.random.default_rng(8)
        q = rng.uniform(-1, 1, (100, 7))
        x = basis(q)
        truth = q[:, :3]*.03+np.array([0., 0., .8])
        valid = np.ones(100, bool)
        valid[::3] = False
        observed = truth.copy()
        observed[~valid] = np.nan
        w = ridge_fit(x, observed, valid)
        self.assertTrue(np.isfinite(w).all())
        self.assertLess(np.max(np.abs(x@w-truth)), .001)

    def test_discovery_rejects_static_and_time_only_response(self):
        rng = np.random.default_rng(9)
        q = rng.uniform(-1, 1, (220, 7)).astype(np.float32)
        xyz = np.zeros((220, 3, 3), np.float32)
        xyz[:, :, 2] = .8
        xyz[:, 0, :2] = q[:, :2]*.04
        xyz[:, 2, 0] = np.linspace(-.08, .08, 220)
        selected, _, _ = discover_surface(q, xyz, np.ones((220, 3), bool), np.zeros(7), np.ones(7))
        self.assertEqual(selected.tolist(), [0])

    def test_invalid_state_rejected(self):
        for q in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.atlas().predict(q)


class SurfaceNeighborhoodTests(unittest.TestCase):
    def support(self, xyz, valid):
        from embodiment.extent_cycle_v2 import neighborhood_support
        a = dict(chart=np.array([0]), references=np.array([0]),
            gray=np.zeros((1, 32, 32), np.uint8), uv=np.array([[16., 16.]]),
            reference_xyz=np.array([[0., 0., .8]]))
        metadata = dict(frames=[dict(file='a.npz')], calibration={})
        obs = dict(right=np.zeros((32, 32, 3), np.uint8))
        with patch('embodiment.extent_cycle_v2.read_json', return_value=metadata), \
             patch('embodiment.extent_cycle_v2.e7.load_npz', return_value=obs), \
             patch('embodiment.extent_cycle_v2.track_patch', return_value=(xyz, valid, np.zeros((5, 2)))):
            return neighborhood_support(a, np.array([0]))[0][0]

    def test_stable_local_stereo_surface_retained(self):
        xyz = np.array([[0.,0.,.8],[.002,0.,.8],[-.002,0.,.8],[0.,.002,.8],[0.,-.002,.8]])
        self.assertTrue(self.support(xyz, np.ones(5, bool)))

    def test_depth_boundary_rejected_despite_valid_center(self):
        xyz = np.array([[0.,0.,.8],[.002,0.,.8],[-.002,0.,.8],[0.,.002,1.1],[0.,-.002,1.1]])
        self.assertFalse(self.support(xyz, np.ones(5, bool)))

    def test_insufficient_stereo_neighbors_do_not_supply_confidence(self):
        xyz = np.tile([0.,0.,.8], (5,1))
        self.assertFalse(self.support(xyz, np.array([True,True,True,False,False])))


if __name__ == '__main__':
    unittest.main()
