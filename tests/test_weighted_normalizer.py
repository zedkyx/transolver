import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils.normalizer import UnitTransformer, weighted_channel_stats
from utils.quadrature import lumped_area_delaunay, lumped_area_delaunay_batch


class WeightedChannelStatsTest(unittest.TestCase):
    def test_known_function_space_statistics(self):
        x = torch.tensor(
            [
                [[0.0], [10.0], [10.0]],
                [[20.0], [20.0], [30.0]],
            ]
        )
        q = torch.tensor([[8.0, 1.0, 1.0], [1.0, 1.0, 8.0]])
        mean, std = weighted_channel_stats(x, q)
        self.assertTrue(torch.allclose(mean, torch.tensor([[[15.0]]])))
        self.assertTrue(torch.allclose(std.square(), torch.tensor([[[185.0]]])))

    def test_padding_has_zero_contribution(self):
        x = torch.tensor([[[1.0], [3.0], [1.0e6]]])
        q = torch.ones(1, 3)
        mask = torch.tensor([[True, True, False]])
        mean, std = weighted_channel_stats(x, q, mask=mask)
        self.assertTrue(torch.allclose(mean, torch.tensor([[[2.0]]])))
        self.assertTrue(torch.allclose(std, torch.tensor([[[1.0]]])))

    def test_each_sample_has_equal_total_weight(self):
        x = torch.tensor([[[0.0], [0.0]], [[10.0], [10.0]]])
        q = torch.tensor([[1000.0, 1000.0], [0.01, 0.01]])
        mean, _ = weighted_channel_stats(x, q)
        self.assertTrue(torch.allclose(mean, torch.tensor([[[5.0]]])))

    def test_legacy_path_is_unchanged(self):
        torch.manual_seed(7)
        x = torch.randn(4, 5, 3)
        normalizer = UnitTransformer(x)
        self.assertTrue(torch.equal(normalizer.mean, x.mean(dim=(0, 1), keepdim=True)))
        self.assertTrue(
            torch.equal(normalizer.std, x.std(dim=(0, 1), keepdim=True) + 1e-8)
        )

    def test_encode_decode_round_trip(self):
        x = torch.tensor([[[1.0], [4.0], [9.0]]])
        q = torch.tensor([[0.6, 0.3, 0.1]])
        normalizer = UnitTransformer(x, node_weight=q)
        self.assertTrue(torch.allclose(normalizer.decode(normalizer.encode(x)), x))

    def test_delaunay_weights_are_affine_scale_equivalent(self):
        # Raw physical coordinates on a 10 x 2 rectangle.
        pos_raw = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 2.0],
                [10.0, 2.0],
                [4.0, 0.8],
                [7.0, 1.5],
            ],
            dtype=torch.float64,
        )
        # Per-dimension min-max normalization maps the domain to [0, 1]^2.
        pos_norm = (pos_raw - pos_raw.amin(dim=0)) / (
            pos_raw.amax(dim=0) - pos_raw.amin(dim=0)
        )

        q_raw = torch.from_numpy(lumped_area_delaunay(pos_raw.numpy()))
        q_norm = torch.from_numpy(lumped_area_delaunay(pos_norm.numpy()))

        # Under an affine diagonal scaling, every triangle area is multiplied by
        # the same Jacobian determinant. Therefore nodal weights differ only by
        # the physical area scale: 10 * 2 = 20.
        self.assertTrue(torch.allclose(q_raw, q_norm * 20.0, rtol=1e-6, atol=1e-10))

        # Because weighted_channel_stats normalizes q independently per sample,
        # this global scale factor does not change mean/std.
        field = torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]],
            dtype=torch.float64,
        )
        mean_raw, std_raw = weighted_channel_stats(field, q_raw.unsqueeze(0))
        mean_norm, std_norm = weighted_channel_stats(field, q_norm.unsqueeze(0))
        self.assertTrue(torch.allclose(mean_raw, mean_norm, rtol=1e-6, atol=1e-10))
        self.assertTrue(torch.allclose(std_raw, std_norm, rtol=1e-6, atol=1e-10))

    def test_max_edge_threshold_depends_on_coordinate_scale(self):
        # Same four physical points, but raw coordinates are a 10 x 1 rectangle.
        # max_edge is interpreted in the coordinate system used for Delaunay.
        pos_raw = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 1.0],
                [10.0, 1.0],
            ],
            dtype=torch.float64,
        )
        pos_norm = (pos_raw - pos_raw.amin(dim=0)) / (
            pos_raw.amax(dim=0) - pos_raw.amin(dim=0)
        )

        q_raw = torch.from_numpy(lumped_area_delaunay(pos_raw.numpy(), max_edge=2.0))
        q_norm = torch.from_numpy(lumped_area_delaunay(pos_norm.numpy(), max_edge=2.0))

        # In raw coordinates all Delaunay triangles contain a long edge > 2.0
        # and are filtered out, leaving only eps fallback weights.
        self.assertLess(float(q_raw.sum()), 1e-8)

        # In normalized coordinates the same max_edge=2.0 keeps the unit-square
        # triangles, so weights represent unit-area geometry.
        self.assertGreater(float(q_norm.sum()), 0.99)
        self.assertFalse(torch.allclose(q_raw / q_raw.sum(), q_norm / q_norm.sum()))

    def test_intrinsic_1d_points_use_interval_weights(self):
        # Points can be stored as [x, 0] but are intrinsically 1D.
        pos = torch.tensor(
            [[[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]]],
            dtype=torch.float64,
        )
        q = torch.from_numpy(lumped_area_delaunay_batch(pos.numpy(), show_progress=False))
        expected = torch.tensor([[0.125, 0.5, 0.375]], dtype=torch.float32)
        self.assertTrue(torch.allclose(q, expected, rtol=1e-6, atol=1e-8))

        q_cut = torch.from_numpy(
            lumped_area_delaunay_batch(pos.numpy(), max_edge=0.5, show_progress=False)
        )
        expected_cut = torch.tensor([[0.125, 0.125, 1e-12]], dtype=torch.float32)
        self.assertTrue(torch.allclose(q_cut, expected_cut, rtol=1e-6, atol=1e-8))

if __name__ == "__main__":
    unittest.main()
