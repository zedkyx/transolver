import unittest

import torch

from utils.normalizer import UnitTransformer, weighted_channel_stats


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

if __name__ == "__main__":
    unittest.main()
