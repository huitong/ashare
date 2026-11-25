"""Target alignment and causal handling of Qlib's normalized fields."""

import unittest

import numpy as np
import torch

from ashare.joint_token import JointTokenizer, JointTransformer, daily_features, eligible_targets


class JointTokenTests(unittest.TestCase):
    def test_recovered_volume_and_trailing_only_feature(self):
        close = np.arange(1, 41, dtype=float)
        raw_volume = np.full(40, 100.0)
        raw_volume[22] = 200
        factor = np.linspace(0.1, 2, 40)
        cap = np.full(40, 1e6)
        features = daily_features(close, raw_volume / factor, factor, cap)
        np.testing.assert_allclose(features[22], [10, np.log(2), close[22] / close[21] - 1])
        truncated = daily_features(close[:23], raw_volume[:23] / factor[:23], factor[:23], cap[:23])
        np.testing.assert_allclose(features[:23], truncated, equal_nan=True)

    def test_missing_tokens_reset_context(self):
        actual = eligible_targets(np.array([[2, 4, 6, -1, 2, 4, 6, 8]]), context=2)
        self.assertEqual(np.flatnonzero(actual[0]).tolist(), [2, 6, 7])

    def test_token_order_preserves_return_bucket(self):
        tokenizer = JointTokenizer([list(range(1, 10))] * 3, list(range(10)))
        encoded = tokenizer.encode(np.array([[[2.2, 3.3, 4.4], [np.nan, 1, 1]]]))
        self.assertEqual(encoded.tolist(), [[234, -1]])

    def test_final_input_cannot_affect_earlier_training_positions(self):
        net = JointTransformer(context=6, width=16, heads=4).eval()
        a = torch.tensor([[10, 20, 30, 40, 50, 60]])
        b = a.clone()
        b[0, -1] = 999
        with torch.inference_mode():
            torch.testing.assert_close(net(a, tail=3)[:, :2], net(b, tail=3)[:, :2])

    def test_zero_residual_reproduces_markov_probabilities(self):
        net = JointTransformer(context=4, width=16, heads=4, markov_residual=True).eval()
        prior = torch.arange(1, 1001, dtype=torch.float32)
        prior /= prior.sum()
        net.log_transition_prior[123] = prior.log()
        with torch.inference_mode():
            actual = net(torch.tensor([[0, 1, 2, 123]])).softmax(-1)
        torch.testing.assert_close(actual[0, 0], prior)


if __name__ == "__main__":
    unittest.main()
