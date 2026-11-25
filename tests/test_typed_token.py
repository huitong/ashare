import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ashare.training_log import write_json
from ashare.typed_data import MAIN, STAR, board, limits, st_from_price, symbol_features
from ashare.typed_token import CORE, TransitionOrder, TypedTransformer, cross_section_loss, split_days
from ashare.web import Explorer


class TypedTokenTests(unittest.TestCase):
    def test_st_proxy_is_causal_and_distinguishes_board(self):
        self.assertEqual(board("sh688009"), STAR)
        close = np.array([20, 21, 22.05, 23.15])
        dates = np.array(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
        comparable = np.array([False, True, True, True])
        proxy = st_from_price(close, close, close, comparable, dates)
        self.assertEqual(proxy.tolist(), [False, False, True, True])
        self.assertEqual(st_from_price(close[:3], close[:3], close[:3], comparable[:3], dates[:3]).tolist(),
                         proxy[:3].tolist())

    def test_verified_limit_overrides_default_and_blocks_open(self):
        date = np.array(["2026-11-20", "2026-11-23"])
        raw = np.array([[10,10,10,10,1,1000,1e7], [11,11,11,11,1,1000,1e7]], dtype=float)
        status = np.array([[np.nan,np.nan,np.nan], [1, 11, 9]])
        st, inferred, up, _, official, _, blocked = limits(raw,date,MAIN,"2020-01-01",status)
        self.assertEqual(st.tolist(), [False,True])
        self.assertEqual(official.tolist(), [False,True])
        self.assertTrue(up[-1] and blocked[-1] and not inferred[-1])
        status[-1] = (1, 12, 8)
        self.assertFalse(limits(raw,date,MAIN,"2020-01-01",status)[-1][-1])

    def test_cap_carry_is_limited_and_next_open_target_is_shifted(self):
        close = 10 + np.arange(30) * 0.04
        raw = np.column_stack((close, close, close, close, np.ones(30), np.full(30, 1e4), np.full(30, 7e5)))
        raw[22:26, -1] = np.nan
        feature, _, _, age, actual = symbol_features(raw)
        np.testing.assert_allclose(age[22:25], [1/3,2/3,1])
        self.assertTrue(np.isfinite(feature[24]).all())
        self.assertTrue(np.isnan(feature[25]).all())
        self.assertAlmostEqual(actual[22], close[23]/close[22]-1)

    def test_sparse_markov_backs_off_for_an_unseen_context(self):
        order = TransitionOrder(np.array([3,3,3]), np.array([1,1,2]), np.ones(5000), strength=3)
        prior = np.array([[0.25,0.3,0.5],[0.25,0.3,0.5]])
        predicted = order.step(np.array([3,4]), np.array([1,1]), prior)
        np.testing.assert_allclose(predicted[1], prior[1])
        self.assertAlmostEqual(predicted[0,0], (2 + 3 * 0.25)/6)

    def test_rank_loss_prefers_strong_tradable_stocks(self):
        target = torch.arange(20, dtype=torch.float)/100
        target[8] = float("nan")
        ordered = torch.arange(20, dtype=torch.float)/4
        self.assertLess(cross_section_loss(ordered,target), cross_section_loss(-ordered,target))

    def test_model_has_six_causal_blocks_and_five_type_outputs(self):
        model = TypedTransformer(context=8).eval()
        aux = torch.zeros(2, 8, 10)
        tokens = torch.full((2,8), 4000, dtype=torch.long)
        with torch.inference_mode():
            market, group, rank = model(tokens, aux, tail=4)
            edited = tokens.clone()
            edited[:,-1] = 20
            other, _, _ = model(edited, aux, tail=4)
        self.assertEqual(len(model.blocks.layers), 6)
        self.assertEqual(market.shape, (2,4,CORE))
        self.assertEqual(group.shape, (2,4,5))
        self.assertEqual(rank.shape,(2,))
        torch.testing.assert_close(other[:,:-1], market[:,:-1])

    def test_training_json_read_only_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/"run-a"/"progress.json", {"run":"run-a","status":"training","started_at":"a"})
            explorer = Explorer("unused", "unused", root)
            self.assertEqual(explorer.dispatch("/api/training", {})[0]["run"], "run-a")
            self.assertEqual(explorer.dispatch("/api/training", {"run":"run-a"})["status"], "training")
            with self.assertRaises(ValueError):
                explorer.dispatch("/api/training", {"run":"../other"})


if __name__ == "__main__":
    unittest.main()
