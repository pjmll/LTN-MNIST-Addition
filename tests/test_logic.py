import json
import unittest
from dataclasses import asdict

import torch

from src.logic.addition import AdditionRule
from src.utils.config import ROOT, LTNConfig, config_from_mapping, load_config


class AdditionRuleTests(unittest.TestCase):
    def test_all_sums_match_independent_guarded_aggregation(self) -> None:
        sums = torch.arange(19)
        scores = torch.arange(190, dtype=torch.float32).reshape(19, 10)
        left, right = 4 * scores.sin(), 4 * scores.cos()
        u, v = left.softmax(-1), right.softmax(-1)
        eps = 1e-4
        for conjunction in ("product", "lukasiewicz"):
            rule = AdditionRule(conjunction, forall_p=3, stable=True)
            for p in (1, 2, 6):
                with self.subTest(conjunction=conjunction, p=p):
                    per_pair = []
                    for z, (u_row, v_row) in enumerate(zip(u, v)):
                        valid = [(a, z - a) for a in range(10) if 0 <= z - a <= 9]
                        self.assertEqual(len(valid), min(z + 1, 19 - z))
                        values = []
                        for a, b in valid:
                            if conjunction == "product":
                                truth = ((1 - eps) * u_row[a] + eps) * (
                                    (1 - eps) * v_row[b] + eps
                                )
                            else:
                                truth = (u_row[a] + v_row[b] - 1).clamp_min(0)
                            values.append((1 - eps) * truth + eps)
                        per_pair.append(torch.stack(values).pow(p).mean().pow(1 / p))
                    errors = 1 - (1 - eps) * torch.stack(per_pair).double()
                    expected = 1 - errors.pow(3).mean().pow(1 / 3)
                    actual = rule.satisfaction(left, right, sums, exists_p=p)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)

    def test_product_propagates_gradients_to_both_positions(self) -> None:
        rule = AdditionRule("product", forall_p=2, stable=True)
        left = torch.zeros((1, 10), requires_grad=True)
        right = torch.zeros((1, 10), requires_grad=True)
        loss = 1 - rule.satisfaction(left, right, torch.tensor([18]), exists_p=2)
        loss.backward()
        for tensor in (left, right):
            self.assertTrue(tensor.grad.isfinite().all())
            self.assertGreater(tensor.grad[0, :9].sum().item(), 0)
            self.assertLess(tensor.grad[0, 9].item(), 0)

    def test_lukasiewicz_keeps_native_zero_gradient_at_uniform_logits(self) -> None:
        rule = AdditionRule("lukasiewicz", forall_p=2, stable=True)
        logits = torch.zeros((2, 10), requires_grad=True)
        loss = 1 - rule.satisfaction(logits, logits, torch.tensor([0, 18]), exists_p=2)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_activation_counts_exclude_invalid_pairs_and_zero_truth(self) -> None:
        probabilities = torch.zeros((3, 10))
        probabilities[:, :2] = torch.tensor([[0.75, 0.25], [0.5, 0.5], [1.0, 0.0]])
        right = probabilities.clone()
        right[0, :2] = torch.tensor([0.25, 0.75])
        sums = torch.tensor([1, 0, 18])
        for conjunction, expected in (("product", 4), ("lukasiewicz", 1)):
            with self.subTest(conjunction=conjunction):
                rule = AdditionRule(conjunction, forall_p=2, stable=True)
                result = rule.evaluate(probabilities.log(), right.log(), sums, exists_p=2)
                self.assertEqual(int(result.valid_candidates), 4)
                self.assertEqual(int(result.active_candidates), expected)

    def test_invalid_sum_targets_are_rejected(self) -> None:
        rule = AdditionRule("product", forall_p=2, stable=True)
        logits = torch.zeros((1, 10))
        for targets in (torch.tensor([-1]), torch.tensor([19]), torch.tensor([1.0])):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                rule.satisfaction(logits, logits, targets, exists_p=2)


class LTNConfigTests(unittest.TestCase):
    def test_six_groups_only_vary_conjunction_and_exists_p(self) -> None:
        epochs = (1, 4, 5, 8, 9, 12, 13, 20)
        schedules = {
            "1": [1] * 8,
            "2": [2] * 8,
            "S": [1, 1, 2, 2, 4, 4, 6, 6],
        }
        baseline = None
        for prefix, conjunction in (("P", "product"), ("L", "lukasiewicz")):
            for suffix, expected in schedules.items():
                with self.subTest(group=prefix + suffix):
                    config = load_config(ROOT / "configs" / "ltn" / f"{prefix}{suffix}.toml")
                    logic = config.ltn
                    self.assertEqual(logic.conjunction, conjunction)
                    self.assertEqual([logic.exists_p_at(epoch) for epoch in epochs], expected)
                    self.assertEqual(logic.forall_p, 2)
                    self.assertTrue(logic.stable)
                    raw = asdict(config)
                    self.assertEqual(config_from_mapping(json.loads(json.dumps(raw))), config)
                    raw.pop("ltn")
                    if baseline is None:
                        baseline = raw
                    self.assertEqual(raw, baseline)

    def test_invalid_schedule_is_rejected(self) -> None:
        for schedule in ((), ((2, 1),), ((1, 2), (1, 4)), ((1, 0),)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                LTNConfig(conjunction="product", exists_schedule=schedule)
