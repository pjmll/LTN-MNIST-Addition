import unittest

import torch

from src.evaluation.metrics import (
    PairMetrics,
    digit_correct_count,
    score_test_predictions,
    sum_correct_counts,
)
from src.models.addition import sum_log_probs


class MetricTests(unittest.TestCase):
    def test_hard_and_probabilistic_decoders_can_disagree(self) -> None:
        scores = torch.tensor([[0.6, 0.4] + [0.0] * 8]).log()
        self.assertEqual(sum_correct_counts(scores, scores, torch.tensor([0])), (1, 0))
        self.assertEqual(sum_correct_counts(scores, scores, torch.tensor([1])), (0, 1))

    def test_exact_distribution_matches_explicit_enumeration(self) -> None:
        left = torch.arange(20, dtype=torch.float64).reshape(2, 10)
        right = -left
        result = sum_log_probs(left, right).exp()
        expected = torch.zeros((2, 19), dtype=torch.float64)
        left_probs, right_probs = left.softmax(-1), right.softmax(-1)
        for a in range(10):
            for b in range(10):
                expected[:, a + b] += left_probs[:, a] * right_probs[:, b]
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.sum(-1), torch.ones(2, dtype=result.dtype))

    def test_uneven_batches_are_weighted_by_sample_count(self) -> None:
        zeros = torch.nn.functional.one_hot(torch.zeros(3, dtype=torch.long), 10).float()
        scores = zeros.log()
        metrics = PairMetrics()
        metrics.update(scores[:2], scores[:2], torch.tensor([0, 0]))
        metrics.update(scores[2:], scores[2:], torch.tensor([1]))
        self.assertEqual(
            metrics.compute(), {"hard_sum_accuracy": 2 / 3, "prob_sum_accuracy": 2 / 3}
        )

    def test_digit_correct_count(self) -> None:
        scores = torch.eye(10)[[0, 3, 9]]
        self.assertEqual(digit_correct_count(scores, torch.tensor([0, 2, 9])), 2)

    def test_error_cancellation_uses_all_pairs_as_denominator(self) -> None:
        # The first pair has errors (+1, -1), the second is fully correct,
        # and the third has a wrong sum.
        digits = torch.tensor([3, 6, 2, 4, 8, 0])
        logits = 10 * torch.eye(10)[[4, 5, 2, 4, 9, 0]]
        indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
        result = score_test_predictions(logits, digits, indices)
        self.assertEqual(result["hard_sum_accuracy"], 2 / 3)
        self.assertEqual(result["digit_accuracy"], 1 / 2)
        self.assertEqual(result["error_cancelled_pairs"], 1)
        self.assertEqual(result["error_cancellation_rate"], 1 / 3)

    def test_empty_metrics_raise(self) -> None:
        with self.assertRaises(ValueError):
            PairMetrics().compute()
