import unittest

from src.evaluation.convergence import summarize_convergence
from src.utils.config import ConvergenceConfig, config_from_mapping


def epoch_rows(*accuracies: float) -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "steps": str(10 * epoch),
            "elapsed_seconds": str(1.5 * epoch),
            "val_hard_sum_accuracy": str(accuracy),
        }
        for epoch, accuracy in enumerate(accuracies, start=1)
    ]


class ConvergenceTests(unittest.TestCase):
    def test_first_streak_resets_after_a_dip_and_accepts_exact_threshold(self) -> None:
        config = ConvergenceConfig(hard_sum_threshold=0.9, consecutive_epochs=2)
        result = summarize_convergence(epoch_rows(0.91, 0.88, 0.90, 0.92, 0.95), config)
        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["streak_start_epoch"], 3)
        self.assertEqual(result["epoch"], 4)
        self.assertEqual(result["steps"], 40)
        self.assertEqual(result["elapsed_seconds"], 6.0)

    def test_unfinished_streak_is_reported_as_not_reached(self) -> None:
        config = ConvergenceConfig(hard_sum_threshold=0.9, consecutive_epochs=2)
        result = summarize_convergence(epoch_rows(0.91, 0.89, 0.92), config)
        self.assertEqual(result["status"], "not_reached")
        self.assertIsNone(result["epoch"])
        self.assertIsNone(result["steps"])
        self.assertIsNone(result["elapsed_seconds"])

    def test_config_is_optional_and_rejects_invalid_values(self) -> None:
        self.assertEqual(summarize_convergence(epoch_rows(1.0), None), {"status": "not_configured"})
        self.assertIsNone(config_from_mapping({}).convergence)
        for threshold, epochs in ((1.1, 2), (float("nan"), 2), (0.9, 0), (True, 1)):
            with self.subTest(threshold=threshold, epochs=epochs), self.assertRaises(ValueError):
                config_from_mapping({
                    "convergence": {"hard_sum_threshold": threshold, "consecutive_epochs": epochs}
                })
