import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import src.training.train as training
from src.datasets.mnist import AdditionPairs, ValidationPairs, make_pairs
from src.utils.config import ExperimentConfig, LTNConfig


class ResumeTests(unittest.TestCase):
    def test_resume_recovers_csv_and_matches_continuous_training(self) -> None:
        config = ExperimentConfig(
            epochs=3,
            batch_size=4,
            device="cpu",
            ltn=LTNConfig("product", ((1, 1), (2, 3))),
        )
        rng = torch.Generator().manual_seed(7)
        images = torch.randint(256, (32, 28, 28), dtype=torch.uint8, generator=rng)
        digits = torch.arange(32) % 10
        train_pairs = make_pairs(torch.arange(24))
        val_pairs = make_pairs(torch.arange(24, 32))
        datasets = (
            AdditionPairs(images, digits, train_pairs),
            ValidationPairs(images, digits, val_pairs),
            train_pairs,
            val_pairs,
        )
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(training, "ROOT", root),
                patch.object(training, "load_train_validation", return_value=datasets),
            ):
                continuous = training.train(config)
                resumed = training.train(config, stop_after=1)
                log = root / "experiments" / resumed / "metrics.csv"
                first_row = log.read_text("utf-8").splitlines()[1]
                log.unlink()  # Simulate checkpoint publication without the CSV.
                self.assertEqual(training.train(config, resume=resumed), resumed)

            self.assertEqual(log.read_text("utf-8").splitlines()[1], first_row)
            rows = list(csv.DictReader(log.read_text("utf-8").splitlines()))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])
            self.assertEqual([row["exists_p"] for row in rows], ["1", "3", "3"])
            checkpoints = [
                torch.load(root / "checkpoints" / run / "final.pt", weights_only=True)
                for run in (continuous, resumed)
            ]
            torch.testing.assert_close(
                checkpoints[0]["model_state"], checkpoints[1]["model_state"], atol=0, rtol=0
            )
