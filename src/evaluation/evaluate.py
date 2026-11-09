import argparse
import json
import re

import torch
from torch.utils.data import DataLoader

from src.datasets.mnist import load_test
from src.evaluation.metrics import evaluate_test
from src.models.cnn import DigitCNN
from src.utils.artifacts import load_checkpoint, write_json
from src.utils.config import ROOT, config_from_mapping
from src.utils.runtime import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final checkpoint on MNIST test.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", args.run_id):
        parser.error("--run-id must be the ID printed by the training command")

    run_dir = ROOT / "experiments" / args.run_id
    config = config_from_mapping(json.loads((run_dir / "config.json").read_text("utf-8")))
    seed_everything(config.seed)
    device = resolve_device(args.device or config.device)
    model = DigitCNN().to(device)
    epoch = load_checkpoint(ROOT / "checkpoints" / args.run_id / "final.pt", model, device)
    digits, pair_indices = load_test(config.data_path, download=config.download)
    rng = torch.Generator().manual_seed(config.seed)
    digit_batches = DataLoader(digits, batch_size=config.batch_size, generator=rng)
    metrics = {
        "checkpoint_epoch": epoch,
        "device": str(device),
        "test_pairs": len(pair_indices),
        "test_digits": len(digits),
        **evaluate_test(model, digit_batches, pair_indices, device),
    }
    write_json(run_dir / "test_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
