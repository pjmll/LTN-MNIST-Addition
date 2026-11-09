import argparse
import csv
import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets.mnist import load_train_validation
from src.evaluation.convergence import summarize_convergence
from src.evaluation.metrics import evaluate_validation
from src.logic.addition import AdditionRule
from src.models.addition import sum_log_probs
from src.models.cnn import DigitCNN
from src.training.state import restore_training_state, save_training_state
from src.utils.artifacts import atomic_file, save_checkpoint, validate_run_id, write_json
from src.utils.config import ROOT, ExperimentConfig, load_config, load_saved_config
from src.utils.runtime import resolve_device, seed_everything


def train_epoch(
    model: DigitCNN,
    batches: Iterable[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rule: AdditionRule | None = None,
    exists_p: int = 2,
) -> dict[str, float | int]:
    """Train on sums and accumulate optimization and conjunction diagnostics."""
    model.train()
    total_loss = samples = 0
    total_grad_norm = steps = zero_grad_steps = 0
    candidate_counts = torch.zeros(2, dtype=torch.long, device=device)
    for left, right, targets in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_pair(left.to(device), right.to(device))
        targets = targets.to(device)
        if rule is None:
            loss = F.nll_loss(sum_log_probs(*logits), targets)
        else:
            result = rule.evaluate(*logits, targets, exists_p=exists_p)
            loss = 1 - result.satisfaction
            candidate_counts[0] += result.active_candidates
            candidate_counts[1] += result.valid_candidates
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite sum loss")
        loss.backward()
        gradients = [param.grad for param in model.parameters() if param.grad is not None]
        grad_norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True).item()
        total_grad_norm += grad_norm
        zero_grad_steps += grad_norm == 0
        steps += 1
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        samples += targets.numel()
    stats = {
        "train_loss": total_loss / samples,
        "train_mean_grad_norm": total_grad_norm / steps,
        "train_zero_grad_fraction": zero_grad_steps / steps,
    }
    if rule is not None:
        active_count, valid_count = candidate_counts.tolist()
        stats.update(
            train_active_candidates=active_count,
            train_valid_candidates=valid_count,
            train_conjunction_activation_rate=active_count / valid_count,
        )
    return stats


METRIC_FIELDS = (
    "epoch", "steps", "elapsed_seconds", "train_seconds", "total_train_seconds",
    "train_loss", "train_satisfaction", "exists_p", "train_mean_grad_norm",
    "train_zero_grad_fraction", "train_active_candidates", "train_valid_candidates",
    "train_conjunction_activation_rate", "val_loss", "val_satisfaction",
    "val_hard_sum_accuracy", "val_prob_sum_accuracy", "val_digit_accuracy",
    "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
)


def write_metrics(path: Path, rows: list[dict]) -> None:
    with atomic_file(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train(
    config: ExperimentConfig, *, resume: str | None = None, stop_after: int | None = None
) -> str:
    stop = config.epochs if stop_after is None else stop_after
    if not 1 <= stop <= config.epochs:
        raise ValueError("--stop-after must be within the configured epoch budget")
    if resume is not None:
        validate_run_id(resume)
        run_id = resume
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_dir = ROOT / "experiments" / run_id
    checkpoint_dir = ROOT / "checkpoints" / run_id
    if resume is None:
        run_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
    elif not (checkpoint_dir / "latest.pt").is_file():
        raise ValueError(f"{run_id} has no latest.pt training state to resume from")
    with FileLock(checkpoint_dir / "training.lock", timeout=0):
        if resume is not None and config != load_saved_config(run_dir):
            raise ValueError("Resume requires the original effective configuration")
        _train(config, run_dir, checkpoint_dir, resume is not None, stop)
    return run_id


def _train(
    config: ExperimentConfig, run_dir: Path, checkpoint_dir: Path, resume: bool, stop: int
) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_set, val_set, train_pairs, val_pairs = load_train_validation(
        config.data_path,
        download=config.download,
        validation_images=config.validation_images,
        data_seed=config.data_seed,
        train_limit=config.train_pairs,
        validation_limit=config.validation_pairs,
        validation_digits=True,
    )
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    evaluation_rng = torch.Generator().manual_seed(config.seed)
    train_batches = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, generator=shuffle_rng
    )
    val_batches = DataLoader(
        val_set, batch_size=config.batch_size, generator=evaluation_rng
    )

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logic = config.ltn
    rule = (
        AdditionRule(logic.conjunction, forall_p=logic.forall_p, stable=logic.stable)
        if logic else None
    )
    train_indices = train_pairs.numpy().astype("<i8", copy=False)
    val_indices = val_pairs.numpy().astype("<i8", copy=False)
    initial_hash = hashlib.sha256()
    for value in model.state_dict().values():
        initial_hash.update(value.cpu().numpy().tobytes())
    metadata = {
        "objective": "ltn_satisfaction" if logic else "sum_probability_nll",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda": torch.version.cuda,
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "LTNtorch", "numpy", "matplotlib")
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "train_pairs": len(train_set),
        "validation_pairs": len(val_set),
        "train_pairs_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_pairs_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
        "initial_state_sha256": initial_hash.hexdigest(),
        "checkpoint_selection": "final_epoch",
    }
    print(
        f"Run {run_dir.name} on {device}; train pairs={len(train_set)}, "
        f"val pairs={len(val_set)}",
        flush=True,
    )
    latest = checkpoint_dir / "latest.pt"
    final = checkpoint_dir / "final.pt"
    generators = (shuffle_rng, evaluation_rng)
    if resume:
        if json.loads((run_dir / "run.json").read_text("utf-8")) != metadata:
            raise ValueError("Resume requires the original runtime, pairing and initialization")
        rows = restore_training_state(latest, model, optimizer, generators, device)
        if stop < len(rows):
            raise ValueError("--stop-after precedes the latest completed epoch")
    else:
        write_json(run_dir / "config.json", asdict(config))
        write_json(run_dir / "run.json", metadata)
        np.savez_compressed(run_dir / "pairs.npz", train=train_indices, validation=val_indices)
        save_checkpoint(checkpoint_dir / "initial.pt", model, epoch=0)
        rows = []
        save_training_state(latest, model, optimizer, generators, device, rows)

    metrics_path = run_dir / "metrics.csv"
    write_metrics(metrics_path, rows)
    elapsed = rows[-1]["elapsed_seconds"] if rows else 0
    total_train = rows[-1]["total_train_seconds"] if rows else 0
    memory = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if rows:
            memory = {
                field: rows[-1][field]
                for field in ("cuda_peak_allocated_mib", "cuda_peak_reserved_mib")
            }
    for epoch in range(len(rows) + 1, stop + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        p = logic.exists_p_at(epoch) if logic else None
        train_stats = train_epoch(
            model, train_batches, optimizer, device, rule=rule, exists_p=p or 2
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = round(perf_counter() - start, 6)
        total_train = round(total_train + train_seconds, 6)
        loss = train_stats["train_loss"]
        metrics = evaluate_validation(
            model, val_batches, device, rule=rule, exists_p=p or 2
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            for field, peak in (
                ("cuda_peak_allocated_mib", torch.cuda.max_memory_allocated(device)),
                ("cuda_peak_reserved_mib", torch.cuda.max_memory_reserved(device)),
            ):
                memory[field] = max(memory.get(field, 0), round(peak / 1024**2, 2))
        elapsed = round(elapsed + perf_counter() - start, 6)
        rows.append({
            "epoch": epoch,
            "steps": epoch * len(train_batches),
            "elapsed_seconds": elapsed,
            "train_seconds": train_seconds,
            "total_train_seconds": total_train,
            **train_stats,
            "train_satisfaction": 1 - loss if logic else None,
            "exists_p": p,
            **{f"val_{key}": value for key, value in metrics.items()},
            **memory,
        })
        # Commit state first. After an interruption, restore the CSV from committed rows.
        save_training_state(latest, model, optimizer, generators, device, rows)
        write_metrics(metrics_path, rows)
        logic_summary = f"  Sat={metrics['satisfaction']:.4f}  p={p}" if logic else ""
        print(
            f"Epoch {epoch}/{config.epochs}  train loss={loss:.4f}  "
            f"val loss={metrics['loss']:.4f}{logic_summary}  "
            f"val HardSum={metrics['hard_sum_accuracy']:.2%}  "
            f"val DigitAcc={metrics['digit_accuracy']:.2%}",
            flush=True,
        )
    if len(rows) == config.epochs and not final.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            convergence = summarize_convergence(csv.DictReader(stream), config.convergence)
        write_json(run_dir / "convergence.json", convergence)
        save_checkpoint(final, model, epoch=config.epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared CNN on MNIST sums.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--resume", help="Resume an existing run using its saved configuration.")
    parser.add_argument("--epochs", type=int, help="Override the run's epoch budget.")
    parser.add_argument("--seed", type=int, help="Override only the model and shuffle seed.")
    parser.add_argument("--stop-after", type=int, help="Pause after this absolute epoch.")
    args = parser.parse_args()
    try:
        if args.resume:
            if args.epochs is not None or args.seed is not None:
                parser.error("--resume uses the original epoch budget and seed")
            validate_run_id(args.resume)
            config = load_saved_config(ROOT / "experiments" / args.resume)
        else:
            config = load_config(args.config or ROOT / "configs" / "smoke.toml")
            if args.epochs is not None:
                config = replace(config, epochs=args.epochs)
            if args.seed is not None:
                config = replace(config, seed=args.seed)
        train(config, resume=args.resume, stop_after=args.stop_after)
    except (ValueError, FileNotFoundError, Timeout) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
