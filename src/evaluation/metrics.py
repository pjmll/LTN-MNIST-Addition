from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from src.logic.addition import AdditionRule
from src.models.addition import sum_log_probs
from src.models.cnn import DigitCNN


def sum_correct_counts(
    left_logits: Tensor, right_logits: Tensor, targets: Tensor
) -> tuple[int, int]:
    """Return correct counts for HardSum and ProbSum in that order."""
    hard = left_logits.argmax(dim=-1) + right_logits.argmax(dim=-1)
    probabilistic = sum_log_probs(left_logits, right_logits).argmax(dim=-1)
    return int((hard == targets).sum()), int((probabilistic == targets).sum())


def digit_correct_count(logits: Tensor, targets: Tensor) -> int:
    return int((logits.argmax(dim=-1) == targets).sum())


@dataclass
class PairMetrics:
    count: int = 0
    hard_correct: int = 0
    prob_correct: int = 0

    def update(self, left_logits: Tensor, right_logits: Tensor, targets: Tensor) -> None:
        hard, prob = sum_correct_counts(left_logits, right_logits, targets)
        self.count += targets.numel()
        self.hard_correct += hard
        self.prob_correct += prob

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("Sum accuracy requires at least one pair")
        return {
            "hard_sum_accuracy": self.hard_correct / self.count,
            "prob_sum_accuracy": self.prob_correct / self.count,
        }


@torch.inference_mode()
def evaluate_pairs(
    model: DigitCNN, batches: Iterable[tuple[Tensor, Tensor, Tensor]], device: torch.device
) -> dict[str, float]:
    model.eval()
    metrics = PairMetrics()
    for left, right, targets in batches:
        logits = model.forward_pair(left.to(device), right.to(device))
        metrics.update(*logits, targets.to(device))
    return metrics.compute()


@torch.inference_mode()
def evaluate_validation(
    model: DigitCNN,
    batches: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
    *,
    rule: AdditionRule | None,
    exists_p: int,
) -> dict[str, float]:
    model.eval()
    metrics = PairMetrics()
    total_loss = digits_correct = 0
    for left, right, targets, digits in batches:
        logits = model.forward_pair(left.to(device), right.to(device))
        targets, digits = targets.to(device), digits.to(device)
        metrics.update(*logits, targets)
        digits_correct += digit_correct_count(logits[0], digits[:, 0])
        digits_correct += digit_correct_count(logits[1], digits[:, 1])
        if rule is None:
            loss = F.nll_loss(sum_log_probs(*logits), targets)
        else:
            loss = 1 - rule.satisfaction(*logits, targets, exists_p=exists_p)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total_loss += loss.item() * targets.numel()
    result = metrics.compute()
    loss = total_loss / metrics.count
    result.update(loss=loss, digit_accuracy=digits_correct / (2 * metrics.count))
    if rule is not None:
        result["satisfaction"] = 1 - loss
    return result


def score_test_predictions(
    logits: Tensor, digits: Tensor, pair_indices: Tensor
) -> dict[str, int | float]:
    """Use one digit prediction per image for both per-image and per-pair metrics."""
    pair_digits = digits[pair_indices]
    left, right = logits[pair_indices].unbind(dim=1)
    pairs = PairMetrics()
    pairs.update(left, right, pair_digits.sum(dim=1))
    errors = logits.argmax(dim=-1)[pair_indices] - pair_digits
    cancelled = int(((errors.sum(dim=1) == 0) & (errors != 0).any(dim=1)).sum())
    return {
        **pairs.compute(),
        "digit_accuracy": digit_correct_count(logits, digits) / digits.numel(),
        "error_cancelled_pairs": cancelled,
        "error_cancellation_rate": cancelled / pairs.count,
    }


@torch.inference_mode()
def evaluate_test(
    model: DigitCNN,
    batches: Iterable[tuple[Tensor, Tensor]],
    pair_indices: Tensor,
    device: torch.device,
) -> dict[str, int | float]:
    """Infer test images in index order, matching the stored pair table."""
    model.eval()
    scores, labels = [], []
    for images, targets in batches:
        scores.append(model(images.to(device)))
        labels.append(targets.to(device))
    if not scores:
        raise ValueError("Test evaluation requires at least one image")
    return score_test_predictions(
        torch.cat(scores), torch.cat(labels), pair_indices.to(device)
    )
