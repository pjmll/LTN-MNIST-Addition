from dataclasses import dataclass

import ltn
import torch
from torch import Tensor


def _digit_truth(probabilities: Tensor, digit: Tensor) -> Tensor:
    return probabilities.gather(1, digit)


def _addition_guard(a: ltn.LTNObject, b: ltn.LTNObject, z: ltn.LTNObject) -> Tensor:
    return a.value + b.value == z.value


@dataclass(frozen=True)
class AdditionBatch:
    satisfaction: Tensor
    active_candidates: Tensor
    valid_candidates: Tensor


class AdditionRule:
    """Evaluate forall (x,y,z): exists a,b with a+b=z: Digit(x,a) and Digit(y,b)."""

    def __init__(self, conjunction: str, *, forall_p: int, stable: bool) -> None:
        operators = {
            "product": ltn.fuzzy_ops.AndProd(stable=stable),
            "lukasiewicz": ltn.fuzzy_ops.AndLuk(),
        }
        self._digit = ltn.Predicate(func=_digit_truth)
        self._and = ltn.Connective(operators[conjunction])
        self._exists = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMean(stable=stable), quantifier="e"
        )
        self._forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=forall_p, stable=stable), quantifier="f"
        )

    def satisfaction(
        self, left_logits: Tensor, right_logits: Tensor, sums: Tensor, *, exists_p: int
    ) -> Tensor:
        return self.evaluate(left_logits, right_logits, sums, exists_p=exists_p).satisfaction

    def evaluate(
        self, left_logits: Tensor, right_logits: Tensor, sums: Tensor, *, exists_p: int
    ) -> AdditionBatch:
        """Ground Digit from shared CNN logits, with the sum as the only target."""
        if left_logits.shape != (sums.numel(), 10) or left_logits.shape != right_logits.shape:
            raise ValueError("Expected two [B, 10] logits tensors and B sum targets")
        if sums.ndim != 1 or sums.dtype != torch.long or sums.numel() == 0:
            raise ValueError("Expected a nonempty int64 tensor of shape [B] for sum targets")
        if ((sums < 0) | (sums > 18)).any():
            raise ValueError("Sum targets must be in [0, 18] so the guard has candidates")
        if right_logits.device != left_logits.device or sums.device != left_logits.device:
            raise ValueError("Both logits tensors and sum targets must be on the same device")

        # LTNtorch 1.0.2 moves newly created Variables to its module-level device.
        previous_device = ltn.device
        ltn.device = left_logits.device
        try:
            x = ltn.Variable("x", left_logits.softmax(dim=-1))
            y = ltn.Variable("y", right_logits.softmax(dim=-1))
            z = ltn.Variable("z", sums)
            digits = torch.arange(10, device=sums.device)
            a = ltn.Variable("a", digits)
            b = ltn.Variable("b", digits)
        finally:
            ltn.device = previous_device

        # Align x, y and z before broadcasting the digit candidates.
        pairs = ltn.diag(x, y, z)
        joint = self._and(self._digit(x, a), self._digit(y, b))
        # Count on the raw conjunction, before the existential aggregator's stability shift.
        axis_order = [joint.free_vars.index(var.free_vars[0]) for var in (z, a, b)]
        truth = joint.value.detach().permute(axis_order)
        valid = digits[:, None] + digits == sums[:, None, None]
        active_count = ((truth > 0) & valid).sum()
        valid_count = valid.sum()
        per_pair = self._exists(
            [a, b], joint, cond_vars=[a, b, z], cond_fn=_addition_guard, p=exists_p
        )
        return AdditionBatch(self._forall(pairs, per_pair).value, active_count, valid_count)
