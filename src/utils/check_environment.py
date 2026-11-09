import argparse
import platform
import sys
from importlib.metadata import version

import ltn
import torch
from torch import nn
from torchvision.ops import nms

from src.utils.runtime import seed_everything


def check_ltn(device: torch.device) -> None:
    """Check predicate, conjunction, quantification, and gradient on the device."""
    previous_device = ltn.device
    ltn.device = device
    try:
        features = ltn.Variable("x", torch.tensor([[0.2], [0.4], [0.8]], device=device))
        model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Sigmoid()).to(device)
        nn.init.constant_(model[0].weight, 0.8)
        predicate = ltn.Predicate(model)
        exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        for op in (ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.AndLuk()):
            model.zero_grad(set_to_none=True)
            truth = predicate(features)
            conjunction = ltn.Connective(op)(truth, truth)
            score = exists(features, conjunction).value
            (1 - score).backward()
            grad = model[0].weight.grad
            if not (0 < float(score.detach()) < 1):
                raise RuntimeError(f"Invalid LTN satisfaction on {device}: {score}")
            if grad is None or not torch.isfinite(grad).all() or not grad.abs().any():
                raise RuntimeError(f"Invalid LTN gradient on {device}: {grad}")
            print(f"{device} {type(op).__name__}: satisfaction={score.item():.6f}, gradient OK")
    finally:
        ltn.device = previous_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Python, PyTorch, torchvision and LTNtorch.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    for name in ("torch", "torchvision", "LTNtorch", "numpy", "matplotlib"):
        print(f"{name}: {version(name)}")
    seed_everything(0)
    cuda = torch.cuda.is_available()
    print(f"PyTorch CUDA runtime: {torch.version.cuda}, available: {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name()}, cuDNN: {torch.backends.cudnn.version()}")
    if args.require_cuda and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    devices = [torch.device("cpu")]
    if cuda:
        devices.append(torch.device("cuda"))
    for device in devices:
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=device)
        if nms(boxes, torch.ones(1, device=device), 0.5).tolist() != [0]:
            raise RuntimeError(f"torchvision native operation failed on {device}")
        check_ltn(device)


if __name__ == "__main__":
    main()
