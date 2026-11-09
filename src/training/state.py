import random
from pathlib import Path

import numpy as np
import torch

from src.utils.artifacts import atomic_file


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generators: tuple[torch.Generator, ...],
    device: torch.device,
    rows: list[dict],
) -> None:
    numpy_state = np.random.get_state()
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rows": rows,
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "generator_states": [generator.get_state() for generator in generators],
    }
    with atomic_file(path, "wb") as stream:
        torch.save(state, stream)


def restore_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generators: tuple[torch.Generator, ...],
    device: torch.device,
) -> list[dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["python_rng"])
    algorithm, keys, *rest = state["numpy_rng"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), *rest))
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    for generator, saved in zip(generators, state["generator_states"], strict=True):
        generator.set_state(saved)
    return state["rows"]
