import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from src.models.cnn import DigitCNN


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", run_id):
        raise ValueError("Use the run ID printed by the training command")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def atomic_file(path: Path, mode: str, **options):
    """Publish a file only after the new contents have been written and closed."""
    temporary = tempfile.NamedTemporaryFile(
        mode=mode, dir=path.parent, suffix=".tmp", delete=False, **options
    )
    temp_path = Path(temporary.name)
    try:
        with temporary:
            yield temporary
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_checkpoint(path: Path, model: DigitCNN, epoch: int) -> None:
    with atomic_file(path, "wb") as stream:
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, stream)


def load_checkpoint(
    path: Path, model: DigitCNN, device: torch.device
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint["epoch"]
