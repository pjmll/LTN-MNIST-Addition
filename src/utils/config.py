import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "A1": ("product", 1),
    "A2": ("product", 2),
    "A3": ("product", None),
    "B1": ("lukasiewicz", 1),
    "B2": ("lukasiewicz", 2),
    "B3": ("lukasiewicz", None),
}


@dataclass(frozen=True)
class ConvergenceConfig:
    hard_sum_threshold: float
    consecutive_epochs: int

    def __post_init__(self) -> None:
        threshold = self.hard_sum_threshold
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("hard_sum_threshold must be a number between 0 and 1")
        if type(self.consecutive_epochs) is not int or self.consecutive_epochs < 1:
            raise ValueError("consecutive_epochs must be a positive integer")


@dataclass(frozen=True)
class LTNConfig:
    conjunction: str
    exists_schedule: tuple[tuple[int, int], ...]
    forall_p: int = 2
    stable: bool = True

    def __post_init__(self) -> None:
        if self.conjunction not in ("product", "lukasiewicz"):
            raise ValueError("conjunction must be product or lukasiewicz")
        if type(self.forall_p) is not int or self.forall_p < 1:
            raise ValueError("forall_p must be a positive integer")
        if type(self.stable) is not bool:
            raise ValueError("stable must be true or false")
        if not isinstance(self.exists_schedule, (list, tuple)) or not self.exists_schedule:
            raise ValueError("exists_schedule must contain at least one [epoch, p] pair")
        schedule = []
        for stage in self.exists_schedule:
            if not isinstance(stage, (list, tuple)) or len(stage) != 2:
                raise ValueError("Each exists_schedule entry must be [epoch, p]")
            start, p = stage
            if type(start) is not int or type(p) is not int or start < 1 or p < 1:
                raise ValueError("Each schedule epoch and p must be positive integers")
            if schedule and start <= schedule[-1][0]:
                raise ValueError("Schedule start epochs must be strictly increasing")
            schedule.append((start, p))
        if schedule[0][0] != 1:
            raise ValueError("exists_schedule must start at epoch 1")
        object.__setattr__(self, "exists_schedule", tuple(schedule))

    def exists_p_at(self, epoch: int) -> int:
        if epoch < 1:
            raise ValueError("Epoch numbering starts at 1")
        return next(p for start, p in reversed(self.exists_schedule) if epoch >= start)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    data_seed: int = 2026
    data_root: str = "data"
    download: bool = True
    validation_images: int = 6000
    train_pairs: int | None = None
    validation_pairs: int | None = None
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "auto"
    ltn: LTNConfig | None = None
    convergence: ConvergenceConfig | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        for name in ("seed", "data_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        for name in ("epochs", "batch_size", "validation_images"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("train_pairs", "validation_pairs"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer when specified")
        if self.validation_images % 2 or self.validation_images >= 60000:
            raise ValueError("validation_images must be even and below 60000")
        if type(self.learning_rate) not in (int, float) or not (
            math.isfinite(self.learning_rate) and self.learning_rate > 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.data_root, str) or not self.data_root.strip():
            raise ValueError("data_root must be a nonempty path")
        if type(self.download) is not bool:
            raise ValueError("download must be true or false")
        if self.group is not None:
            if self.group not in GROUPS or self.ltn is None:
                raise ValueError("group must identify one of the six LTN treatments")
            conjunction, fixed_p = GROUPS[self.group]
            schedule = self.ltn.exists_schedule
            if self.ltn.conjunction != conjunction:
                raise ValueError(f"{self.group} requires {conjunction}")
            if fixed_p is not None and schedule != ((1, fixed_p),):
                raise ValueError(f"{self.group} requires fixed p={fixed_p}")
            if fixed_p is None and (
                len(schedule) < 2
                or any(left[1] >= right[1] for left, right in zip(schedule, schedule[1:]))
            ):
                raise ValueError(f"{self.group} requires strictly increasing p stages")

    @property
    def data_path(self) -> Path:
        return ROOT / self.data_root


def config_from_mapping(values: Mapping[str, object]) -> ExperimentConfig:
    settings = dict(values)
    logic = settings.pop("ltn", None)
    convergence = settings.pop("convergence", None)
    if logic is not None and not isinstance(logic, dict):
        raise ValueError("ltn must be a table")
    if convergence is not None and not isinstance(convergence, dict):
        raise ValueError("convergence must be a table")
    return ExperimentConfig(
        **settings,
        ltn=LTNConfig(**logic) if logic is not None else None,
        convergence=ConvergenceConfig(**convergence) if convergence is not None else None,
    )


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    base = settings.pop("base", None)
    if base is not None:
        with (path.parent / base).open("rb") as stream:
            shared = tomllib.load(stream)
        if "ltn" in shared and "ltn" in settings:
            settings["ltn"] = {**shared["ltn"], **settings["ltn"]}
        settings = {**shared, **settings}
    return config_from_mapping(settings)


def load_saved_config(run_dir: Path) -> ExperimentConfig:
    return config_from_mapping(json.loads((run_dir / "config.json").read_text("utf-8")))
