from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def split_train_images(
    size: int, validation_images: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Split official training indices before constructing either set of pairs."""
    if not 0 < validation_images < size or validation_images % 2 or size % 2:
        raise ValueError("Train and validation splits must both have positive even sizes")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    return order[validation_images:], order[:validation_images]


def make_pairs(indices: Tensor, limit: int | None = None) -> Tensor:
    """Pair corresponding positions from the first and second halves."""
    if indices.ndim != 1 or len(indices) < 2 or len(indices) % 2:
        raise ValueError("Pairing requires a positive even number of image indices")
    count = len(indices) // 2
    if limit is not None and not 1 <= limit <= count:
        raise ValueError(f"Pair limit must be between 1 and {count}")
    pairs = torch.stack((indices[:count], indices[count:]), dim=1)
    return pairs[:limit]


def _normalize(images: Tensor) -> Tensor:
    return images.unsqueeze(-3).to(torch.float32).div_(255)


class AdditionPairs(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Build sum targets once and retain no individual digit targets."""

    def __init__(self, images: Tensor, digits: Tensor, pairs: Tensor) -> None:
        self._images = images
        self._pairs = pairs.clone()
        self._sums = digits[self._pairs].sum(dim=1)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        left, right = _normalize(self._images[self._pairs[index]])
        return left, right, self._sums[index]


class DigitTestImages(Dataset[tuple[Tensor, Tensor]]):
    """Expose individual digits to the test evaluator."""

    def __init__(self, images: Tensor, digits: Tensor) -> None:
        self._images = images
        self._digits = digits

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return _normalize(self._images[index]), self._digits[index]


class ValidationPairs(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Expose both digit labels for evaluation on the existing validation pairs."""

    def __init__(self, images: Tensor, digits: Tensor, pairs: Tensor) -> None:
        self._addition = AdditionPairs(images, digits, pairs)
        self._digits = digits[pairs]

    def __len__(self) -> int:
        return len(self._addition)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (*self._addition[index], self._digits[index])


def load_train_validation(
    root: Path,
    *,
    download: bool,
    validation_images: int,
    data_seed: int,
    train_limit: int | None,
    validation_limit: int | None,
    validation_digits: bool = False,
) -> tuple[AdditionPairs, AdditionPairs | ValidationPairs, Tensor, Tensor]:
    source = MNIST(root, train=True, download=download)
    train_indices, val_indices = split_train_images(
        len(source), validation_images, data_seed
    )
    train_pairs = make_pairs(train_indices, train_limit)
    val_pairs = make_pairs(val_indices, validation_limit)
    validation_type = ValidationPairs if validation_digits else AdditionPairs
    return (
        AdditionPairs(source.data, source.targets, train_pairs),
        validation_type(source.data, source.targets, val_pairs),
        train_pairs,
        val_pairs,
    )


def load_test(root: Path, *, download: bool) -> tuple[DigitTestImages, Tensor]:
    """Return ordered test images and their pair indices for evaluation only."""
    source = MNIST(root, train=False, download=download)
    pairs = make_pairs(torch.arange(len(source)))
    return DigitTestImages(source.data, source.targets), pairs
