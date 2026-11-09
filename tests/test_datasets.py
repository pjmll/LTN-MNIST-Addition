import unittest

import torch
from torch.utils.data import DataLoader

from src.datasets.mnist import AdditionPairs, DigitTestImages, make_pairs, split_train_images


class DatasetTests(unittest.TestCase):
    def test_split_is_disjoint_reproducible_and_complete(self) -> None:
        train, val = split_train_images(40, 8, seed=21)
        torch.rand(15)
        again_train, again_val = split_train_images(40, 8, seed=21)
        torch.testing.assert_close(train, again_train)
        torch.testing.assert_close(val, again_val)
        self.assertEqual((len(train), len(val)), (32, 8))
        self.assertFalse(set(train.tolist()) & set(val.tolist()))
        self.assertEqual(set(train.tolist() + val.tolist()), set(range(40)))

    def test_pairing_uses_halves_without_image_reuse(self) -> None:
        indices = torch.tensor([5, 7, 2, 3, 4, 8])
        pairs = make_pairs(indices)
        self.assertEqual(pairs.tolist(), [[5, 3], [7, 4], [2, 8]])
        self.assertCountEqual(pairs.flatten().tolist(), indices.tolist())
        torch.testing.assert_close(make_pairs(indices, limit=2), pairs[:2])

    def test_invalid_split_and_pair_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_train_images(40, 7, seed=21)
        with self.assertRaises(ValueError):
            make_pairs(torch.arange(5))
        with self.assertRaises(ValueError):
            make_pairs(torch.arange(6), limit=4)

    def test_batch_exposes_only_two_images_and_their_sums(self) -> None:
        images = torch.zeros((6, 28, 28), dtype=torch.uint8)
        images[3].fill_(255)
        digits = torch.tensor([0, 9, 2, 0, 9, 3])
        dataset = AdditionPairs(images, digits, make_pairs(torch.arange(6)))
        digits.fill_(-1)  # Targets are already reduced to sums inside the dataset.
        batch = next(iter(DataLoader(dataset, batch_size=3)))
        self.assertEqual(len(batch), 3)
        left, right, sums = batch
        self.assertEqual(left.shape, (3, 1, 28, 28))
        self.assertEqual(right.shape, left.shape)
        self.assertEqual(left.dtype, torch.float32)
        self.assertEqual(sums.dtype, torch.long)
        self.assertEqual(sums.tolist(), [0, 18, 5])
        self.assertEqual(float(left[0].max()), 0.0)
        self.assertEqual(float(right[0].min()), 1.0)
        self.assertEqual(int(images[3, 0, 0]), 255)

    def test_digit_test_images_share_pair_preprocessing(self) -> None:
        images = torch.full((2, 28, 28), 127, dtype=torch.uint8)
        digits = torch.tensor([2, 8])
        pair = AdditionPairs(images, digits, make_pairs(torch.arange(2)))
        test = DigitTestImages(images, digits)
        left, right, total = pair[0]
        torch.testing.assert_close(left, test[0][0])
        torch.testing.assert_close(right, test[1][0])
        self.assertEqual([int(test[0][1]), int(test[1][1]), int(total)], [2, 8, 10])
