# src/torch_segment_tomogram_boundaries/data/dataloading.py
import random
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler

from torch_segment_tomogram_boundaries.data.dataset import PTFileDataset
from torch_segment_tomogram_boundaries.data.transforms import get_transforms


class StratifiedSampler(Sampler[int]):
    """
    Stratified sampler to ensure each batch contains samples from different volumes.

    This sampler groups samples by their volume ID (extracted from the filename)
    and then yields indices in an interleaved fashion. This ensures that when the
    DataLoader creates a batch, it is composed of slices from different original
    tomograms, improving model generalization.
    """

    def __init__(self, file_paths: List[Path], batch_size: int, seed: int = 42):
        super().__init__(file_paths)
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.groups = self._group_by_volume()
        self.num_samples = len(file_paths)
        self._rng = random.Random(seed)

    def _group_by_volume(self) -> List[List[int]]:
        """Groups file indices by the volume stem from their filename."""
        # Regex to extract the volume stem from filenames like 'train_Tomo2_view_123.pt'
        # It captures the part between the first '_' and the '\_view' part.
        stem_regex = re.compile(r"_(.*?)(?=_view_\d+\.pt)")

        groups = defaultdict(list)
        for i, path in enumerate(self.file_paths):
            match = stem_regex.search(path.name)
            # Use the full stem as a fallback if regex fails
            stem = match.group(1) if match else path.stem.split("_")[1]
            groups[stem].append(i)

        return list(groups.values())

    def __iter__(self) -> Iterator[int]:
        """Yields a sequence of indices with volumes interleaved."""
        # Shuffle indices within each group
        for group in self.groups:
            self._rng.shuffle(group)

        # Interleave the groups
        # Create a list of iterators for each group
        iterators = [iter(group) for group in self.groups]

        # Round-robin yielding of indices
        while iterators:
            # Shuffle the order of volumes for each round-robin cycle
            self._rng.shuffle(iterators)

            for i in range(len(iterators) - 1, -1, -1):
                it = iterators[i]
                try:
                    yield next(it)
                except StopIteration:
                    # Remove exhausted iterators
                    iterators.pop(i)

    def __len__(self) -> int:
        return self.num_samples


class SegmentationDataModule(pl.LightningDataModule):
    """
    A simplified DataModule that loads full, pre-standardized 2D images
    and applies the robust augmentation pipeline.
    """

    def __init__(
        self,
        train_pt_files: List[Path],
        val_pt_files: List[Path],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "num_workers")
        self.train_pt_files = train_pt_files
        self.val_pt_files = val_pt_files
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        This method is called by Lightning to prepare data. It creates the
        datasets and assigns the correct augmentation pipeline to each.
        """
        self.train_dataset = PTFileDataset(
            self.train_pt_files, transform=get_transforms(is_training=True)
        )
        self.val_dataset = PTFileDataset(
            self.val_pt_files, transform=get_transforms(is_training=False)
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the training DataLoader with the StratifiedSampler."""
        sampler = StratifiedSampler(
            self.train_pt_files, batch_size=self.hparams.batch_size, seed=42
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True,  # Recommended for stratified sampling
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the standard validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
        )
