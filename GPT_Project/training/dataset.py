import numpy as np
import torch

from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(
        self,
        data_path,
        block_size
    ):

        self.data = np.load(
            data_path,
            mmap_mode="r"
        )

        self.block_size = block_size

    def __len__(self):

        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):

        start = np.random.randint(
            0,
            len(self.data) - self.block_size - 1
        )

        x = self.data[
            start:
            start + self.block_size
        ]

        y = self.data[
            start + 1:
            start + self.block_size + 1
        ]

        x = torch.tensor(
            x,
            dtype=torch.long
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y