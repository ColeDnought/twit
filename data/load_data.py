from torch.utils.data import Subset
from csv import DictReader
from math import gcd
from pathlib import Path

import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


DATASET_DIRS = ["warblrb10k_public", "ff1010bird", "BirdVox-DCASE-20k"]


class WarblrbDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str | Path = "data", sample_rate: int = 16000):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate  # resample everything to this rate
        self.labels: list[tuple[Path, bool]] = []
        for dataset_dir in DATASET_DIRS:
            self._load_dataset(self.root_dir / dataset_dir)

    def _load_dataset(self, dataset_dir: Path) -> None:
        metadata = dataset_dir / "metadata.csv"
        with open(metadata, "r") as f:
            self.labels.extend(
                (dataset_dir / f"{row['itemid']}.wav", bool(int(row["hasbird"])))
                for row in DictReader(f)
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, bool]:
        audio_path, label = self.labels[idx]
        sr, data = wavfile.read(audio_path)
        if sr != self.sample_rate:
            # polyphase resample to the model's expected rate (dev sets are 44.1 kHz)
            g = gcd(sr, self.sample_rate)
            data = resample_poly(data, self.sample_rate // g, sr // g)
        return torch.from_numpy(data.astype("float32")), label

def collate_fn(batch):
    """Pad variable-length waveforms to the batch max and return valid lengths."""
    xs, ys = zip(*batch)
    xs = [x.float() / 32768.0 for x in xs]              # int16 -> float in [-1, 1]
    lengths = torch.tensor([x.shape[-1] for x in xs])
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)              # [B, L_max]
    y = torch.tensor(ys, dtype=torch.float32)           # BCEWithLogitsLoss wants float
    return x.unsqueeze(1), y, lengths                   # [B, 1, L_max], [B], [B])


def get_class_imbalance(dataset: torch.utils.data.Dataset) -> torch.Tensor:
    """neg/pos count ratio for BCEWithLogitsLoss pos_weight (train split, not global)."""
    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, dataset.indices
    else:
        base, indices = dataset, range(len(dataset))
    ys = [base.labels[i][1] for i in indices]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    return torch.tensor(n_neg / n_pos)
