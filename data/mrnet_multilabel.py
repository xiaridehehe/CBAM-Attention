"""MRNet workbook and native-depth NPY volumes."""
import ast
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from torch.utils.data import Dataset
# Load Pillow's native extension before torchvision loads its own DLLs on Windows.
from PIL import Image as _pillow_image
from torchvision.transforms import InterpolationMode, RandomAffine, RandomHorizontalFlip

DEFAULT_DATA_ROOT = Path("D:/dataset/MRNet")
LABEL_NAMES = ("Abnormal", "ACL", "Meniscus")
LABEL_CODES = (1, 2, 3)
PLANES = ("ax", "sag", "cor")


def read_records(path):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book.active.values)
        headers = next(rows)
        records = []
        for values in rows:
            row = dict(zip(headers, values))
            if row["Patient ID"] is None:
                continue
            patient = str(row["Patient ID"]).strip().zfill(4)
            target = ast.literal_eval(str(row["Label"]))
            if not isinstance(target, (list, tuple)) or len(target) != 3 or any(v not in (0, 1) for v in target):
                raise ValueError(f"Invalid label for {patient}: {target}")
            division = str(row["Division"]).strip().lower()
            if division not in ("train", "test"):
                raise ValueError(f"Unknown division for {patient}: {division}")
            row.update({"Patient ID": patient, "target": list(map(float, target)), "Division": division})
            records.append(row)
        return records
    finally:
        book.close()


def split_records(records):
    train = [row for row in records if row["Division"] == "train"]
    val = [row for row in records if row["Division"] == "test"]
    if not train or not val:
        raise ValueError("Both training and validation splits must be nonempty")
    for split in (train, val):
        if len({row["Patient ID"] for row in split}) != len(split):
            raise ValueError("Duplicate patient within split")
    if {row["Patient ID"] for row in train} & {row["Patient ID"] for row in val}:
        raise ValueError("Patient IDs overlap between splits")
    return train, val


class MRNetDataset(Dataset):
    def __init__(self, records, root, size=256, augment=False, degrees=10, translate=0.05):
        self.records = records
        self.size = size
        self.paths = []
        self.affine = RandomAffine(degrees, translate=(translate, translate),
                                   interpolation=InterpolationMode.BILINEAR)
        self.flip = RandomHorizontalFlip()
        self.augment = augment
        for row in records:
            paths = []
            for plane in PLANES:
                value = str(row[f"{plane}_DICOM_Directory"])
                windows_path = PureWindowsPath(value)
                path = Path(value)
                if not (path.is_absolute() or windows_path.is_absolute()):
                    path = Path(root).joinpath(*windows_path.parts)
                if not path.is_file():
                    raise FileNotFoundError(f'{row["Patient ID"]}/{plane}: {path}')
                paths.append(path)
            self.paths.append(paths)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        views = []
        for path in self.paths[index]:
            volume = np.load(path, allow_pickle=False).astype(np.float32)
            if volume.ndim != 3 or not volume.size or not np.isfinite(volume).all():
                raise ValueError(f"Invalid NPY volume: {path}")
            x = torch.from_numpy(volume).div_(255)
            if x.shape[-2:] != (self.size, self.size):
                x = torch.nn.functional.interpolate(
                    x[:, None], (self.size, self.size), mode="bilinear", align_corners=False
                )[:, 0]
            if self.augment:
                # Treat slices as channels so one transform is applied to the entire view.
                x = self.flip(self.affine(x))
            views.append(x[:, None].expand(-1, 3, -1, -1))
        row = self.records[index]
        return views, torch.tensor(row["target"], dtype=torch.float32), row["Patient ID"]
