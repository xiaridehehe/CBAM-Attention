"""Read the three exported DICOM series for each exam in data_split.xlsx."""
from pathlib import Path, PureWindowsPath

import numpy as np
import pydicom
import torch
from openpyxl import load_workbook
from torch.nn import functional as F
from torch.utils.data import Dataset

# Import the existing dataset module first to initialize Pillow before torchvision.
from data.mrnet_multilabel import PLANES
from torchvision.transforms import InterpolationMode, RandomAffine, RandomHorizontalFlip

DEFAULT_DATA_ROOT = Path("E:/Program1/KAMRNet-main/dataset/resampled_xy_0p4_dicom")
LABEL_CODES = tuple(range(1, 16))
LABEL_NAMES = tuple(f"label_{code}" for code in LABEL_CODES)


def read_splits(path):
    splits = {"0": [], "1": []}
    patient_split = {}
    exams = set()
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book["Data"].values)
        headers = next(rows)
        for values in rows:
            row = dict(zip(headers, values))
            if row["Patient ID"] is None:
                continue
            patient = str(row["Patient ID"]).strip()
            exam = (patient, str(row["Image Number"]).strip())
            division = str(row["Division"]).strip()
            if division not in splits:
                raise ValueError(f"Unknown Division {division!r} for {exam}")
            if patient_split.setdefault(patient, division) != division:
                raise ValueError(f"Patient crosses splits: {patient}")
            if exam in exams:
                raise ValueError(f"Duplicate exam: {exam}")
            exams.add(exam)
            codes = {int(code.strip()) for code in str(row["Label"]).split(",")}
            if not codes.issubset(LABEL_CODES):
                raise ValueError(f"Unknown label codes {codes} for {exam}")
            row["target"] = [float(code in codes) for code in LABEL_CODES]
            splits[division].append(row)
    finally:
        book.close()
    train, val = splits["0"], splits["1"]
    if not train or not val:
        raise ValueError("Both training and validation splits must contain exams")
    return train, val


def load_volume(paths):
    slices = []
    normal = None
    series = None
    for path in paths:
        ds = pydicom.dcmread(path)
        direction = np.asarray(ds.ImageOrientationPatient, dtype=float)
        current_normal = np.cross(direction[:3], direction[3:])
        if normal is None:
            normal = current_normal
            series = ds.SeriesInstanceUID
        if ds.SeriesInstanceUID != series or not np.allclose(normal, current_normal, atol=1e-4):
            raise ValueError(f"Inconsistent DICOM series geometry: {path}")
        position = float(np.dot(normal, np.asarray(ds.ImagePositionPatient, dtype=float)))
        pixels = ds.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError(f"Expected a single grayscale DICOM slice: {path}")
        pixels *= float(getattr(ds, "RescaleSlope", 1))
        pixels += float(getattr(ds, "RescaleIntercept", 0))
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixels = -pixels
        slices.append((position, pixels))
    slices.sort(key=lambda item: item[0])
    if any(b[0] - a[0] < 1e-5 for a, b in zip(slices, slices[1:])):
        raise ValueError(f"Duplicate DICOM slice positions: {paths[0].parent}")
    volume = np.stack([pixels for _, pixels in slices])
    lo, hi = np.percentile(volume, (1, 99))
    return torch.from_numpy(np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32))


class KAMRNetDicomDataset(Dataset):
    def __init__(self, records, root, size=256, augment=False, degrees=10, translate=0.05):
        self.records = records
        self.size = size
        self.augment = augment
        self.affine = RandomAffine(degrees, translate=(translate, translate),
                                   interpolation=InterpolationMode.BILINEAR)
        self.flip = RandomHorizontalFlip()
        self.paths = []
        for row in records:
            views = []
            for plane in PLANES:
                folder = Path(root).joinpath(*PureWindowsPath(row[f"{plane}_DICOM_Directory"]).parts)
                paths = sorted(folder.glob("*.dcm"))
                if not paths:
                    raise FileNotFoundError(f"No DICOM slices in {folder}")
                if len(paths) != int(row[f"{plane}_Slices"]):
                    raise ValueError(f"DICOM slice count differs from workbook: {folder}")
                views.append(paths)
            self.paths.append(views)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        views = []
        for paths in self.paths[index]:
            x = load_volume(paths)
            if x.shape[-2:] != (self.size, self.size):
                x = F.interpolate(x[:, None], (self.size, self.size), mode="bilinear",
                                  align_corners=False)[:, 0]
            if self.augment:
                x = self.flip(self.affine(x))
            views.append(x[:, None].expand(-1, 3, -1, -1))
        row = self.records[index]
        return views, torch.tensor(row["target"], dtype=torch.float32), str(row["Image Number"])
