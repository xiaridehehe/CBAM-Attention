"""Batch three-view exams while preserving their original slice counts."""
import torch


def collate_packed_views(batch):
    exams, targets, ids = zip(*batch)
    for views in exams:
        if len(views) != 3 or any(
            view.ndim != 4 or view.shape[0] == 0 or view.shape[1] != 3 for view in views
        ):
            raise ValueError("Each exam must have three nonempty [slices, 3, H, W] views")
    views = [view for exam in exams for view in exam]
    counts = tuple(view.shape[0] for view in views)
    return torch.cat(views, dim=0), counts, torch.stack(targets), ids
