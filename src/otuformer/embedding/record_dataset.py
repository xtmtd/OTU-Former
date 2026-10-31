"""Dataset views backed by validated v2 data records.

Both views decode the bytes they verify, so a replaced source cannot change what
inference consumes. They yield the same batch shapes as the legacy CSV/directory
datasets, which keeps the existing inference loops unchanged.
"""

from __future__ import annotations

import io
from dataclasses import replace

from PIL import Image
from torch.utils.data import Dataset

from otuformer.data.paths import require_contained
from otuformer.data.provenance import read_verified_bytes
from otuformer.data.records import DataBundle
from otuformer.training.dataset import build_eval_transform


class RecordImageDataset(Dataset):
    def __init__(
        self, bundle: DataBundle, extract_size: int, eval_transform: str = "center-crop"
    ) -> None:
        self.bundle = bundle
        self.records = bundle.records
        self.transform = build_eval_transform(eval_transform, extract_size)
        # Same contract as CSVImageDataset: ``samples is None`` means this view
        # emits no sample column at all, so an image-only selection cannot produce
        # a length-zero column.
        samples = [record.sample for record in self.records]
        self.samples: list[str] | None = (
            None
            if all(sample is None for sample in samples)
            else [sample or "" for sample in samples]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _verified_snapshot(self, record):
        """The snapshot to read, pinned to the path containment just accepted.

        Checked here at the point of consumption: an entry-time check cannot see a
        path swapped while the model was loading.
        """
        resolved = require_contained(
            self.bundle.images_dir, record.access_path, record.image_key
        )
        if resolved == record.image_file.path:
            return record.image_file
        return replace(record.image_file, path=resolved)

    def _tensor(self, index: int):
        record = self.records[index]
        data = read_verified_bytes(self._verified_snapshot(record))
        with Image.open(io.BytesIO(data)) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int):
        record = self.records[index]
        tensor = self._tensor(index)
        if self.samples is None:
            return tensor, record.output_id
        return tensor, record.output_id, self.samples[index]


class RecordLabelDataset(Dataset):
    """Image view with sorted unique nonempty raw labels from the same records."""

    def __init__(
        self, bundle: DataBundle, extract_size: int, eval_transform: str = "center-crop"
    ) -> None:
        self.bundle = bundle
        self.records = bundle.records
        self.transform = build_eval_transform(eval_transform, extract_size)
        labels = [record.metadata.get("label", "") for record in self.records]
        missing = [
            record.image_key
            for record, label in zip(self.records, labels)
            if label == ""
        ]
        if missing:
            sample = ", ".join(missing[:3])
            raise ValueError(
                "attention-query training needs a label for every selected training "
                f"row; {len(missing)} row(s) have no label (e.g. {sample})"
            )
        self.class_names = sorted({label for label in labels})
        self.label_to_index = {
            name: index for index, name in enumerate(self.class_names)
        }

    def __len__(self) -> int:
        return len(self.records)

    def _verified_snapshot(self, record):
        resolved = require_contained(
            self.bundle.images_dir, record.access_path, record.image_key
        )
        if resolved == record.image_file.path:
            return record.image_file
        return replace(record.image_file, path=resolved)

    def __getitem__(self, index: int):
        record = self.records[index]
        data = read_verified_bytes(self._verified_snapshot(record))
        with Image.open(io.BytesIO(data)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[record.metadata["label"]]
