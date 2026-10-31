"""Shared v2 data ingestion contracts.

Re-exports the reader entry point only; import ``otuformer.data.reader`` or the
sibling modules for record types and helpers. No torch or CLI imports here.
"""

from .reader import read_dataset
from .records import DataBundle, DataValidationError

__all__ = ["read_dataset", "DataBundle", "DataValidationError"]
