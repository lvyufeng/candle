"""Abstract base classes for DCP storage backends.

Mirrors ``torch.distributed.checkpoint.storage``.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .metadata import MetadataIndex, Metadata, _StorageInfo


# ---------------------------------------------------------------------------
# Write result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WriteResult:
    """Result of writing a single item to storage."""
    index: MetadataIndex
    size_in_bytes: int
    storage_data: Optional[_StorageInfo] = None


# ---------------------------------------------------------------------------
# Future-like wrapper returned by write_data / read_data
# ---------------------------------------------------------------------------

class _Future:
    """Trivial synchronous future wrapping an already-computed value."""

    def __init__(self, value=None):
        self._value = value

    def wait(self):
        return self._value


# ---------------------------------------------------------------------------
# StorageWriter ABC
# ---------------------------------------------------------------------------

class StorageWriter(ABC):
    """Interface that DCP save uses to persist tensor data."""

    @abstractmethod
    def reset(self, checkpoint_id=None):
        """Prepare for a new checkpoint write."""

    @abstractmethod
    def set_up_storage_writer(self, is_coordinator):
        """One-time setup (e.g. create directories)."""

    @abstractmethod
    def prepare_local_plan(self, plan):
        """Adjust the local save plan (e.g. assign file names)."""

    @abstractmethod
    def prepare_global_plan(self, global_plan):
        """Adjust the global save plan after coordination."""

    @abstractmethod
    def write_data(self, plan, planner):
        """Write tensor data according to *plan*. Returns a Future[List[WriteResult]]."""

    @abstractmethod
    def finish(self, metadata, results):
        """Finalize the checkpoint (e.g. write .metadata). Coordinator only."""

    def storage_meta(self):
        """Return optional StorageMeta for provenance tracking."""
        return None

    def validate_checkpoint_id(self, checkpoint_id):
        """Validate that *checkpoint_id* is usable. Default: no-op."""


# ---------------------------------------------------------------------------
# StorageReader ABC
# ---------------------------------------------------------------------------

class StorageReader(ABC):
    """Interface that DCP load uses to read tensor data."""

    @abstractmethod
    def reset(self, checkpoint_id=None):
        """Prepare for a new checkpoint read."""

    @abstractmethod
    def read_metadata(self):
        """Read and return the Metadata object. Coordinator only."""

    @abstractmethod
    def set_up_storage_reader(self, metadata, is_coordinator):
        """One-time setup after metadata is available."""

    @abstractmethod
    def prepare_local_plan(self, plan):
        """Adjust the local load plan."""

    @abstractmethod
    def prepare_global_plan(self, global_plan):
        """Adjust the global load plan after coordination."""

    @abstractmethod
    def read_data(self, plan, planner):
        """Read tensor data according to *plan*. Returns a Future[None]."""

    def validate_checkpoint_id(self, checkpoint_id):
        """Validate that *checkpoint_id* is usable. Default: no-op."""
