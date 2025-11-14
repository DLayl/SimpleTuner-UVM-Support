"""
In-memory data backend tuned for GH200 systems.

The backend eagerly loads an entire dataset directory into Grace DDR5
and serves future reads directly from memory.  It is intentionally
compatible with the :class:`BaseDataBackend` interface used throughout
SimpleTuner so it can be dropped into existing configurations.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.training import image_file_extensions
from simpletuner.helpers.training.multi_process import rank_info, should_log

logger = logging.getLogger("GH200InMemoryBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


_DEFAULT_EXTENSIONS = tuple({ext.lower() for ext in image_file_extensions})


def _resolve_compression(name: Optional[str]):
    if not name:
        return None, None, "none"
    normalized = name.strip().lower()
    if normalized == "zlib":
        import zlib

        return lambda data: zlib.compress(data, level=1), zlib.decompress, "zlib"
    if normalized == "lz4":
        try:
            import lz4.frame
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Compression library 'lz4' is not installed.") from exc

        return lz4.frame.compress, lz4.frame.decompress, "lz4"
    if normalized == "snappy":
        try:
            import snappy
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Compression library 'snappy' is not installed.") from exc

        return snappy.compress, snappy.decompress, "snappy"
    raise ValueError(f"Unknown compression library: {name}")


class GH200InMemoryBackend(BaseDataBackend):
    """
    *Experimental*: data backend that stages an entire dataset in RAM.
    """

    def __init__(
        self,
        accelerator,
        id: str,
        instance_data_dir: str,
        *,
        file_extensions: Optional[Iterable[str]] = None,
        compression: Optional[str] = None,
        memory_limit_gb: Optional[float] = None,
        num_workers: int = 16,
    ) -> None:
        self.accelerator = accelerator
        self.id = id
        self.type = "in_memory"
        self.instance_data_dir = Path(instance_data_dir)
        self._file_extensions = tuple(ext.lower().lstrip(".") for ext in (file_extensions or _DEFAULT_EXTENSIONS))
        self._compress_fn, self._decompress_fn, self._compression_name = _resolve_compression(compression)
        self._memory_limit_bytes = None if memory_limit_gb is None else int(memory_limit_gb * (1024**3))
        self._num_workers = max(1, num_workers)
        self._storage: Dict[str, bytes] = {}
        self._lock = threading.RLock()

        if not self.instance_data_dir.exists():
            raise FileNotFoundError(f"Instance data directory '{self.instance_data_dir}' does not exist.")

        self._load_dataset()

    # ------------------------------------------------------------------
    # BaseDataBackend compliance
    # ------------------------------------------------------------------
    def get_instance_representation(self) -> dict:
        return {
            "backend_type": "in_memory",
            "id": self.id,
            "instance_data_dir": str(self.instance_data_dir),
            "file_extensions": list(self._file_extensions),
            "compression": self._compression_name,
            "memory_limit_gb": None if self._memory_limit_bytes is None else self._memory_limit_bytes / (1024**3),
            "num_workers": self._num_workers,
        }

    @staticmethod
    def from_instance_representation(representation: dict) -> "GH200InMemoryBackend":
        if representation.get("backend_type") != "in_memory":
            raise ValueError(f"Expected backend_type 'in_memory', got {representation.get('backend_type')}")

        return GH200InMemoryBackend(
            accelerator=None,
            id=representation["id"],
            instance_data_dir=representation["instance_data_dir"],
            file_extensions=representation.get("file_extensions"),
            compression=representation.get("compression"),
            memory_limit_gb=representation.get("memory_limit_gb"),
            num_workers=representation.get("num_workers", 16),
        )

    def read(self, identifier: str, as_byte_io: bool = False):
        data = self._storage[identifier]
        payload = self._decompress_fn(data) if self._decompress_fn else data
        return BytesIO(payload) if as_byte_io else payload

    def write(self, identifier: str, data: Any):
        if isinstance(data, BytesIO):
            payload = data.getvalue()
        elif isinstance(data, bytes):
            payload = data
        elif isinstance(data, torch.Tensor):
            buffer = BytesIO()
            torch.save(data, buffer)
            payload = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported data type for write: {type(data)}")
        if self._compress_fn:
            payload = self._compress_fn(payload)
        with self._lock:
            self._storage[identifier] = payload

    def delete(self, identifier: str):
        with self._lock:
            self._storage.pop(identifier, None)

    def exists(self, identifier: str) -> bool:
        return identifier in self._storage

    def open_file(self, identifier: str, mode: str):
        if "r" in mode and identifier in self._storage:
            payload = self._storage[identifier]
            data = self._decompress_fn(payload) if self._decompress_fn else payload
            return BytesIO(data)
        raise NotImplementedError("GH200InMemoryBackend only supports reading staged files.")

    def list_files(self, file_extensions: List[str], instance_data_dir: str = "") -> List[Tuple[str, List, List[str]]]:
        results: Dict[str, List[str]] = {}
        extensions = set(ext.lower().lstrip(".") for ext in (file_extensions or self._file_extensions))
        for rel_path in self._storage.keys():
            path_obj = Path(rel_path)
            if extensions and path_obj.suffix.lower().lstrip(".") not in extensions:
                continue
            parent = str(path_obj.parent)
            results.setdefault(parent, []).append(str(self.instance_data_dir / path_obj))
        return [(folder, [], files) for folder, files in results.items()]

    def get_abs_path(self, sample_path: str = None) -> str:
        if sample_path is None:
            return str(self.instance_data_dir)
        if os.path.isabs(sample_path):
            return sample_path
        return str(self.instance_data_dir / sample_path)

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        rel_path = self._relative_key(filepath)
        try:
            payload = self.read(rel_path)
        except KeyError:
            raise FileNotFoundError(filepath)
        buffer = BytesIO(payload)
        image = Image.open(buffer)
        image.load()
        return image

    def read_image_batch(self, filepaths: Iterable[str], delete_problematic_images: bool = False):
        return [self.read_image(path, delete_problematic_images) for path in filepaths]

    def create_directory(self, directory_path):
        # No-op: data lives in-memory.
        return directory_path

    def torch_load(self, filename):
        rel_path = self._relative_key(filename)
        payload = self.read(rel_path)
        buffer = BytesIO(payload)
        return torch.load(buffer, map_location="cpu")

    def torch_save(self, data, filename):
        buffer = BytesIO()
        torch.save(data, buffer)
        self.write(self._relative_key(filename), buffer.getvalue())

    def write_batch(self, identifiers, files):
        for identifier, payload in zip(identifiers, files):
            if isinstance(payload, torch.Tensor):
                buffer = BytesIO()
                torch.save(payload, buffer)
                self.write(self._relative_key(identifier), buffer.getvalue())
            else:
                self.write(self._relative_key(identifier), payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _relative_key(self, path: str) -> str:
        path_obj = Path(path)
        try:
            return str(path_obj.relative_to(self.instance_data_dir))
        except ValueError:
            return str(path_obj)

    def _gather_files(self) -> List[Path]:
        extensions = {f".{ext}" for ext in self._file_extensions}
        files: List[Path] = []
        for extension in extensions:
            files.extend(self.instance_data_dir.rglob(f"*{extension}"))
            files.extend(self.instance_data_dir.rglob(f"*{extension.upper()}"))
        return sorted({path.resolve() for path in files})

    def _load_dataset(self) -> None:
        files = self._gather_files()
        if not files:
            logger.warning("%s No files discovered under %s", rank_info(), self.instance_data_dir)
            return

        total_bytes = sum(path.stat().st_size for path in files)
        if self._memory_limit_bytes and total_bytes > self._memory_limit_bytes:
            raise MemoryError(
                f"Dataset requires {total_bytes / (1024 ** 3):.1f} GB but the limit is "
                f"{self._memory_limit_bytes / (1024 ** 3):.1f} GB."
            )

        start = time.time()
        logger.info(
            "%s Loading %d files (%.1f GB raw) into GH200InMemoryBackend[%s] with %d workers.",
            rank_info(),
            len(files),
            total_bytes / (1024**3),
            self.id,
            self._num_workers,
        )

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            futures = [executor.submit(self._load_single_file, path) for path in files]
            for future in as_completed(futures):
                rel_path, payload = future.result()
                with self._lock:
                    self._storage[rel_path] = payload

        elapsed = time.time() - start
        logger.info(
            "%s Finished staging %d files into memory in %.1fs (compression=%s).",
            rank_info(),
            len(self._storage),
            elapsed,
            self._compression_name,
        )

    def _load_single_file(self, path: Path) -> Tuple[str, bytes]:
        with path.open("rb") as handle:
            payload = handle.read()
        if self._compress_fn:
            payload = self._compress_fn(payload)
        rel_path = str(path.relative_to(self.instance_data_dir))
        return rel_path, payload

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def write_metadata_snapshot(self, output_path: Path) -> None:
        snapshot = {
            "id": self.id,
            "files": list(self._storage.keys()),
            "compression": self._compression_name,
            "instance_data_dir": str(self.instance_data_dir),
            "timestamp": time.time(),
        }
        output_path.write_text(json.dumps(snapshot, indent=2))
