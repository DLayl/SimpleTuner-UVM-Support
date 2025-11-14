"""GH200 in-memory backend builder."""

import logging
from typing import Any, Dict, Optional

from simpletuner.gh200.in_memory_backend import GH200InMemoryBackend
from simpletuner.helpers.data_backend.config.base import BaseBackendConfig

from .base import BaseBackendBuilder

logger = logging.getLogger("InMemoryBackendBuilder")


class InMemoryBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> GH200InMemoryBackend:
        backend_cfg: Dict[str, Any] = config.to_dict()["config"]
        in_memory_config: Dict[str, Any] = backend_cfg.get("in_memory", {})

        file_extensions = in_memory_config.get("file_extensions")
        compression = in_memory_config.get("compression")
        memory_limit = in_memory_config.get("memory_limit_gb")
        num_workers = in_memory_config.get("num_workers", 32)

        instance_dir = backend_cfg.get("instance_data_dir", config.instance_data_dir)
        if not instance_dir:
            raise ValueError(f"(id={config.id}) In-memory backend requires an 'instance_data_dir'.")

        backend = GH200InMemoryBackend(
            accelerator=self.accelerator,
            id=config.id,
            instance_data_dir=instance_dir,
            file_extensions=file_extensions,
            compression=compression,
            memory_limit_gb=memory_limit,
            num_workers=num_workers,
        )

        return backend

    def build_with_metadata(
        self, config: BaseBackendConfig, args: Dict[str, Any], instance_data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"(id={config.id}) Loading in-memory dataset for GH200 deployment.")

        data_backend = self.build(config)

        metadata_backend = self.create_metadata_backend(
            config=config,
            data_backend=data_backend,
            args=args,
            instance_data_dir=instance_data_dir or data_backend.get_abs_path(),
        )

        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": data_backend.get_abs_path(),
            "config": config.to_dict()["config"],
        }
