import getpass
import json
import os
import warnings
from multiprocessing import shared_memory
from typing import Dict, Optional, Set

import numpy as np
import torch
from emm.slab_allocator import MemoryUsageReader

from sglang.multi_model.utils.get_memory_pool_size import get_model_path_to_model_size


class ResourceManager:
    """
    Resource manager for managing GPU memory allocation and usage tracking.
    Handles active model memory usage, KV cache memory allocation, etc.
    """

    def __init__(
        self,
        gpu_id: int,
        mem_frac: float,
        active_model_names: Set[str],
        engine_info_dict,
        enable_worker_pool: bool,
        model_names_to_model_paths: Dict[str, str],
        num_workers: Optional[int] = None,
    ):
        self.gpu_id = gpu_id
        self.mem_frac = mem_frac
        self.active_model_names = set(active_model_names)
        self.enable_worker_pool = enable_worker_pool
        self.model_names_to_model_paths = model_names_to_model_paths
        self.num_workers = num_workers
        
        self._init_model_names_mappings(engine_info_dict)
        self._init_shared_memory_readers()
        self.total_usable_kv_cache_memory = self.get_total_usable_kv_cache_memory()

    def _init_shared_memory_readers(self):
        """Initialize shared memory readers."""
        self.username = getpass.getuser()
        if self.enable_worker_pool:
            self._worker_to_mem_reader = {}
            for worker_id in range(self.num_workers):
                self._worker_to_mem_reader[worker_id] = MemoryUsageReader(
                    f"ipc_{self.gpu_id}_{worker_id}_{self.username}"
                )
        else:
            self._model_name_to_shm = {}
            for model_name in self.active_model_names:
                self._model_name_to_shm[model_name] = shared_memory.SharedMemory(
                    f"ipc_{self.gpu_id}_{model_name}_{self.username}"
                )

    def get_total_usable_gpu_memory(self):
        """Get total usable GPU memory in GB."""
        total_gpu_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory / (
            1 << 30
        )
        return total_gpu_mem * self.mem_frac

    def get_model_cell_size(self, model_name: str):
        """Get model cell size."""
        return self._model_name_to_cell_size[model_name]

    def add_active_model(self, model_name: str):
        """Add active model."""
        self.active_model_names.add(model_name)
        self.total_usable_kv_cache_memory = self.get_total_usable_kv_cache_memory()

    def remove_active_model(self, model_name: str):
        """Remove active model."""
        self.active_model_names.remove(model_name)
        self.total_usable_kv_cache_memory = self.get_total_usable_kv_cache_memory()

    def get_total_usable_kv_cache_memory(self):
        """Get total usable KV cache memory in GB."""
        total_usable_gpu_memory = self.get_total_usable_gpu_memory()
        active_model_weights = sum(
            self._model_names_to_weights_memory[model_name]
            for model_name in self.active_model_names
        )
        active_model_req_to_token_pool_memory = sum(
            self._model_names_to_req_to_token_pool_memory[model_name]
            for model_name in self.active_model_names
        )
        return (
            total_usable_gpu_memory
            - active_model_weights
            - active_model_req_to_token_pool_memory
        )

    def get_total_used_kv_cache_memory(self):
        """Get total used KV cache memory in bytes."""
        if self.enable_worker_pool:
            used_sum = 0
            for worker_id in range(self.num_workers):
                used_sum += self._worker_to_mem_reader[
                    worker_id
                ].get_memory_usage_in_bytes()
            return used_sum
        else:
            used_sum = 0
            # Create snapshot to avoid concurrent modification
            active_model_names_snapshot = list(self.active_model_names)
            for model_name in active_model_names_snapshot:
                used = self._get_used_kv_cache_memory(model_name)
                used_sum += used
            return used_sum

    def get_available_kv_cache_memory(self):
        """Get available KV cache memory in bytes."""
        total = self.total_usable_kv_cache_memory * (1 << 30)
        used = self.get_total_used_kv_cache_memory()
        safe_margin = 1024 * 1024 * 1024  # 1GB safety margin
        available = total - used - safe_margin
        return available

    def _get_used_kv_cache_memory(self, model_name: str):
        """Get used KV cache memory for the specified model."""
        ipc_name = f"ipc_{self.gpu_id}_{model_name}_{self.username}"
        if ipc_name not in self._model_name_to_shm:
            try:
                self._model_name_to_shm[ipc_name] = shared_memory.SharedMemory(ipc_name)
            except FileNotFoundError:
                warnings.warn(
                    f"Shared memory {ipc_name} not found. Assuming no used kv cache memory."
                )
                return 0
        shm = self._model_name_to_shm[ipc_name]
        memory_in_use = np.ndarray((1,), dtype=np.int64, buffer=shm.buf)
        return memory_in_use[0]  # in bytes

    def _init_model_names_mappings(self, engine_info_dict):
        """Initialize model name mappings."""
        if not self.enable_worker_pool:
            self._model_names_to_req_to_token_pool_memory = {}
            self._model_names_to_weights_memory = {}
            for model_name, engine_info_list in engine_info_dict.items():
                engine_info = engine_info_list[0]
                self._model_names_to_req_to_token_pool_memory[model_name] = (
                    engine_info.memory_usage.req_to_token_pool_memory
                )
                self._model_names_to_weights_memory[model_name] = (
                    engine_info.memory_usage.model_weights_memory
                )
        else:
            self._model_names_to_req_to_token_pool_memory = {
                model_name: 0.25
                for model_name in self.model_names_to_model_paths.keys()
            }
            model_paths = list(set(self.model_names_to_model_paths.values()))
            model_path_to_model_size = get_model_path_to_model_size(model_paths)
            self._model_names_to_weights_memory = {
                model_name: model_path_to_model_size[
                    self.model_names_to_model_paths[model_name]
                ]
                for model_name in self.model_names_to_model_paths.keys()
            }