# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class MinimaxCacheParams:
    minimax_cache: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return MinimaxCacheParams(self.minimax_cache[layer_idx, ...],
                                  self.state_indices_tensor)


class MinimaxCacheManager(ConstantSizeCache):

    def __init__(self, dtype, cache_shape):
        super().__init__(cache_shape[1])  # max_batch_size is cache_shape[1]
        self._minimax_cache = torch.empty(size=cache_shape,
                                          dtype=dtype,
                                          device="cuda",
                                          pin_memory=True)
        self._cache_usage = 0
        self._max_cache_usage = 0.8  # 最大缓存使用率

    @property
    def cache(self):
        return self._minimax_cache

    def _copy_cache(self, from_index: int, to_index: int):
        assert len(self.cache) > 0
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def _clear_cache_if_needed(self):
        if self._cache_usage > self._max_cache_usage:
            torch.cuda.empty_cache()
            self._cache_usage = 0
            
    def _update_cache_usage(self):
        self._cache_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        self._clear_cache_if_needed()
