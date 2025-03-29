# SPDX-License-Identifier: Apache-2.0
"""MiniMax VL 01 model implementation."""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any
import torch

from vllm.model_executor.models.modeling_minimax_vl_01 import MiniMaxVL01ForConditionalGeneration
from transformers import PretrainedConfig

# 添加配置类
class MiniMaxVL01Config(PretrainedConfig):
    model_type = "minimax_vl_01"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 避免_LazyConfigMapping初始化错误
        self.mapping = {}

# 重新导出主要的类，这样它们就可以从这个模块直接导入
__all__ = ["MiniMaxVL01ForConditionalGeneration", "MiniMaxVL01Config"] 