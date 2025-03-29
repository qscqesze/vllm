# SPDX-License-Identifier: Apache-2.0
"""MiniMax VL 01 model implementation."""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, Any
import torch

from vllm.model_executor.models.modeling_minimax_vl_01 import MiniMaxVL01ForConditionalGeneration
from transformers import PretrainedConfig

# 添加配置类
class MiniMaxVL01Config(PretrainedConfig):
    model_type = "minimax_vl_01"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        rotary_dim=128,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attention_types=None,
        rope_theta=10000,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        
        # 默认使用注意力类型为全注意力 (type=1)
        self.attn_type_list = attention_types or [1] * num_hidden_layers
        
        # 调用父类初始化
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # 避免_LazyConfigMapping初始化错误
        self.mapping = {}
        
        # 确保text_config存在且具有num_attention_heads属性
        if hasattr(self, "text_config"):
            if not hasattr(self.text_config, "num_attention_heads"):
                # 如果缺少必要的属性，从其他属性或默认值中设置
                self.text_config.num_attention_heads = self.num_attention_heads
        else:
            # 如果没有text_config，创建一个基本的
            self.text_config = PretrainedConfig()
            # 从主配置复制基本属性
            for attr in ["num_attention_heads", "hidden_size", "num_hidden_layers", 
                        "intermediate_size", "head_dim", "rotary_dim", 
                        "max_position_embeddings", "rms_norm_eps"]:
                if hasattr(self, attr):
                    setattr(self.text_config, attr, getattr(self, attr))
        
        # 确保vision_config存在，以启用视觉能力
        if not hasattr(self, "vision_config"):
            # 添加基本的视觉配置
            self.vision_config = PretrainedConfig()
            # 设置常见的视觉模型参数
            self.vision_config.hidden_size = 1024
            self.vision_config.image_size = 336
            self.vision_config.patch_size = 14
            self.vision_config.num_hidden_layers = 24
            self.vision_config.num_attention_heads = 16
            self.vision_config.intermediate_size = 4096
            self.vision_config.projection_dim = hidden_size

# 重新导出主要的类，这样它们就可以从这个模块直接导入
__all__ = ["MiniMaxVL01ForConditionalGeneration", "MiniMaxVL01Config"] 