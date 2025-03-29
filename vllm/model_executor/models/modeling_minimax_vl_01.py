# SPDX-License-Identifier: Apache-2.0
"""Inference-only MiniMaxText01 model."""
import copy
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union, Set, Any
import torch
import torch.distributed
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers import PretrainedConfig, AutoProcessor
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul, QuickGELU
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm, LayerNorm
from vllm.model_executor.layers.lightning_attn import (
    lightning_attention2_parallel, linear_decode_forward_triton)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsMultiModal

from .minimax_cache import MinimaxCacheManager, MinimaxCacheParams
from .utils import PPMissingLayer, make_layers
from vllm.multimodal import MULTIMODAL_REGISTRY
from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from transformers import PretrainedConfig, CLIPVisionConfig
from vllm.inputs import InputProcessingContext

def replace_weight_name(name: str,
                        key: str = None,
                        to: str = None,
                        count: int = None,
                        prefix: str = None) -> str:
    name = name.replace(key, to) if count is None else \
        name.replace(key, to, count)
    return name


def weight_loader_with_alias(alias: str):

    def wrapper(func: callable):

        def inner_func(param: torch.Tensor,
                       loaded_weight: torch.Tensor,
                       *args,
                       prefix: str = None,
                       **kwargs):
            value = func(param, loaded_weight, *args, **kwargs)
            return value

        return inner_func

    return wrapper


class MiniMaxText01RMSNormTP(CustomOp):
    name = "MiniMaxText01RMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.ones(int(hidden_size /
                                                  self.tp_world)))

        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps
        return

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(
                variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)


class MiniMaxText01RotaryEmbedding(CustomOp):
    name = "MiniMaxText01RotaryEmbedding"

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool,
        cache_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.cache_dtype = cache_dtype
        cache = self._compute_cos_sin_cache().to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(
        self,
        base: Union[int, float],
    ) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        query_cast = query.to(self.cache_dtype)
        key_cast = key.to(self.cache_dtype)
        ops.rotary_embedding(positions, query_cast, key_cast, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)
        query = query_cast.to(query.dtype)
        key = key_cast.to(key.dtype)
        return query, key


class MiniMaxText01MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxText01MoE(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        layer_idx: int = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        self.quant_config = quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.weight.weight_loader = MiniMaxText01MoE.gate_weight_loader

        self.experts = FusedMoE(
            num_experts=self.num_total_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size * self.tp_size,
            params_dtype=self.params_dtype,
            reduce_results=True,
            renormalize=True,
            quant_config=self.quant_config,
            tp_size=self.tp_size,
            prefix=f"{prefix}.experts",
        )
        return

    @staticmethod
    def gate_weight_loader(param: nn.Parameter,
                           loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(
            hidden_states, router_logits_fp32.to(hidden_states.dtype))
        final_hidden = final_hidden_states.view(num_tokens, hidden_size)
        return final_hidden


class MiniMaxText01LinearKernel:

    @staticmethod
    def jit_linear_forward_prefix(q: torch.Tensor,
                                  k: torch.Tensor,
                                  v: torch.Tensor,
                                  kv_caches: torch.Tensor,
                                  slope_rate: torch.Tensor,
                                  block_size: int,
                                  layer_idx: int = None,
                                  **kwargs) -> torch.Tensor:

        slope_rate = slope_rate.to(torch.float32)
        should_pad_dim = q.dim() == 3
        if should_pad_dim:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = d
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output, kv_history = lightning_attention2_parallel(
            q, k, v, slope_rate, block_size=block_size, kv_history=kv_history)
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")


class MiniMaxText01LinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_inner_size: int,
        num_heads: int,
        head_dim: int,
        max_position: int,
        block_size: int,
        num_hidden_layer: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = 0,
        linear_layer_idx: int = 0,
        prefix: str = "linear_attn",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.BLOCK = block_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.hidden_inner_size = hidden_inner_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        assert self.total_num_heads % self.tp_size == 0
        self.tp_heads = self.total_num_heads // self.tp_size
        self.qkv_size = self.num_heads * self.head_dim
        self.tp_hidden = self.head_dim * self.tp_heads

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
            self.hidden_inner_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_inner_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.norm = MiniMaxText01RMSNormTP(
            self.hidden_inner_size,
            eps=1e-5,
        )

        slope_rate = MiniMaxText01LinearAttention._build_slope_tensor(
            self.num_heads)
        self.slope_rate = slope_rate * (1 - layer_idx /
                                        (num_hidden_layer - 1) + 1e-5)
        self.tp_slope = self.slope_rate[self.tp_rank *
                                        self.tp_heads:(self.tp_rank + 1) *
                                        self.tp_heads].contiguous()

    @staticmethod
    def weight_direct_load(param: torch.Tensor,
                           loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        return

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(n_attention_heads),
                              dtype=torch.float32).reshape(
                                  n_attention_heads, 1, 1)
        return slopes  # [h, 1, 1]

    def _prefill_and_mix_infer(self, q, k, v, kv_cache, state_indices_tensor,
                               attn_metadata):
        hidden = []
        for _prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
            _start = attn_metadata.query_start_loc[_prefill_idx]
            _end = attn_metadata.query_start_loc[_prefill_idx + 1]
            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slot_id = state_indices_tensor[_prefill_idx]
            slice_layer_cache = kv_cache[slot_id, ...]

            out_slice = MiniMaxText01LinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope,
                self.BLOCK,
                layer_idx=self.layer_idx)
            hidden.append(out_slice.contiguous())
        if attn_metadata.num_decode_tokens > 0:
            hidden.append(
                self._decode_infer(q, k, v, kv_cache, state_indices_tensor,
                                   attn_metadata))
        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor,
                      attn_metadata):
        q = q[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[attn_metadata.num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[getattr(attn_metadata, "num_prefills", 0
                                               ):]
        hidden = linear_decode_forward_triton(q, k, v, kv_cache, self.tp_slope,
                                              slot_id, 32)
        return hidden

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: MinimaxCacheParams,  # layer of tensor
            **kwargs) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv32 = qkv.to(torch.float32)
        qkvact = torch.nn.functional.silu(qkv32)
        qkvact = qkvact.view((qkv.shape[0], self.tp_heads, -1))
        q, k, v = torch.split(qkvact, [self.head_dim] * 3, dim=-1)
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        kv_cache = kv_caches.minimax_cache
        state_indices_tensor = kv_caches.state_indices_tensor

        decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
        if not decode_only:
            # prefill and mix
            hidden = self._prefill_and_mix_infer(q, k, v, kv_cache,
                                                 state_indices_tensor,
                                                 attn_metadata)
        else:
            # decode only
            hidden = self._decode_infer(q, k, v, kv_cache,
                                        state_indices_tensor, attn_metadata)

        hidden = self.norm._forward(hidden)
        gate, _ = self.output_gate(hidden_states)
        hidden = F.sigmoid(gate) * hidden
        hidden = hidden.to(hidden_states.dtype)
        hidden, _ = self.out_proj(hidden)
        return hidden


class MiniMaxText01Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rotary_dim: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "mha",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        return

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                **kwargs) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = attn_metadata.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxText01DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        expert_num: int = 1,  # moe or mlp
        layer_id: int = None,  # current layer index
        linear_layer_id: Optional[int] = None,
        prefix: str = "decoder",
    ) -> None:
        self._ilayer = layer_id
        self._irank = get_tensor_model_parallel_rank()
        super().__init__()

        self.hidden_size = config.hidden_size
        self.expert_num = expert_num

        rope_theta = getattr(config, "rope_theta", 10000)

        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        max_position_embeddings = config.max_position_embeddings
        if hasattr(config, "max_model_len") and isinstance(
                config.max_model_len, int):
            max_position_embeddings = min(config.max_position_embeddings,
                                          config.max_model_len)
        if config.attention_type == 0:
            use_headxdim = True
            hidden_inner = (head_dim * config.num_attention_heads
                            if use_headxdim else config.hidden_size)
            self.self_attn = MiniMaxText01LinearAttention(
                hidden_size=self.hidden_size,
                hidden_inner_size=hidden_inner,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                max_position=max_position_embeddings,
                block_size=config.block if hasattr(config, "block") else 256,
                num_hidden_layer=config.num_hidden_layers,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                linear_layer_idx=linear_layer_id,
                prefix=prefix)
        elif config.attention_type == 1:
            # 获取sliding_window属性，如果不存在则为None
            sliding_window = getattr(config, "sliding_window", None)
            self.self_attn = MiniMaxText01Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                rotary_dim=config.rotary_dim
                if hasattr(config, "rotary_dim") else head_dim,
                num_kv_heads=config.num_key_value_heads,
                max_position=max_position_embeddings,
                rope_theta=rope_theta,
                sliding_window=sliding_window,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                cache_config=cache_config,
                prefix=prefix)
        else:
            raise ValueError(
                f"Unsupported attention type: {self.config.attention_type}")

        if expert_num == 1:
            self.mlp = MiniMaxText01MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                layer_idx=self._ilayer,
                prefix=prefix)
        else:
            self.block_sparse_moe = MiniMaxText01MoE(
                num_experts=expert_num,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_idx=self._ilayer,
                quant_config=quant_config,
                prefix=prefix)

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        if config.attention_type == 0:
            self.layernorm_attention_alpha = getattr(
                config, 'layernorm_linear_attention_alpha', 1)
            self.layernorm_attention_beta = getattr(
                config, 'layernorm_linear_attention_beta', 1)
        else:
            self.layernorm_attention_alpha = getattr(
                config, 'layernorm_full_attention_alpha', 1)
            self.layernorm_attention_beta = getattr(
                config, 'layernorm_full_attention_beta', 1)
        self.layernorm_mlp_alpha = getattr(config, 'layernorm_mlp_alpha', 1)
        self.layernorm_mlp_beta = getattr(config, 'layernorm_mlp_beta', 1)
        self.postnorm = getattr(config, 'postnorm', False)
        self.shared_moe = False
        return

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: Union[List[Dict], Optional[
                torch.
                Tensor]],  # linear-attn / flash-attn(possible with warmup)
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            is_warmup: bool = False,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        layernorm_input = hidden_states
        layernorm_output = self.input_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input
        self_attention_output = self.self_attn(
            hidden_states=layernorm_output,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        residual = residual * self.layernorm_attention_alpha
        self_attention_output = (self_attention_output *
                                 self.layernorm_attention_beta)

        layernorm_input = residual + self_attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        residual = layernorm_output if self.postnorm else layernorm_input

        if self.expert_num == 1:
            hidden_states = self.mlp(layernorm_output)
        else:
            moe_hidden_states = self.block_sparse_moe(
                copy.deepcopy(layernorm_output))
            if self.shared_moe:

                # shared-moe part use all fp32 compute
                before_moe_dtype = layernorm_output.dtype
                moe_hidden_fp32 = moe_hidden_states.to(torch.float32)
                output_mlp = self.shared_mlp(layernorm_output).to(
                    torch.float32)

                # actually gate for shared moe
                coef, _ = self.coefficient(layernorm_output.to(torch.float32))

                if self.shared_moe_mode == 'softmax':
                    # TODO: require test.
                    coef = torch.nn.functional.softmax(coef, dim=-1)
                    hidden_states = moe_hidden_fp32 * (
                        1 - coef) + output_mlp * coef
                elif self.shared_moe_mode == 'sigmoid':
                    coef = torch.nn.functional.sigmoid(coef)
                    hidden_states = moe_hidden_fp32 * (
                        1 - coef) + output_mlp * coef

                # dtype cast back
                hidden_states = hidden_states.to(before_moe_dtype)
            else:
                hidden_states = moe_hidden_states

        residual = residual * self.layernorm_mlp_alpha
        hidden_states = hidden_states * self.layernorm_mlp_beta

        hidden_states = residual + hidden_states

        return hidden_states, None

    @staticmethod
    def shared_moe_coefficient_loader(param: torch.Tensor,
                                      loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


OPEN_DEBUG = False
class MiniMaxVL01Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        scheduler_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.decoder_attention_types = getattr(
            config, "attn_type_list", False) or getattr(
                config, "decoder_attention_types", False)
        if not self.decoder_attention_types:
            # by default, use self-attn
            self.decoder_attention_types = [1] * config.num_hidden_layers
        self.num_layers = config.num_hidden_layers

        self._layer_barrier = False
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def layer_fn(prefix):
            layer_idx = int(prefix.split('.')[-1])
            layer_config = config
            layer_config.attention_type = self.decoder_attention_types[
                layer_idx]
            layer_config.layer_idx = layer_idx

            decoder_kwargs = {
                "quant_config": quant_config,
                "layer_id": layer_idx,
                "cache_config": cache_config
            }

            if layer_config.attention_type == 0:
                decoder_kwargs["linear_layer_id"] = sum(
                    1 for i in range(layer_idx)
                    if self.decoder_attention_types[i] == 0)
            else:
                decoder_kwargs["linear_layer_id"] = None

            if hasattr(config, "num_local_experts") and isinstance(
                    config.num_local_experts, list):
                decoder_kwargs["expert_num"] = config.num_local_experts[
                    layer_idx]
            elif hasattr(config, "num_local_experts") and isinstance(
                    config.num_local_experts, int):
                decoder_kwargs["expert_num"] = config.num_local_experts
            else:
                decoder_kwargs["expert_num"] = 1

            return MiniMaxText01DecoderLayer(layer_config,
                                             **decoder_kwargs,
                                             prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, layer_fn, prefix=f"{prefix}.layers")

        linear_layer_nums = sum(1 for i in range(config.num_hidden_layers)
                                if self.decoder_attention_types[i] == 0)
        max_slots_number = scheduler_config.max_num_seqs
        self.cache_shape = (linear_layer_nums, max_slots_number,
                            config.num_attention_heads //
                            get_tensor_model_parallel_world_size(),
                            config.head_dim, config.head_dim)
        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        self.minimax_cache = MinimaxCacheManager(dtype=self._dtype,
                                                 cache_shape=self.cache_shape)

        rope_theta = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        
        # 修复：确保max_position_embeddings变量被正确定义
        max_position_embeddings = config.max_position_embeddings
        if hasattr(config, "max_model_len") and isinstance(
                config.max_model_len, int):
            max_position_embeddings = min(config.max_position_embeddings,
                                          config.max_model_len)
            
        self.rotary_emb = MiniMaxText01RotaryEmbedding(
            head_dim,
            rotary_dim=config.rotary_dim
            if hasattr(config, "rotary_dim") else head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            cache_dtype=torch.float32,
        )

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            self.norm = PPMissingLayer()
        self.embed_scale = 1.0
        return

    def _clear_prefill_cache(self, attn_metadata,
                             minimax_cache_tensors: torch.Tensor, **kwargs):
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in (
                self.minimax_cache.cache_indices_mapping.items()):
            seq_to_slot_maps.update(seq_to_slot_map)

        slots_to_clear = []
        for _prefill_id in range(getattr(attn_metadata, "num_prefills", 0)):
            seq_id = seq_id_map[_prefill_id]
            if attn_metadata.context_lens_tensor[
                    _prefill_id] == 0 and seq_id in seq_to_slot_maps:
                slots_to_clear.append(seq_to_slot_maps[seq_id])

        if slots_to_clear:
            slots_tensor = torch.tensor(slots_to_clear,
                                        device=minimax_cache_tensors.device,
                                        dtype=torch.long)
            minimax_cache_tensors[:, slots_tensor, ...] = 0

    def forward(
                self,
                input_ids: Optional[torch.Tensor],
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                intermediate_tensors=None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return None
        if "request_ids_to_seq_ids" not in kwargs:
            kwargs["request_ids_to_seq_ids"] = {}
        if "finished_requests_ids" not in kwargs:
            kwargs["finished_requests_ids"] = []
        (
            minimax_cache_tensors,
            state_indices_tensor,
        ) = self.minimax_cache.current_run_tensors(input_ids, attn_metadata,
                                                   **kwargs)
        if getattr(attn_metadata, "num_prefills", 0) > 0:
            self._clear_prefill_cache(attn_metadata, minimax_cache_tensors,
                                      **kwargs)

        minimax_cache_params = MinimaxCacheParams(minimax_cache_tensors,
                                                  state_indices_tensor)
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embed_scale * self.embed_tokens(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        kv_cache_index = 0
        minimax_cache_index = 0
        attn_metadata.rotary_emb = self.rotary_emb
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _caches = None
            if isinstance(layer.self_attn, MiniMaxText01Attention):
                _caches = kv_caches[kv_cache_index]
                kv_cache_index += 1
            if isinstance(layer.self_attn, MiniMaxText01LinearAttention):
                current_state_layer = minimax_cache_index
                _caches = minimax_cache_params.at_layer_idx(
                    current_state_layer)
                minimax_cache_index += 1
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                kv_caches=_caches,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states



class MinimaxVLProcessingInfo:
    """MiniMax VL 模型的处理信息类"""
    
    def __init__(self, ctx: InputProcessingContext):
        self.ctx = ctx
    
    def get_hf_config(self) -> PretrainedConfig:
        """直接返回原始配置，不进行类型检查"""
        return self.ctx.hf_config
    
    def get_tokenizer(self) -> Any:
        """获取模型的分词器"""
        return self.ctx.tokenizer
    
    def get_hf_processor(self, **kwargs: object) -> Any:
        """获取或初始化处理器"""
        # 直接使用AutoProcessor加载MiniMax自己的处理器
        return self.ctx.init_processor(
            AutoProcessor,
            trust_remote_code=True,
            **kwargs,
        )
    
    def get_image_size_with_most_features(self) -> Tuple[int, int]:
        """获取最大特征的图像尺寸"""
        # 从MiniMax模型配置中获取图像尺寸
        config = self.get_hf_config()
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
            return config.vision_config.image_size, config.vision_config.image_size
        # 备选：从text_config中获取
        elif hasattr(config, "text_config") and hasattr(config.text_config, "vision_image_size"):
            size = config.text_config.vision_image_size
            return size, size
        # 默认值
        return 336, 336
    
    def get_max_image_tokens(self) -> int:
        """获取每个图像的最大令牌数"""
        width, height = self.get_image_size_with_most_features()
        # 尝试从配置中获取patch_size
        config = self.get_hf_config()
        patch_size = 14  # 默认值
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            patch_size = config.vision_config.patch_size
        return (width // patch_size) * (height // patch_size)
    
    def get_mm_max_tokens_per_item(self, model_config: Dict[str, Any]) -> Dict[str, int]:
        """获取每种模态每项的最大令牌数"""
        return {"image": self.get_max_image_tokens()}


class MinimaxVLProcessor:
    """MiniMax VL 模型的多模态处理器"""
    
    def __init__(self, ctx: InputProcessingContext):
        self.ctx = ctx
        self.info = MinimaxVLProcessingInfo(ctx)
        
    # 处理聊天模板的方法
    def apply_chat_template(self, messages, **kwargs):
        """应用聊天模板到消息列表"""
        return self.info.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
    # 实现其他必要的处理方法...

# 修改处理器注册方式
@MULTIMODAL_REGISTRY.register_processor(
    MinimaxVLProcessor, 
    info=MinimaxVLProcessingInfo,
    dummy_inputs={"image": ["<image>"]})  # 添加缺少的dummy_inputs参数
class LlavaForConditionalGeneration(MiniMaxVL01Model, SupportsMultiModal):
    """MiniMax VL 模型，支持多模态处理"""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        scheduler_config=None,
        prefix: str = "",
    ) -> None:
        # 首先调用父类的初始化方法
        super().__init__(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            prefix=prefix
        )
        
        self.quant_config = quant_config
        
        # 保存配置
        self.config = config
        
        # 检查是否有图像令牌索引（image_token_index）
        self.image_token_id = getattr(config, "image_token_index", None)
        
        # 标记此模型是否具有视觉塔
        self.has_vision_tower = hasattr(config, "vision_config")
        
        # 添加视觉塔和多模态投影器
        if self.has_vision_tower:
            # 创建视觉配置
            vision_config = CLIPVisionConfig(
                hidden_size=config.vision_config.hidden_size,
                image_size=config.vision_config.image_size,
                patch_size=config.vision_config.patch_size,
                num_hidden_layers=config.vision_config.num_hidden_layers,
                num_attention_heads=config.vision_config.num_attention_heads,
                intermediate_size=config.vision_config.intermediate_size,
                projection_dim=config.vision_config.projection_dim,
            )
            
            # 初始化视觉模型
            self.vision_tower = MinimaxVLVisionTransformer(
                vision_config=vision_config,
                quant_config=quant_config,
                prefix=f"{prefix}.vision_tower"
            )
            
            # 添加多模态投影器
            self.multi_modal_projector = MinimaxVLMultiModalProjector(
                vision_config=vision_config,
                text_config=config.text_config,
                quant_config=quant_config,
                prefix=f"{prefix}.multi_modal_projector"
            )
        
        # 初始化lm_head
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.hidden_size,
                self.vocab_size,
                org_num_embeddings=self.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = PPMissingLayer()
        
        # 为词汇嵌入和语言模型头设置自定义权重加载器
        if get_pp_group().is_first_rank and hasattr(self, 'embed_tokens'):
            self.embed_tokens.weight.weight_loader = self.embed_tokens_weight_loader
            
        if get_pp_group().is_last_rank and hasattr(self, 'lm_head'):
            self.lm_head.weight.weight_loader = self.lm_head_weight_loader
    
    @staticmethod
    def embed_tokens_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """自定义的词汇嵌入权重加载器，处理词汇大小不匹配的情况"""
        # 获取张量模型并行世界大小和排名
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        # 检查维度兼容性
        if param.dim() != loaded_weight.dim():
            raise ValueError(f"Dimension mismatch: param dim {param.dim()}, loaded weight dim {loaded_weight.dim()}")
        
        # 检查隐藏维度是否匹配
        if param.shape[-1] != loaded_weight.shape[-1]:
            raise ValueError(f"Hidden size mismatch: param shape {param.shape}, loaded weight shape {loaded_weight.shape}")
        
        # 计算每个分片的大小
        vocab_size = loaded_weight.shape[0]
        shard_size = vocab_size // tp_size
        
        # 计算当前分片的范围
        start_idx = tp_rank * shard_size
        end_idx = min((tp_rank + 1) * shard_size, vocab_size)
        
        # 复制权重到参数
        if start_idx < vocab_size:
            # 确保我们只复制参数能容纳的部分
            copy_size = min(end_idx - start_idx, param.shape[0])
            param[:copy_size].data.copy_(loaded_weight[start_idx:start_idx + copy_size])
    
    @staticmethod
    def lm_head_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """自定义的语言模型头权重加载器，处理词汇大小不匹配的情况"""
        # 获取张量模型并行世界大小和排名
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        # 检查是否需要转置权重
        if loaded_weight.shape[0] == param.shape[1] and loaded_weight.shape[1] == param.shape[0] * tp_size:
            loaded_weight = loaded_weight.transpose(0, 1)
        
        # 计算每个分片的大小
        vocab_size = loaded_weight.shape[0]
        shard_size = vocab_size // tp_size
        
        # 计算当前分片的范围
        start_idx = tp_rank * shard_size
        end_idx = min((tp_rank + 1) * shard_size, vocab_size)
        
        # 复制权重到参数
        if start_idx < vocab_size:
            # 确保我们只复制参数能容纳的部分
            copy_size = min(end_idx - start_idx, param.shape[0])
            
            # 处理隐藏维度不匹配的情况
            if param.shape[1] != loaded_weight.shape[1]:
                # 创建一个新的权重张量，大小与参数匹配
                new_weight = torch.zeros(
                    loaded_weight.shape[0], 
                    param.shape[1],
                    device=loaded_weight.device, 
                    dtype=loaded_weight.dtype
                )
                
                # 复制加载的权重到新权重的前部分
                min_dim = min(param.shape[1], loaded_weight.shape[1])
                new_weight[:, :min_dim] = loaded_weight[:, :min_dim]
                
                # 使用新权重进行复制
                param[:copy_size].data.copy_(new_weight[start_idx:start_idx + copy_size])
            else:
                # 如果隐藏维度匹配，直接复制
                param[:copy_size].data.copy_(loaded_weight[start_idx:start_idx + copy_size])

    def make_empty_intermediate_tensors(self) -> IntermediateTensors:
        """创建空的中间张量，用于模型并行处理。"""
        return IntermediateTensors({
            "hidden_states": None,
            "residual": None
        })
    
    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, prefix: str = ""):
        """从vllm_config创建模型实例的工厂方法"""
        return cls(
            config=vllm_config.model_config.hf_config,
            quant_config=vllm_config.quant_config,
            cache_config=vllm_config.cache_config,
            scheduler_config=vllm_config.scheduler_config,
            prefix=prefix
        )
    
    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        """获取多模态嵌入"""
        # 如果没有视觉塔，返回None
        if not self.has_vision_tower:
            return None
            
        # 解析和验证图像输入
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
            
        # 处理图像输入
        return self._process_image_input(image_input)
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """获取输入嵌入，包括文本和可能的视觉嵌入"""
        # 获取文本嵌入
        inputs_embeds = self.embed_scale * self.embed_tokens(input_ids)
        
        # 如果没有多模态嵌入，直接返回文本嵌入
        if multimodal_embeddings is None:
            return inputs_embeds
            
        # 如果有多模态嵌入，将其与文本嵌入合并
        # 找到图像标记的位置
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        
        if len(image_positions) == 0:
            # 没有图像标记，直接返回文本嵌入
            return inputs_embeds
            
        # 替换图像标记位置的嵌入
        for i, pos in enumerate(image_positions):
            if i < len(multimodal_embeddings):
                # 获取当前图像的嵌入
                image_embedding = multimodal_embeddings[i:i+1]
                
                # 替换文本嵌入中的图像标记
                inputs_embeds[pos:pos+1] = image_embedding
                
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """前向传播，处理输入并生成输出"""
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            # 只有在有视觉塔的情况下才获取多模态嵌入
            if self.has_vision_tower:
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
                input_ids = None
            
        return super().forward(
            input_ids=input_ids, 
            positions=positions, 
            kv_caches=kv_caches,
            intermediate_tensors=intermediate_tensors, 
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """计算logits"""
        if get_pp_group().is_last_rank:
            return self.lm_head(hidden_states)
        return None
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """采样生成下一个token"""
        if get_pp_group().is_last_rank:
            sampler = Sampler()
            return sampler(logits, sampling_metadata)
        return None
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """自定义权重加载函数，处理权重路径映射问题"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        
        # 创建参数字典
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        
        # 创建权重名称映射
        name_mapping = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight"
        }
        
        # 记录调试信息
        if OPEN_DEBUG:
            print(f"Model has {len(params_dict)} parameters")
            print(f"Loading {len(list(weights))} weights")
        
        # 特殊处理lm_head权重
        lm_head_weight = None
        lm_head_name = None
        
        for name, loaded_weight in weights:
            # 检查是否是lm_head权重
            if "lm_head.weight" in name:
                lm_head_weight = loaded_weight
                lm_head_name = name
                continue
            
            # 检查是否需要重新映射名称
            mapped_name = name
            for old_name, new_name in name_mapping.items():
                if name == old_name or name.endswith(old_name):
                    mapped_name = name.replace(old_name, new_name)
                    break
            
            # 处理堆叠参数
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                mapped_name = mapped_name.replace(weight_name, param_name)
                
                # 跳过不存在的参数
                if mapped_name not in params_dict:
                    if OPEN_DEBUG:
                        print(f"Skipping {mapped_name} (not found in model)")
                    continue
                
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped_name)
                except Exception as e:
                    if OPEN_DEBUG:
                        print(f"Error loading {mapped_name}: {e}")
                        print(f"Param shape: {param.shape}, Weight shape: {loaded_weight.shape}")
                    raise
                break
            else:
                # 处理专家参数映射
                if "rotary_emb.inv_freq" in mapped_name:
                    continue
                
                # 跳过不存在的参数
                if mapped_name not in params_dict:
                    if OPEN_DEBUG:
                        print(f"Skipping {mapped_name} (not found in model)")
                    continue
                
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped_name)
                except Exception as e:
                    if OPEN_DEBUG:
                        print(f"Error loading {mapped_name}: {e}")
                        print(f"Param shape: {param.shape}, Weight shape: {loaded_weight.shape}")
                    raise
        
        # 特殊处理lm_head权重
        if lm_head_weight is not None and "lm_head.weight" in params_dict:
            param = params_dict["lm_head.weight"]
            weight_loader = getattr(param, "weight_loader", self.lm_head_weight_loader)
            
            try:
                # 使用自定义的权重加载器
                weight_loader(param, lm_head_weight)
                loaded_params.add("lm_head.weight")
            except Exception as e:
                if OPEN_DEBUG:
                    print(f"Error loading lm_head.weight: {e}")
                    print(f"Param shape: {param.shape}, Weight shape: {lm_head_weight.shape}")
                raise
        
        return loaded_params

    def _parse_and_validate_image_input(self, **kwargs) -> Optional[Dict[str, torch.Tensor]]:
        """解析并验证图像输入"""
        pixel_values = kwargs.get("pixel_values", None)
        image_embeds = kwargs.get("image_embeds", None)
        
        if pixel_values is None and image_embeds is None:
            return None
            
        if pixel_values is not None:
            # 验证和处理像素值
            if not isinstance(pixel_values, torch.Tensor):
                # 尝试自动处理图像
                processor = MinimaxVLImageProcessor()
                pixel_values = processor.preprocess_images(pixel_values)
                
            if pixel_values.ndim != 4:
                raise ValueError(f"pixel_values应该有4个维度 [batch_size, channels, height, width]，但现在是 {pixel_values.shape}")
                
            return {"type": "pixel_values", "data": pixel_values}
            
        if image_embeds is not None:
            # 验证图像嵌入
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError(f"image_embeds应该是torch.Tensor，但现在是 {type(image_embeds)}")
                
            return {"type": "image_embeds", "data": image_embeds}
            
        return None
        
    def _process_image_input(self, image_input):
        """处理图像输入，转换为嵌入"""
        if image_input["type"] == "image_embeds":
            # 嵌入已经准备好
            return image_input["data"]
            
        # 像素值需要通过视觉塔转换为嵌入
        pixel_values = image_input["data"]
        image_features = self.vision_tower(pixel_values)
        
        # 选择图像特征（如果需要，跳过CLS token）
        image_features = image_features[:, 1:, :]  # 跳过CLS token
        
        # 通过投影器转换为文本空间
        image_embeds = self.multi_modal_projector(image_features)
        
        return image_embeds


class MinimaxVLLayerNorm(nn.Module):
    """视觉模型的标准化层"""
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MinimaxVLAttention(nn.Module):
    """视觉模型的注意力模块"""
    
    def __init__(self, vision_config):
        super().__init__()
        self.embed_dim = vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # 初始化q,k,v投影
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        # 生成q,k,v
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 改变形状以适应多头注意力
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class MinimaxVLMLP(nn.Module):
    """视觉模型的MLP模块"""
    
    def __init__(self, vision_config):
        super().__init__()
        self.fc1 = nn.Linear(vision_config.hidden_size, vision_config.intermediate_size)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(vision_config.intermediate_size, vision_config.hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MinimaxVLEncoderLayer(nn.Module):
    """视觉编码器的一个层"""
    
    def __init__(self, vision_config):
        super().__init__()
        self.embed_dim = vision_config.hidden_size
        self.self_attn = MinimaxVLAttention(vision_config)
        self.layer_norm1 = MinimaxVLLayerNorm(self.embed_dim)
        self.mlp = MinimaxVLMLP(vision_config)
        self.layer_norm2 = MinimaxVLLayerNorm(self.embed_dim)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MinimaxVLVisualEmbedding(nn.Module):
    """视觉嵌入层，将图像转换为序列嵌入"""
    
    def __init__(self, vision_config, device=None, dtype=None):
        super().__init__()
        self.config = vision_config
        self.embed_dim = vision_config.hidden_size
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(self.num_positions, self.embed_dim))
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        
        # 获取图像特征
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, hidden_size, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # shape = [batch_size, n_patches, hidden_size]
        
        # 添加分类embedding（作为全局特征）
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        
        # 添加位置embedding
        embeddings = embeddings + self.position_embedding
        
        return embeddings


class MinimaxVLVisionTransformer(nn.Module):
    """完整的视觉Transformer编码器"""
    
    def __init__(self, vision_config, quant_config=None, prefix=""):
        super().__init__()
        self.config = vision_config
        self.embed_dim = vision_config.hidden_size
        
        # 嵌入层
        self.embeddings = MinimaxVLVisualEmbedding(vision_config)
        
        # 层标准化
        self.pre_layrnorm = MinimaxVLLayerNorm(self.embed_dim)
        
        # Transformer编码器层
        self.encoder = nn.ModuleList([
            MinimaxVLEncoderLayer(vision_config) 
            for _ in range(vision_config.num_hidden_layers)
        ])
        
        # 最终层标准化
        self.post_layernorm = MinimaxVLLayerNorm(self.embed_dim)

    def forward(self, pixel_values):
        # [B, C, H, W] -> [B, L, D]
        hidden_states = self.embeddings(pixel_values)
        
        hidden_states = self.pre_layrnorm(hidden_states)
        
        # 通过每个编码器层
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
            
        hidden_states = self.post_layernorm(hidden_states)
        
        return hidden_states


class MinimaxVLMultiModalProjector(nn.Module):
    """将视觉特征映射到文本空间的投影器"""
    
    def __init__(self, vision_config, text_config, quant_config=None, prefix=""):
        super().__init__()
        self.vision_hidden_size = vision_config.hidden_size
        self.text_hidden_size = text_config.hidden_size
        
        self.linear_1 = ColumnParallelLinear(
            self.vision_hidden_size,
            self.text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1"
        )
        
        self.act = QuickGELU()
        
        self.linear_2 = RowParallelLinear(
            self.text_hidden_size,
            self.text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2"
        )

    def forward(self, x):
        hidden_states, _ = self.linear_1(x)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MinimaxVLImageProcessor:
    """图像预处理工具类"""
    
    def __init__(self):
        pass
        
    def preprocess_images(self, images):
        """处理输入图像，转换为模型期望的格式"""
        # 此处根据实际需求实现图像处理逻辑
        # 如果输入已经是张量，可能需要进行验证和规范化
        if isinstance(images, torch.Tensor):
            return images
            
        # 如果是其他格式，可能需要转换为张量
        # 这里简单返回None作为占位符
        return None


class MinimaxVLProcessingInfo:
    """处理信息类，提供处理器所需的配置信息"""
    
    def __init__(self, ctx):
        self.ctx = ctx
        
    def get_tokenizer(self):
        """获取分词器"""
        return self.ctx.get_tokenizer()
        
    def get_hf_config(self):
        """获取HuggingFace配置"""
        return self.ctx.get_hf_config()
