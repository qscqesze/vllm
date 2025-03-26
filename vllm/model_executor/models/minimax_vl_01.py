# SPDX-License-Identifier: Apache-2.0
""" PyTorch MiniMax model."""
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Optional, Set, Tuple, TypedDict, Union, Callable

import torch
import torch.nn as nn
from transformers import (BatchFeature, MiniMaxConfig, MiniMaxImageProcessor,
                          MiniMaxProcessor)
from functools import partial

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear, 
                                              RowParallelLinear,
                                              ReplicatedLinear)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.resampler import Resampler2, get_abs_pos
from vllm.model_executor.layers.quantization import QuantizationConfig

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)
from .minimax_text_01 import MiniMaxText01ForCausalLM

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 71011
_NEWLINE_TOKEN_ID = 71019


class MiniMaxImagePatchInputs(TypedDict):
    type: Literal["image_patches"]
    flat_data: torch.Tensor
    """
    Shape: 
    `(batch_size * num_patches, patch_size_x * patch_size_y * num_channels)`
    """

    patches_per_image: list[int]
    """
    The number of total patches for each image in the batch.

    This is used to split the embeddings which has the first two dimensions
    flattened just like `flat_data`.
    """


class MiniMaxProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(MiniMaxConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(MiniMaxProcessor, **kwargs)

    def get_image_processor(self) -> MiniMaxImageProcessor:
        return self.get_hf_processor().image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        target_width, target_height = self.get_image_size_with_most_features()

        max_ncols, max_nrows = self.get_image_feature_grid_size(
            image_width=target_width,
            image_height=target_height,
        )
        max_image_tokens = (max_ncols + 1) * max_nrows

        return {"image": max_image_tokens}

    def get_image_feature_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        image_processor = self.get_image_processor()
        target_width = image_processor.size["width"]
        target_height = image_processor.size["height"]
        patch_width = image_processor.patch_size["width"]
        patch_height = image_processor.patch_size["height"]

        if not (image_width <= target_width and image_height <= target_height):
            height_scale_factor = target_height / image_height
            width_scale_factor = target_width / image_width
            optimal_scale_factor = min(height_scale_factor, width_scale_factor)

            image_height = int(image_height * optimal_scale_factor)
            image_width = int(image_width * optimal_scale_factor)

        ncols = math.ceil(image_width / patch_width)
        nrows = math.ceil(image_height / patch_height)
        return ncols, nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(width=image_processor.size["width"],
                         height=image_processor.size["height"])


class MiniMaxDummyInputsBuilder(BaseDummyInputsBuilder[MiniMaxProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="",
            mm_data=mm_data,
        )


class MiniMaxMultiModalProcessor(BaseMultiModalProcessor[MiniMaxProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # Avoid warning from HF logger for text-only input
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        image_patches = processed_outputs.get("image_patches")
        if image_patches is not None:
            images = mm_data["images"]
            assert isinstance(images, list)

            # Original output: (1, num_images, Pn, Px * Py * C)
            # New output: (num_images, Pn, Px * Py * C)
            assert (isinstance(image_patches, list)
                    and len(image_patches) == 1)
            assert (isinstance(image_patches[0], torch.Tensor)
                    and len(image_patches[0]) == len(images))

            processed_outputs["image_patches"] = image_patches[0]

        return processed_outputs

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        # HF processor adds boa_token_id
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        boa_token_id = vocab["<0x04>"]

        return prompt_tokens + [boa_token_id]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(image_patches=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        bos_token_id = hf_config.bos_token_id
        assert isinstance(bos_token_id, int)

        tokenizer = self.info.get_tokenizer()
        eot_token_id = tokenizer.bos_token_id
        assert isinstance(eot_token_id, int)

        def get_replacement_MiniMax(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = self.info.get_image_feature_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            image_tokens = ([_IMAGE_TOKEN_ID] * ncols +
                            [_NEWLINE_TOKEN_ID]) * nrows

            return PromptUpdateDetails(
                full=image_tokens + [bos_token_id],
                features=image_tokens,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[eot_token_id],
                replacement=get_replacement_MiniMax,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(MiniMaxMultiModalProcessor,
                                        info=MiniMaxProcessingInfo,
                                        dummy_inputs=MiniMaxDummyInputsBuilder)
class VisualAttention(nn.Module):
    """self-attention layer class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim \
            and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, \
                'Visual Attention implementation only supports self-attention'
        self.in_proj = ReplicatedLinear(embed_dim, 3 * embed_dim)
        self.out_proj = ReplicatedLinear(embed_dim, embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query/key/value: [sq, b, h]
        sq, b, _ = x.size()
        mixed_x_layer, _ = self.in_proj(x)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            self.hidden_size_per_attention_head, dim=-1)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            attention_probs = torch.baddbmm(attn_mask, q_scaled,
                                            key_layer.transpose(-2, -1))
        else:
            attention_probs = torch.bmm(q_scaled, key_layer.transpose(-2, -1))
        attention_probs = attention_probs.softmax(dim=-1)

        value_layer = value_layer.view(
            sq, b * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head).transpose(0, 1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(
            b, self.num_attention_heads_per_partition, sq,
            self.hidden_size_per_attention_head)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output, _ = self.out_proj(context_layer)

        return output


class MiniMaxVLMLP(nn.Module):
    """MLP for the visual component of the MiniMax model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.c_fc = ColumnParallelLinear(hidden_size,
                                         intermediate_size,
                                         bias=True,
                                         quant_config=quant_config)
        self.act_fn = get_act_fn("gelu")
        self.c_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x):
        x, _ = self.c_fc(x)
        x = self.act_fn(x)
        x, _ = self.c_proj(x)
        return x


class VisualAttentionBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = MiniMaxVLMLP(
            hidden_size=d_model,
            intermediate_size=mlp_width,
            quant_config=quant_config,
        )

    def attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, attn_mask=attn_mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(width,
                                 heads,
                                 mlp_ratio,
                                 norm_layer=norm_layer,
                                 quant_config=quant_config)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 n_queries: int = 256,
                 output_dim: int = 512,
                 quant_config: Optional[QuantizationConfig] = None,
                 **kwargs):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height,
                          image_width // patch_width)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale *
                                                 torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(width,
                                            layers,
                                            heads,
                                            mlp_ratio,
                                            norm_layer=norm_layer,
                                            quant_config=quant_config)

        self.attn_pool = Resampler2(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
            adaptive=False,
            do_post_projection=False,
        )

        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter(
            (output_dim**-0.5) * torch.randn(output_dim, output_dim))

        self.image_token_id = _IMAGE_TOKEN_ID
        self.newline_token_id = _NEWLINE_TOKEN_ID

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )

        # to patches
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + get_abs_pos(self.positional_embedding, int(math.sqrt(
            x.size(1))))

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj

        return x


class MiniMaxImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, 3, image_size, image_size)`
    """


class MiniMaxImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, 256, hidden_size)`
    """


MiniMaxImageInputs = Union[MiniMaxImagePixelInputs, MiniMaxImageEmbeddingInputs]


class MiniMaxVL01ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.vocab_size = config.text_config.vocab_size
        self.image_token_id = _IMAGE_TOKEN_ID
        self.image_feature_size = config.patch_size**2 * config.num_channels

        # 添加视觉模型
        self.visual = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            width=config.hidden_size,
            layers=config.num_hidden_layers,
            heads=config.num_attention_heads,
            mlp_ratio=4.0,
            output_dim=config.hidden_size,
            quant_config=quant_config,
        )
        
        self.vision_embed_tokens = ColumnParallelLinear(
            self.image_feature_size,
            config.hidden_size,
            quant_config=quant_config,
            gather_output=True,
        )
        self.language_model = MiniMaxText01ForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @property
    def sampler(self):
        return self.language_model.sampler

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[MiniMaxImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_patches = kwargs.pop("image_patches", None)
        
        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return MiniMaxImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return MiniMaxImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )
            
        if image_patches is not None:
            if not isinstance(image_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image patches. "
                                 f"Got type: {type(image_patches)}")

            image_patches_flat = flatten_bn(image_patches)

            return MiniMaxImagePatchInputs(
                type="image_patches",
                flat_data=self._validate_pixel_values(
                    flatten_bn(image_patches_flat, concat=True)),
                patches_per_image=[x.size(0) for x in image_patches_flat],
            )

        return None

    def _process_image_input(
            self, image_input: MiniMaxImageInputs) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        elif image_input["type"] == "pixel_values":
            return self.visual(image_input["data"])
        elif image_input["type"] == "image_patches":
            image_patches_flat = image_input["flat_data"]
            patches_per_image = image_input["patches_per_image"]

            assert self.vision_embed_tokens is not None
            vision_embeddings_flat, _ = self.vision_embed_tokens(
                image_patches_flat)
            return vision_embeddings_flat.split(patches_per_image, dim=0)

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                _IMAGE_TOKEN_ID)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

# 将 AbabForCausalLM 设置为 MiniMaxVL01ForCausalLM 的别名
AbabForCausalLM = MiniMaxVL01ForCausalLM
