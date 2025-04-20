"""PyTorch Wav2Vec-S model."""

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.onnx.operators
from torch import Tensor, nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2NoLayerNormConvLayer,
    Wav2Vec2LayerNormConvLayer,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2EncoderLayer,
    Wav2Vec2Encoder,
    Wav2Vec2Model,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2Attention,
    Wav2Vec2FlashAttention2
)
from transformers import Wav2Vec2Config, Cache, DynamicCache
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from .configuration_wav2vec_s import Wav2VecSConfig


def gen_block_atten_mask(x, padding_mask, main_context=1, right_context=0):
    """
    Args:
        x: inpout embedding, TxBxC
    """
    x = x.transpose(0, 1)
    if padding_mask is None:
        padding_mask = x.new_zeros((x.size(1), x.size(0)), dtype=torch.bool)
    
    bsz, seq_len = padding_mask.shape 
    block_num = seq_len // main_context
    # block_idx = torch.arange(seq_len).to(padding_mask.device) // main_context
    block_idx = torch.div(torch.arange(seq_len).to(padding_mask.device), main_context, rounding_mode='floor')

    if right_context == 0:
        attn_mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)
    else:
        with torch.no_grad():
            rc_block_idx = torch.arange(block_num)
            rc_block_pos = rc_block_idx.unsqueeze(1).repeat(1, right_context).view(-1).to(padding_mask.device)
            rc_blcok_step = (rc_block_idx.unsqueeze(1) + 1) * main_context

            rc_inc_idx = torch.arange(right_context).unsqueeze(0)
            rc_idx = (rc_blcok_step + rc_inc_idx).view(-1).to(padding_mask.device)
            rc_idx_mask = (rc_idx > (seq_len -1)).to(padding_mask)
            rc_idx = rc_idx.clamp(0, seq_len -1)
            
            rc_padding_mask = padding_mask.index_select(1, rc_idx)
            rc_padding_mask= rc_padding_mask | rc_idx_mask.unsqueeze(0)
            padding_mask = torch.cat((padding_mask, rc_padding_mask), dim=1)
            
            full_idx = torch.cat((block_idx, rc_block_pos), dim= 0)
            attn_mask1 = full_idx.unsqueeze(1) < block_idx.unsqueeze(0)
            attn_mask2= full_idx.unsqueeze(1).ne(rc_block_pos.unsqueeze(0))
            attn_mask = torch.cat([attn_mask1, attn_mask2], dim=1)

        rc_x = x.index_select(0, rc_idx)
        x = torch.cat((x, rc_x), dim= 0)

    attn_mask_float = x.new(attn_mask.shape).fill_(0)
    attn_mask_float = attn_mask_float.masked_fill(attn_mask.to(torch.bool), -1e4)

    padding_mask = padding_mask.to(torch.bool)
    attn_mask = attn_mask_float.masked_fill(padding_mask.unsqueeze(1), -1e4)

    x = x.transpose(0, 1)

    return x, attn_mask.unsqueeze(1), padding_mask


@dataclass
class Wav2VecSBaseModelOutput(ModelOutput):
    """
    Base class for models that have been trained with the Wav2Vec2 loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


class Wav2VecSFeatureEncoder(Wav2Vec2FeatureEncoder):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__(config)

        if config.num_hidden_layers == 12:
            conv_layers = [Wav2Vec2LayerNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        else:
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        self.conv_layers = nn.ModuleList(conv_layers)


class Wav2VecSAttention(Wav2Vec2Attention):
    """Block-wise Multi-headed attention"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
            is_causal=is_causal,
            config=config
        )

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[self.layer_idx][0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[self.layer_idx][0]
            value_states = past_key_value[self.layer_idx][1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped



class Wav2VecSSdpaAttention(Wav2VecSAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions or layer_head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Wav2Vec2Model is using Wav2Vec2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[self.layer_idx][0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[self.layer_idx][0]
            value_states = past_key_value[self.layer_idx][1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        query_states = self._shape(query_states, tgt_len, bsz)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None

WAV2VECS_ATTENTION_CLASSES = {
    "eager": Wav2VecSAttention,
    "sdpa": Wav2VecSSdpaAttention,
    "flash_attention_2": Wav2Vec2FlashAttention2,
}

class Wav2VecSEncoderLayer(Wav2Vec2EncoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.attention = WAV2VECS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Cache] =None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights = self.attention(
            hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2VecSEncoderLayerStableLayerNorm(Wav2Vec2EncoderLayerStableLayerNorm):
    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.attention = WAV2VECS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Cache] =None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2VecSEncoder(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        max_source_positions = 8000
        padding_idx=1
        self.pos_conv_embed = SinusoidalPositionalEmbedding(
            config.hidden_size,
            padding_idx=padding_idx,
            init_size=max_source_positions + padding_idx + 1,
        )

        self.layers = nn.ModuleList(
            [Wav2VecSEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.context_type = config.context_type
        self.main_context = config.main_context
        self.right_context = config.right_context

    def forward(
        self,
        hidden_states: torch.tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_cache: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            seq_len = ~(attention_mask.clone())
            paddding_mask = ~(attention_mask.clone())
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        else:
            seq_len = torch.zeros((hidden_states.size(0), hidden_states.size(1)),dtype=torch.bool).to(hidden_states.device)
            paddding_mask = None

        if self.training or past_key_values is None:
            position_embeddings = self.pos_conv_embed(seq_len)
        else:
            past_size = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_len = torch.zeros((hidden_states.size(0), past_size + hidden_states.size(1)),dtype=torch.bool).to(hidden_states.device)
            position_embeddings = self.pos_conv_embed(seq_len)[:,past_size:]

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        curr_frames = hidden_states.size(1)
        if self.training or not use_cache:
            hidden_states, attention_mask, padding_mask = gen_block_atten_mask(
                hidden_states, 
                paddding_mask, 
                main_context=self.main_context, 
                right_context=self.right_context
            )

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, 
                        past_key_value=past_key_values, 
                        attention_mask=attention_mask, 
                        output_attentions=output_attentions
                    )

                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                layer_attn = layer_outputs[1]
                if attention_mask is not None:
                    layer_attn = layer_attn[:,:curr_frames,:curr_frames]
                
                if not self.training and self.right_context > 0:
                    layer_attn = layer_attn[:,:-self.right_context,:-self.right_context]
                
                all_self_attentions = all_self_attentions + (layer_attn,)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if attention_mask is not None:
            hidden_states = hidden_states[:,:curr_frames,:]

        if not self.training and use_cache and self.right_context > 0:
            hidden_states = hidden_states[:,:-self.right_context,:]
            past_key_values.crop(past_key_values.get_seq_length() - self.right_context)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2VecSEncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    def __init__(self, config):
        super().__init__(config)
        max_source_positions = 8000
        padding_idx=1
        self.pos_conv_embed = SinusoidalPositionalEmbedding(
            config.hidden_size,
            padding_idx=padding_idx,
            init_size=max_source_positions + padding_idx + 1,
        )

        self.layers = nn.ModuleList(
            [Wav2VecSEncoderLayerStableLayerNorm(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.context_type = config.context_type
        self.main_context = config.main_context
        self.right_context = config.right_context

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            seq_len = ~(attention_mask.clone())
            paddding_mask = ~(attention_mask.clone())
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        else:
            seq_len = torch.zeros((hidden_states.size(0), hidden_states.size(1)),dtype=torch.bool).to(hidden_states.device)
            paddding_mask = None

        if self.training or past_key_values is None:
            position_embeddings = self.pos_conv_embed(seq_len)
        else:
            past_size = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_len = torch.zeros((hidden_states.size(0), past_size + hidden_states.size(1)),dtype=torch.bool).to(hidden_states.device)
            position_embeddings = self.pos_conv_embed(seq_len)[:,past_size:]

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        curr_frames = hidden_states.size(1)
        if self.training or not use_cache:
            hidden_states, attention_mask, padding_mask = gen_block_atten_mask(
                hidden_states, 
                paddding_mask, 
                main_context=self.main_context, 
                right_context=self.right_context
            )

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, 
                        past_key_value=past_key_values, 
                        attention_mask=attention_mask, 
                        output_attentions=output_attentions
                    )

                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                layer_attn = layer_outputs[1]
                if attention_mask is not None:
                    layer_attn = layer_attn[:,:curr_frames,:curr_frames]
                
                if not self.training and self.right_context > 0:
                    layer_attn = layer_attn[:,:-self.right_context,:-self.right_context]
                
                all_self_attentions = all_self_attentions + (layer_attn,)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if attention_mask is not None:
            hidden_states = hidden_states[:,:curr_frames,:]

        if not self.training and use_cache and self.right_context > 0:
            hidden_states = hidden_states[:,:-self.right_context,:]
            past_key_values.crop(past_key_values.get_seq_length() - self.right_context)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2VecSModel(Wav2Vec2Model):
    def __init__(self, config: Wav2VecSConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2VecSFeatureEncoder(config)
        if config.do_stable_layer_norm:
            self.encoder = Wav2VecSEncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2VecSEncoder(config)

    def extract_cnn_features(self, input_values):
        extract_features = self.feature_extractor(input_values)
        return extract_features.transpose(1, 2)

    def forward_encoder(
        self,
        extract_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, Wav2VecSBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if use_cache and past_key_values is not None:
            past_size = past_key_values.get_seq_length()
            extract_features = extract_features[:,past_size:,:]

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2VecSBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, Wav2VecSBaseModelOutput]:

        extract_features = self.extract_cnn_features(input_values)

        outputs = self.forward_encoder(
            extract_features=extract_features,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )

        return outputs


class Wav2VecSPreTrainedModel(Wav2Vec2PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2VecSConfig
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # Wav2VecSForPreTraining last 2 linear layers need standard Linear init.
        if isinstance(module, Wav2VecSForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        # gumbel softmax requires special init
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)


class Wav2VecSForPreTraining(Wav2VecSPreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2VecSModel(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # Initialize weights and apply final processing
        self.post_init()
