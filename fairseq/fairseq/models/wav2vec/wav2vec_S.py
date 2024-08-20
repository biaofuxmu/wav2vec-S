# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random
from fairseq import utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast,
    SinusoidalPositionalEmbedding
)
from fairseq.utils import index_put

from .utils import pad_to_multiple

from fairseq.models.wav2vec import (
    Wav2Vec2Model, 
    TransformerEncoder, 
    EXTRACTOR_MODE_CHOICES, 
    MASKING_DISTRIBUTION_CHOICES, 
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel
)

logger = logging.getLogger(__name__)


@dataclass
class Wav2VecSConfig(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )

    latent_temp: str = field(
        default="(2, 0.5, 0.999995)",
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(
        default=False, metadata={"help": "If fp16 is being used"}
    )
    # block-wise wav2vec
    context_type: str = field(
        default="constant",
        metadata={"help": "context type"},
    )
    main_context:int = field(
        default= 16, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default= 16, metadata={"help":"right context frame"}
    )
    load_pretrained_model_from: str = field(
        default="",
        metadata={"help": "model to take w2v2 weights from (for initialization"},
    )
    pos_type: str = field(
        default="sin",
        metadata={"help": "position embedding type of encoder "},
    )


@register_model("wav2vec_S", dataclass=Wav2VecSConfig)
class Wav2VecSModel(Wav2Vec2Model):
    def __init__(self, cfg: Wav2VecSConfig):
        super().__init__(cfg)

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            layer_norm_num=1 if cfg.encoder_layers==12 else 7,
        )
        logger.info(f"norm type: {cfg.extractor_mode}")
        self.build_encoder(cfg)
        self.load_pretrained_model(cfg)

    def build_encoder(self, cfg: Wav2VecSConfig):
        self.encoder = BlockwiseTransformerEncoder(cfg)


class BlockwiseTransformerEncoder(TransformerEncoder):
    def __init__(self, args: Wav2VecSConfig):
        super().__init__(args)
        logger.info(f"position embeding type: {args.pos_type}")
        self.pos_type = args.pos_type
        if self.pos_type != "conv":
            max_source_positions = 8000
            padding_idx=1
            self.pos_conv = SinusoidalPositionalEmbedding(
                self.embedding_dim,
                padding_idx=padding_idx,
                init_size=max_source_positions + padding_idx + 1,
            )
        self.required_seq_len_multiple = args.required_seq_len_multiple
        self.context_type = getattr(args, "context_type", "constant")
        self.main_context = getattr(args, "main_context", 16)
        self.right_context = getattr(args, "right_context", 8)

        logger.info(f"context_type: {self.context_type}, main context: {self.main_context}, right context: {self.right_context}")

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)
            x_len = padding_mask
        else:
            x_len = torch.zeros((x.size(0), x.size(1)),dtype=torch.bool).to(x.device)
        
        if self.pos_type == "conv":
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
        else:
            x_conv = self.pos_conv(x_len)
        
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # block mask 

        if self.context_type == "sampling":
            main_context = random.randint(4, 16) * 2
            right_context = random.randint(2, 8) * 2
            right_context = min(right_context,  main_context // 2)
        elif self.context_type == "constant":
            main_context = self.main_context
            right_context = self.right_context
        else:
            raise ValueError(
                "The mode of context_type: ({}) cannot be used. Please check.".format(
                    self.context_type
                )
            )

        curr_frames = x.shape[0]

        x, padding_mask, attn_mask = gen_block_attn_mask(
            x, padding_mask, main_context, right_context
        )

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x, 
                    self_attn_padding_mask=padding_mask, 
                    self_attn_mask=attn_mask, 
                    need_weights=False
                )
                layer_results.append(x)

        # block
        x = x[:curr_frames]
        padding_mask = padding_mask[:,:curr_frames]

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]
            layer_results = [u[:-pad_length] for u in layer_results]

        return x



def gen_block_attn_mask(
    x:Tensor,
    padding_mask:Tensor, 
    main_context:int = 1,
    right_context:int = 0
):
    """
    Args:
        x: inpout embedding, TxBxC
    """
    if padding_mask is None:
        padding_mask = x.new_zeros((x.size(1), x.size(0)), dtype=torch.bool)
    
    bsz, seq_len = padding_mask.shape 
    block_num = seq_len // main_context
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
            attn_mask = torch.cat([attn_mask1,attn_mask2], dim=1)
        
        rc_x = x.index_select(0, rc_idx)
        x = torch.cat((x, rc_x), dim= 0)
    
    attn_mask_float = x.new(attn_mask.shape).fill_(0)
    attn_mask_float = attn_mask_float.masked_fill(attn_mask.to(torch.bool), -1e4)

    return x, padding_mask, attn_mask_float

