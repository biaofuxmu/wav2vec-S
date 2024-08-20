from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
from torch import Tensor
import torch.nn as nn
from fairseq import options, utils, checkpoint_utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.data import Dictionary
from fairseq.models.transformer import (
    TransformerDecoder,
)

from omegaconf import II
from rain.layers import (
    PositionalEmbedding, 
    WaitkDecoder,
    OnlineW2V2TransformerEncoder
)
from fairseq.dataclass.utils import (
    gen_parser_from_dataclass,
)
from .w2v2_transformer import (
    W2V2STTransformerModelConfig, W2V2STTransformer, base_architecture
)


@dataclass
class OnlineW2V2TransformerConfig(W2V2STTransformerModelConfig):
    max_source_positions: Optional[int] = II("task.max_audio_positions")
    max_target_positions:Optional[int] = II("task.max_text_positions")
    main_context:int = field(
        default= 16, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default= 16, metadata={"help":"right context frame"}
    )
    decoder_delay_blocks :int = field(
        default= 16, metadata={"help":"wait-k delay"}
    )
    decoder_blocks_per_token :int = field(
        default= 4, metadata={"help":"wait k token blocks"}
    )
    w2v2_model_path :str = field(
        default="/path/to/wav2vec", metadata={"help":"path to wav2vec model"}
    )
    use_linear_layer :bool = field(
        default=False, metadata={"help":"a linear layer after wav2vec2"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    online_type:ChoiceEnum(["offline", "waitk"])="offline"


@register_model("online_w2v2_transformer", dataclass = OnlineW2V2TransformerConfig)
class OnlineW2V2Transformer(W2V2STTransformer):
    @classmethod
    def build_encoder(cls, args):
        encoder = OnlineW2V2TransformerEncoder(args)
        return encoder
    
    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        print("dec attn heads:",args.decoder_attention_heads)
        if args.online_type == "offline":
            decoder = TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=False
            )
        elif args.online_type == "waitk":
            decoder = WaitkDecoder(
                args,
                tgt_dict,
                embed_tokens
            )
        else:
            raise NotImplementedError(f"unknown online type {args.online_type}")

        if decoder.embed_positions is not None and args.rand_pos_decoder > 0:
            decoder.embed_positions = PositionalEmbedding(
                decoder.max_target_positions,
                decoder.embed_dim,
                decoder.padding_idx,
                rand_max=args.rand_pos_decoder,
                learned=args.decoder_learned_pos,
            )
        return decoder
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_source=None,
        prev_target=None
    ):
        encoder_out = self.encoder(src_tokens, src_lengths)
        
        prev_tokens = prev_source if self.task_type == "asr" else prev_target
        assert prev_tokens is not None, f"no prev_tokens for {self.task_type}"
        decoder_out = self.decoder(
            prev_tokens,
            encoder_out=encoder_out,
        )
        
        return decoder_out


@register_model_architecture("online_w2v2_transformer", "online_w2v2_transformer_offline")
def offline_audio(args):
    # args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder = getattr(args, "rand_pos_decoder", 30)
    args.main_context = getattr(args,"main_context",16)
    args.right_context = getattr(args,"right_context",16)
    args.decoder_delay_blocks = getattr(args,"decoder_delay_blocks",32)
    args.decoder_blocks_per_token = getattr(args,"decoder_blocks_per_token",8)
    args.online_type = getattr(args,"online_type","offline")
    args.use_linear_layer = getattr(args,"use_linear_layer", False)
    args.freeze_finetune_updates = getattr(args,"freeze_finetune_updates", 0)
    base_architecture(args)

@register_model_architecture("online_w2v2_transformer", "online_w2v2_transformer_waitk")
def waitk_audio(args):
    args.online_type = getattr(args,"online_type","waitk")
    offline_audio(args)
