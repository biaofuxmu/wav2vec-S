import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
import argparse
import math
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils, checkpoint_utils
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.utils import index_put
from fairseq.modules import GradMultiply, SinusoidalPositionalEmbedding
from fairseq.models import FairseqEncoder
from fairseq.models.wav2vec import Wav2Vec2Model, TransformerEncoder, TransformerSentenceEncoderLayer, ConvFeatureExtractionModel
from fairseq.incremental_decoding_utils import with_incremental_state, FairseqIncrementalState
import logging

logger = logging.getLogger(__name__)

class IncrementalDictState(FairseqIncrementalState):
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "buffer")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "buffer", buffer)

def with_incremental_dict(cls):
    cls.__bases__ = (IncrementalDictState,) + tuple(
        b for b in cls.__bases__ if b != IncrementalDictState
    )
    return cls

def gen_block_atten_mask(
    x:Tensor,
    padding_mask:Tensor, 
    main_context:int = 1,
    right_context:int = 0,
    attn_mask_value=-1e+4
):
    """
    Args:
        x: inpout embedding, TxBxC
    """
    if padding_mask is None:
        padding_mask = x.new_zeros((x.size(1), x.size(0)), dtype=torch.bool)
    
    bsz, seq_len = padding_mask.shape 
    block_num = seq_len // main_context
    block_idx = torch.arange(seq_len).to(padding_mask.device) // main_context
    # block_idx = torch.div(torch.arange(seq_len).to(padding_mask.device), main_context, rounding_mode='floor')

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
    attn_mask_float = attn_mask_float.masked_fill(attn_mask.to(torch.bool), attn_mask_value)

    return x, padding_mask, attn_mask_float


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


@with_incremental_dict
class UnidirectW2V2TransformerEncoderLayer(TransformerSentenceEncoderLayer):
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (FloatTensor): -1e8 for masked, 0 for normal, remove magic tricks from fairseq
            incremental_state: for online decoding, calculate current 
                `main_context+right_context` frames, and cache `main_context` frames
                for next inference

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e4
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        residual = x

        if incremental_state is not None:
            # catch h instead of key, value, maybe waste computation, just for simplify code
            input_state= self.self_attn._get_input_buffer(incremental_state)
            if "prev_key" in input_state and attn_mask is not None:
                prev_len = input_state["prev_key"].shape[2]
                pre_attn_mask = attn_mask.new(attn_mask.shape[0], prev_len).fill_(0)
                attn_mask= torch.cat((pre_attn_mask, attn_mask),dim=1)

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                incremental_state= incremental_state,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x

        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                incremental_state= incremental_state,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x


@with_incremental_dict
class BlockwiseW2V2TransformerEncoder(TransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        self.layers = nn.ModuleList(
            [
                UnidirectW2V2TransformerEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )
        logger.info(f"pos_type = {args.pos_type}")
        self.pos_type = args.pos_type
        if args.pos_type != "conv":
            max_source_positions = 2048
            padding_idx=1
            
            self.pos_conv = SinusoidalPositionalEmbedding(
                self.embedding_dim,
                padding_idx=padding_idx,
                init_size=max_source_positions + padding_idx + 1,
            )
        self.required_seq_len_multiple = args.required_seq_len_multiple
        self.main_context = args.main_context
        self.right_context = args.right_context
    
    def forward(self, x, padding_mask=None, incremental_state=None, finished=False, is_infer=False):
        x, padding_mask = self.extract_features(x, padding_mask, incremental_state, finished, is_infer)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x, padding_mask

    def extract_features(self, x, padding_mask=None, incremental_state=None, finished=False, is_infer=False):
        # if incremental_state is not None:
        #     return self.forward_infer(x, padding_mask, incremental_state, finished)
        
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
        curr_frames = x.shape[0]

        x, padding_mask, attn_mask = gen_block_atten_mask(
            x, padding_mask, self.main_context, self.right_context
        )

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                x = layer(
                    x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, need_weights=False
                )


        # block
        x = x[:curr_frames]
        padding_mask = padding_mask[:,:curr_frames]

        # T x B x C -> B x T x C
        # x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:-pad_length]
            padding_mask = padding_mask[:, :-pad_length]

        if is_infer and not finished and self.right_context >0:
            x = x[:-self.right_context]
            padding_mask = padding_mask[:, :-self.right_context]

        return x, padding_mask

    def forward_infer(
        self,
        fbank:torch.Tensor,
        fbk_lengths:torch.Tensor,
        incremental_state= None,
        finished=False,
        **kwargs
    ):
        
        #padding at the first block or offline
        if incremental_state is None or len(incremental_state)== 0:
            B,T,C= fbank.shape
            fbk_lengths+= self.extra_frames
            head= fbank.new(B,self.extra_frames,C).fill_(0)
            fbank= torch.cat((head,fbank),dim=1)
        # x is already TBC
        x, encoder_padding_mask = self.conv_layers(fbank, fbk_lengths, incremental_state)
        
        
        fake_tokens= encoder_padding_mask.long()
        # layernorm after garbage convs
        x= self.layernorm_embedding(x)
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state= self._get_input_buffer(incremental_state)
                full_tokens=fake_tokens
                if "prev_tokens" in input_state:
                    full_tokens= torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x= x+ pos_emb[:, -fake_tokens.shape[1]:].contiguous().transpose(0,1)
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(fake_tokens).transpose(0,1)
            
        attn_mask = None
        if self.right_context >0 and incremental_state is not None:
            # cache current input for next block
            input_state= self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim= 0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["rc_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim =1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )

        curr_frames= x.shape[0]

        x, encoder_padding_mask,attn_mask,rel_pos = gen_block_atten_mask(
            x, encoder_padding_mask, self.main_context, self.right_context
        )
       
        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state = incremental_state,rel_pos =rel_pos
            )


        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        removed_length= x.shape[0]- curr_frames
        
        x=x[:curr_frames]
        encoder_padding_mask=encoder_padding_mask[:,:curr_frames]
        if not finished and self.right_context >0:
            removed_length+= self.right_context
            x= x[:-self.right_context]
            encoder_padding_mask= encoder_padding_mask[:,:-self.right_context]
        if incremental_state is not None:
            self.rollback_steps(incremental_state, removed_length)
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    def rollback_steps(self, incremental_state, removed_length:int):
        if incremental_state is None:
            return
        if removed_length == 0:
            return
        for layer in self.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"][:,:,:-removed_length]
            input_buffer["prev_value"]= input_buffer["prev_value"][:,:,:-removed_length]
            input_buffer["prev_key_padding_mask"] = None
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)


class BlockWiseWav2Vec2Model(Wav2Vec2Model):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(BlockWiseWav2Vec2Model, BlockWiseWav2Vec2Model).add_args(parser)
        # block-wise wav2vec
        parser.add_argument(
            "--main-context",
            type=int,
            default=16,
            help="main context frame",
        )
        parser.add_argument(
            "--right-context",
            type=int,
            default=16,
            help="right context frame",
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        feature_enc_layers = eval(cfg.conv_feature_layers)
        logger.info(f"extractor_mode = {cfg.extractor_mode}")
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
            layer_norm_num=1 if cfg.encoder_layers==12 else 7,
        )
        self.encoder = BlockwiseW2V2TransformerEncoder(cfg)
        

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)

    def forward(self, source, padding_mask=None, incremental_state=None, finished=False, is_infer=False):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x = features

        x, encoder_padding_mask = self.encoder(
            x, 
            padding_mask=padding_mask, 
            incremental_state=incremental_state, 
            finished=finished,
            is_infer=is_infer
        )

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }


class OnlineW2V2TransformerEncoder(FairseqEncoder):
    def __init__(self, args,):
        super().__init__(Dictionary())

        self.main_context = args.main_context
        self.right_context = args.right_context

        wav2vec_ckpt = torch.load(args.w2v2_model_path)
        
        if wav2vec_ckpt['args'] is None:
            w2v2_args = argparse.Namespace(**wav2vec_ckpt['cfg']["model"])
        else:
            w2v2_args = wav2vec_ckpt['args']
            w2v2_args.extractor_mode = "layer_norm"
            w2v2_args.pos_type = "sin"

        w2v2_args.main_context = args.main_context
        w2v2_args.right_context = args.right_context

        self.w2v2_model = BlockWiseWav2Vec2Model.build_model(w2v2_args, task=None)
        self.w2v2_model.load_state_dict(wav2vec_ckpt['model'], strict=False)
        logger.info(f"Load w2v2 model from: {args.w2v2_model_path}")
        
        self.use_linear_layer = args.use_linear_layer
        
        self.encoder_proj = None
        if self.use_linear_layer and w2v2_args.encoder_embed_dim != args.encoder_embed_dim:
            self.encoder_proj = nn.Linear(w2v2_args.encoder_embed_dim, args.encoder_embed_dim)
            logger.info(f"use_linear_layer: {args.use_linear_layer}")

        self.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", -1)
        logger.info(f"freeze_finetune_updates: {self.freeze_finetune_updates}")
        
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @property
    def init_frames(self):
        return self.main_context + self.right_context
    
    @property
    def step_frames(self):
        return self.main_context
    
    def forward(self, src_tokens, src_lengths, incremental_state=None, finished=False, is_infer=False):
        padding_mask = lengths_to_padding_mask(src_lengths)

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.w2v2_model(src_tokens, padding_mask, incremental_state, finished, is_infer)
        
        if self.use_linear_layer and self.encoder_proj is not None:
            x = output["encoder_out"][0]
            encoder_padding_mask = output["encoder_padding_mask"][0]
            
            x = self.encoder_proj(x)
            
            return {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask],  # B x T
                "encoder_embedding": [],  # B x T x C
                "encoder_states": [],  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
                "dec1_state":[], # reserved for joint decoding
                "dec1_padding_mask":[],
            }

        return output


    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        return self.forward(
            src_tokens=net_input["src_tokens"],
            src_lengths=net_input["src_lengths"],
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
        
        if len(encoder_out["dec1_state"]) ==0:
            dec1_states=[]
        else:
            dec1_states = [(encoder_out["dec1_state"][0]).index_select(1, new_order)]
        
        if len(encoder_out["dec1_padding_mask"]) == 0:
            dec1_padding_mask= []
        else:
            dec1_padding_mask = [encoder_out["dec1_padding_mask"][0].index_select(0,new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "dec1_state":dec1_states, # TxBxC
            "dec1_padding_mask":dec1_padding_mask, # BxT
        }


def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 0)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 3, 2)] * 4"
    conv_feature_layers += " + [(512,2,2)]"
    conv_feature_layers += " + [(512,2,2)]"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", False)
    args.quantize_input = getattr(args, "quantize_input", False)
    args.same_quantizer = getattr(args, "same_quantizer", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)

    args.latent_vars = getattr(args, "latent_vars", 320)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.65)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = str(getattr(args, "latent_temp", "(2,0.5,0.999995)"))

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)

    args.main_context = getattr(args, "main_context", 8)
    args.right_context = getattr(args, "right_context", 4)
    args.required_seq_len_multiple = getattr(args, "required_seq_len_multiple", 2)
    args.simul_mode = getattr(args, "simul_mode", None)