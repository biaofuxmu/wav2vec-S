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
    FairseqEncoderDecoderModel
)
from fairseq.data import Dictionary
from fairseq.models.transformer import (
    Embedding,
)
from omegaconf import II
from rain.layers import (
    PositionalEmbedding,
    UnidirectTransoformerEncoder,
    TransducerMHADecoder,
    OnlineW2V2TransformerEncoder
)
from . import w2v2_transformer


@dataclass
class W2V2AttentionTransducerConfig(w2v2_transformer.W2V2STTransformerModelConfig):
    train_as_ed:bool = field(
        default=False,
        metadata= {"help":"shallow coupling model trained as encoder-decoder"}
    )
    tokens_per_step:int = field(
        default=20000,
        metadata={"help":"tokens per step for output head splitting"}
    )
    delay_scale:float = field(
        default=1.0,
        metadata={"help":"scale for delay loss"}
    )
    transducer_downsample:int = field(
        default= 4,
        metadata = {"help":"source downsample ratio for transducer"}
    )
    transducer_activation: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="tanh", metadata={"help": "activation function to use"}
    )
    main_context:int = field(
        default= 16, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default= 16, metadata={"help":"right context frame"}
    )
    jointer_layers:int = field(
        default=1, metadata= {"help":"number of jointer layers"}
    )
    jointer_embed_dim:int=field(
        default=256, metadata={"help":"dim of joint embed"}
    )
    jointer_attention_heads:int=field(
        default=4, metadata= {"help":"head number for jointer"}
    )
    jointer_ffn_embed_dim:int=field(
        default=1024,metadata={"help":"dim of jointer ffn"}
    )
    jointer_type: ChoiceEnum(["simple", "mha"]) = field(
        default="mha", metadata={"help": "type for jointer"}
    )
    transducer_smoothing:float = field(
        default= 0.,  metadata = {"help":"label smoothing for transducer loss"}
    )
    delay_func:ChoiceEnum(['zero', 'diag_positive', 'diagonal']) =field(
        default= "diag_positive", metadata= {"help":"function for delay loss"}
    )
    transducer_ce_scale:float= field(
        default= 1.0, metadata= {'help':'scale for ce loss'}
    )
    transducer_label_smoothing:float= field(
        default=0.1, metadata ={'help':"label smoothing for ce loss"}
    )
    transducer_temperature:float= field(
        default=1.0, metadata={"help":"temperature for output probs"}
    )
    w2v2_model_path :str = field(
        default="/path/to/wav2vec", metadata={"help":"path to wav2vec model"}
    )
    use_linear_layer :bool = field(
        default=False, metadata={"help":"a linear layer after wav2vec2"}
    )
    freeze_w2v2_enc :int = field(
        default=12, metadata={"help":"freeze first n layers of wav2vec2 encoder"}
    )
    step_mode:str= field(
        default="constant", metadata={"help":"sampling mode of decision setp for transducer"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )

@register_model("w2v2_transducer", dataclass=W2V2AttentionTransducerConfig) 
class W2V2TransducerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        
        self.task_type= getattr(args,"task_type", "mt")
        if self.task_type not in ("asr", "st", "mt"):
            raise ValueError(f"Transducer may not work with task_type {self.task_type}")
        self.padding_idx= encoder.dictionary.pad()
        self.bos= encoder.dictionary.bos()
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        #base_architecture(args)
        #mt_transducer(args)
        if getattr(args, "max_audio_positions", None) is None:
            args.max_audio_positions = w2v2_transformer.DEFAULT_MAX_AUDIO_POSITIONS
        if getattr(args, "max_text_positions", None) is None:
            args.max_text_positions = w2v2_transformer.DEFAULT_MAX_TEXT_POSITIONS
        task_type= getattr(args,"task_type", "asr")
        if task_type not in ("asr", "st", "mt"):
            raise ValueError(f"unknown task type {task_type}")
        if task_type== "mt":
            args.max_source_positions = args.max_text_positions
        else:
            args.max_source_positions = args.max_audio_positions
        args.max_target_positions= args.max_text_positions

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        task_type= getattr(args,"task_type", "asr")

        if task_type == "mt":
            src_embed= cls.build_embedding(args, src_dict, args.encoder_embed_dim)
            if not args.share_all_embeddings:
                tgt_embed= cls.build_embedding(args, tgt_dict,args.decoder_embed_dim)
            else:
                tgt_embed= src_embed
            encoder= cls.build_encoder(args, src_dict, src_embed)
            decoder= cls.build_decoder(args, tgt_dict,tgt_embed)
        elif task_type == "asr":
            encoder= cls.build_encoder(args, None, None)
            tgt_embed= cls.build_embedding(args, src_dict, args.decoder_embed_dim)
            decoder= cls.build_decoder(args, src_dict, tgt_embed)
        elif task_type == "st":
            encoder = cls.build_encoder(args, None, None)
            tgt_embed= cls.build_embedding(args, tgt_dict, args.decoder_embed_dim)
            decoder= cls.build_decoder(args,tgt_dict, tgt_embed)
        else:
            raise NotImplementedError(f"task type {task_type} not supported")
        
        model = cls(args, encoder, decoder)
        
        if args.pretrained_encoder_path is not None:
            model.load_pretrained_encoder(args.pretrained_encoder_path)

        if args.pretrained_decoder_path is not None:
            model.load_pretrained_decoder(args.pretrained_decoder_path)

        if args.freeze_w2v2_enc:
            for name, param in model.named_parameters():
                if name.startswith("encoder.w2v2_model."):
                    if "encoder.layers." in name:
                        layer = int(name.split(".")[4])

                        if layer < args.freeze_w2v2_enc:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    else:
                        param.requires_grad = False

        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def get_targets(self, sample, net_output):
        if self.task_type == "asr":
            return sample["source"]
        else:
            return sample["target"]
    
    def get_ntokens(self, sample):
        if self.task_type == "asr":
            return sample["ntokens_src"]
        elif self.task_type=="st":
            return sample["ntokens_tgt"]
        else:
            return sample["ntokens_tgt"]

    @classmethod
    def build_encoder(cls, args, src_dict= None, embed_tokens=None):
        if args.task_type=="mt":
            assert src_dict is not None, "mt task need src_dict"
            encoder = UnidirectTransoformerEncoder(args,src_dict, embed_tokens)
        else:
            encoder = OnlineW2V2TransformerEncoder(args)
        
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        jointer_type = "simple"
        if hasattr(args,'jointer_type'):
            jointer_type = args.jointer_type
        if jointer_type == "mha":
            model = TransducerMHADecoder(args, tgt_dict, embed_tokens)
        else:
            raise ValueError(f"unknown jointer type {jointer_type}")
        embed_dim = embed_tokens.embedding_dim
        if model.lm.embed_positions is not None and args.rand_pos_decoder >0:
            model.lm.embed_positions = PositionalEmbedding(
                model.lm.max_target_positions,
                embed_dim,
                model.lm.padding_idx,
                rand_max = args.rand_pos_decoder,
                learned=args.decoder_learned_pos,
            )
        return model
    
    def load_pretrained_encoder(self, path):
        loaded_state_dict=w2v2_transformer.upgrade_state_dict_with_pretrained_weights(
            self.encoder.state_dict(), path, prefix="encoder."
        )
        self.encoder.load_state_dict(loaded_state_dict)

    def load_pretrained_decoder(self,path):
        loaded_state_dict= w2v2_transformer.upgrade_state_dict_with_pretrained_weights(
            self.decoder.state_dict(), path, prefix="decoder."
        )
        self.decoder.load_state_dict(loaded_state_dict)

    def forward_transducer(
        self,
        sample
    ):
        assert not self.args.train_as_ed, "train_as_ed should work with traditional criterion"
        src = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]

        if self.args.task_type== "asr":
            tgt = sample["source"]
        else:
            tgt = sample["target"]

        encoder_out = self.encoder(src, src_lengths)
        tgt_lengths = tgt.ne(self.padding_idx).sum(1)
        bos_head= tgt.new(tgt.size(0),1).fill_(self.bos)

        # acts should be U+1 
        prev_tokens = torch.cat((bos_head, tgt), dim= 1)
        joint_h, group_lengths = self.decoder(prev_tokens, encoder_out)
       
        return joint_h, group_lengths, tgt, tgt_lengths
    
    def forward(
        self, **net_input
    ):
        assert self.args.train_as_ed, "traditional training need train_as_ed to be True"
        src= net_input["src_tokens"]
        src_lengths= net_input["src_lengths"]

        if self.args.task_type== "mt":
            prev_output_tokens=net_input["prev_output_tokens"]
        elif self.args.task_type == "asr":
            prev_output_tokens= net_input["prev_source"]
        elif self.args.task_type== "st":
            prev_output_tokens= net_input["prev_target"]
        else:
            raise ValueError(f"bad task_type {self.args.task_type}")
       
        prev_output_tokens[:,0] = self.bos
        encoder_out= self.encoder(src, src_lengths)
        logits, _ =self.decoder(prev_output_tokens, encoder_out)
        return logits, {}
    
    def train_step(self, sample, scaler=None):
        joint_h, group_lengths, tgt,tgt_lengths= self.forward_transducer(sample)
        return self.decoder.transducer_out.train_step(
            joint_h, tgt, group_lengths, tgt_lengths, scaler=scaler
        )
    
    def eval_step(self, sample):
        # torch.cuda.empty_cache() 
        with torch.no_grad():
            joint_h, group_lengths, tgt, tgt_lengths= self.forward_transducer(sample)
            loss_info= self.decoder.transducer_out.eval_step(
                joint_h,tgt,group_lengths, tgt_lengths
            )
        return loss_info      

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)



@register_model_architecture("w2v2_transducer", "w2v2_transducer_base")
def w2v2_transducer(args):
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    args.main_context= getattr(args, "main_context",16)
    args.right_context= getattr(args, "right_context", 16)
    args.transducer_downsample = getattr(args, "transducer_downsample",16)
    args.step_mode = getattr(args, "step_mode", "constant")
    args.train_as_ed= getattr(args,"train_as_ed",False)
    args.tokens_per_step = getattr(args,"tokens_per_step", 10000)
    args.delay_scale= getattr(args,"delay_scale", 1.0)
    args.transducer_smoothing = getattr(args, "transducer_smoothing", 0.)
    args.use_linear_layer = getattr(args,"use_linear_layer", False)
    args.freeze_w2v2_enc = getattr(args,"freeze_w2v2_enc", False)
    args.freeze_finetune_updates = getattr(args,"freeze_finetune_updates", 0)
    w2v2_transformer.base_architecture(args)


@register_model_architecture("w2v2_transducer", "w2v2_caat")
def w2v2_caat(args):
    args.jointer_type=getattr(args, "jointer_type","mha")
    args.jointer_layers=getattr(args, "jointer_layers",6)
    args.jointer_embed_dim= getattr(args, "jointer_embed_dim", 256)
    args.jointer_attention_heads= getattr(args, "jointer_attention_heads", 4)
    args.jointer_ffn_embed_dim= getattr(args, "jointer_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim= getattr(args, "decoder_ffn_embed_dim", 1024)
    w2v2_transducer(args)

@register_model_architecture("w2v2_transducer", "w2v2_caat_offline")
def w2v2_caat_offline(args):
    args.train_as_ed = True
    args.transducer_downsample = -1
    w2v2_caat(args)

@register_model_architecture("transducer", "w2v2_caat_2048")
def w2v2_caat_2048(args):
    args.jointer_ffn_embed_dim= getattr(args, "jointer_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim= getattr(args, "decoder_ffn_embed_dim", 2048)
    w2v2_caat(args)
    
@register_model_architecture("transducer", "w2v2_caat_2048_offline")
def w2v2_caat_2048_offline(args):
    args.train_as_ed = True
    args.transducer_downsample = -1
    w2v2_caat_2048(args)