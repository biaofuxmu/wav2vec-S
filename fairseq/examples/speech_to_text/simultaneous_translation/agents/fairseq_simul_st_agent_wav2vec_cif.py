import json
from logging import raiseExceptions
import math
import os
import sys

import numpy as np
import torch
import yaml

from fairseq import checkpoint_utils, tasks
from fairseq.file_io import PathManager

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

from examples.simultaneous_translation.models.convtransformer_simul_trans import *


SHIFT_SIZE = 1
WINDOW_SIZE = 1
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"

THRESHOLD = 1


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
                                         num_frames * self.num_samples_per_shift:
                                         ]

        torch.manual_seed(1)
        output = input_samples
        output = np.asarray(output)
        output = output/32768.0
        output = output.astype("float32")

        return torch.from_numpy(output)


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):
        if len(self.value) == 0:
            self.value = value
            return

        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class FairseqSimulSTAgent(SpeechAgent):
    speech_segment_size = 25  # in ms, 4 pooling ratio * 10 ms step size
    # speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args
        self.lang = args.lang
        self.load_model_vocab(args)

        if getattr(
                self.model.decoder.layers[0].encoder_attn,
                'pre_decision_ratio',
                None
        ) is not None:
            self.speech_segment_size *= (
                self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
            )
        print("speech_segment_size: ", self.speech_segment_size)
        args.global_cmvn = None
        if args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        if args.global_stats:
            with PathManager.open(args.global_stats, "r") as f:
                global_cmvn = json.loads(f.read())
                self.global_cmvn = {"mean": global_cmvn["mean"], "std": global_cmvn["stddev"]}

        self.feature_extractor = OnlineFeatureExtractor(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish
        print("force_finish : ", self.force_finish)

        torch.set_grad_enabled(False)

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument('--w2v2-model-path', type=str, required=True,
                            help='path to pretrained wav2vec2 model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--waitk-lagging", type=int, default=None,
                            help="waitk lagging")
        parser.add_argument("--fixed-pre-decision-ratio", type=int, default=None,
                            help="fixed pre decision ratio")
        parser.add_argument("--lang", type=str, default=None,
                    help="target language")
        parser.add_argument("--fixed-pre-decision", type=bool, default=False,
                            help="whether using fixed-pre-decision")
        parser.add_argument("--adaptive-decision", type=bool, default=True,
                            help="whether using adaptive-decision")
        parser.add_argument("--threshold", type=float, default=1, help="cif threshold")

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not PathManager.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        if args.waitk_lagging is not None:
            state["cfg"]["model"].__dict__["waitk_lagging"] = args.waitk_lagging
            print("new waitk_lagging: ", args.waitk_lagging)
        if args.fixed_pre_decision_ratio is not None:
            state["cfg"]["model"].__dict__["fixed_pre_decision_ratio"] = args.fixed_pre_decision_ratio
            print("new fixed_pre_decision_ratio: ", args.fixed_pre_decision_ratio)
        if args.w2v2_model_path is not None:
            state["cfg"]["model"].__dict__["w2v2_model_path"] = args.w2v2_model_path
            print("new w2v2_model_path: ", args.w2v2_model_path)
        # sys.stderr.write("new waitk_lagging: " + str(state["cfg"]["model"].__dict__["waitk_lagging"]) + "\n")
        # sys.stderr.write(
            # "new fixed_pre_decision_ratio: " + str(state["cfg"]["model"].__dict__["fixed_pre_decision_ratio"]) + "\n")
        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config is not None:
            task_args.config_yaml = args.config

        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=False)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        self.prev_src = 0
        self.fires = 0

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # Merge sub word to full word.
        if self.model.decoder.dictionary.eos() == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.model.decoder.dictionary.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        if (
                len(units) > 0
                and self.model.decoder.dictionary.eos() == units[-1]
                or len(states.units.target) > self.max_len
        ):
            tokens = [self.model.decoder.dictionary.string([unit]) for unit in units]
            return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

        return None

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )
        # print("src_indices", src_indices.shape)
        # print("src_lengths", src_lengths.shape)
        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION
        if self.lang is None:
            tgt_indices = self.to_device(
                torch.LongTensor(
                    [self.model.decoder.dictionary.eos()]
                    + [x for x in states.units.target.value if x is not None]
                ).unsqueeze(0)
            )
        else:
            lang_tok="<lang:%s>"%self.lang
            tgt_indices = self.to_device(
                torch.LongTensor(
                    [self.model.decoder.dictionary.eos()] + [self.model.decoder.dictionary.index(lang_tok)]
                    + [x for x in states.units.target.value if x is not None]
                ).unsqueeze(0)
            )            
        # states.incremental_states = {}
        if "steps" in states.incremental_states and states.incremental_states["steps"]["src"] > self.prev_src:
            self.prev_src = states.incremental_states["steps"]["src"]

        states.incremental_states["steps"] = {
            "src": states.encoder_states["encoder_out"][0].size(0),
            "tgt": 1 + len(states.units.target),
        }

        states.incremental_states["online"] = {"only": torch.tensor(not states.finish_read())}

        # print('states.encoder_states["alphas"]', states.encoder_states["alphas"].sum().numpy())
        # print('states.encoder_states["encoder_out"][0].size(0)', states.encoder_states["encoder_out"][0].size(0))
        # print('prev_src', self.prev_src)
        # print('states.finish_read()', states.finish_read())
        if states.encoder_states["alphas"].sum().cpu().numpy() < 1 and not states.finish_read():
            return READ_ACTION
        if states.incremental_states["steps"]["src"] < self.prev_src:   # 处理cif缩短后长度反而比tgt_indices短的问题
            # print('states.units.target', states.units.target)
            if not states.finish_read():
                return READ_ACTION
            elif len(states.units.target) < 2: # 只有 <bos> tag
                return WRITE_ACTION
            elif states.units.target[-2] != states.units.target[-1]: # 避免生成重复的情况
                return WRITE_ACTION
            else:
                return None

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            incremental_state=states.incremental_states,
        )
        # x, outputs = self.model.decoder.forward(
        #     prev_output_tokens=tgt_indices,
        #     encoder_out=states.encoder_states,
        #     incremental_state=None,
        # )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        # print('alphas:', states.encoder_states["alphas"][0][MinusOne])
        # states.integrate += states.encoder_states["alphas"][0][MinusOne]
        # if states.integrate >= THRESHOLD:
        #     states.fires += 1
        #     states.integrate = states.integrate - THRESHOLD
        if states.finish_read():
            return WRITE_ACTION
        if self.args.adaptive_decision:
            # print('len(states.units.source.value)', len(states.units.source.value))
            # if (len(states.units.source.value) / (self.speech_segment_size * 16) - len(states.units.target.value)) < self.args.waitk_lagging:
            #     return READ_ACTION
            if states.incremental_states["steps"]["src"] - len(states.units.target.value) < self.args.waitk_lagging:
                return READ_ACTION
            # if torch.round(states.encoder_states['alphas'].sum(-1)/self.args.threshold).int() - len(states.units.target.value) < self.args.waitk_lagging and not states.finish_read():
            #     return READ_ACTION
            else:
                return WRITE_ACTION
        else:
            raise NotImplementedError

        # if outputs.action == 0:
        #     return READ_ACTION
        # else:
        #     return WRITE_ACTION

    def predict(self, states):
        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        index = index[0, 0].item()

        if (
                self.force_finish
                and index == self.model.decoder.dictionary.eos()
                and not states.finish_read()
        ):
            # If we want to force finish the translation
            # (don't stop before finish reading), return a None
            # self.model.decoder.clear_cache(states.incremental_states)
            index = None
        
        if len(states.units.target) >= self.args.max_len:
            return self.model.decoder.dictionary.eos()
        return index