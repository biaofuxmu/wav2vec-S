#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_config_yaml_raw,
    get_zip_manifest,
    save_df_to_tsv,
    cal_gcmvn_stats,
    gen_vocab,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform


log = logging.getLogger(__name__)

class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "ja", "zh"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        # print(_root, wav_root, txt_root)
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            filename = txt_root / f"{split}.{_lang}"
            with open(filename) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def process(args, MANIFEST_COLUMNS):
    root = Path(args.data_root).absolute()
    lang = args.tgt_lang
    cur_root = root / f"en-{lang}"
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")
    train_text = []
    # Extract features
    for split in MUSTC.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = MUSTC(args.data_root, lang, split)
        for wav, offset, n_frames, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(f"{wav}:{offset}:{n_frames}")
            manifest["n_frames"].append(n_frames)
            manifest["src_text"].append(src_utt)
            manifest["src_lang"].append("en")
            manifest["tgt_text"].append(tgt_utt)
            manifest["tgt_lang"].append(lang)
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["src_text"])
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=1000, max_n_frames=480000)
        save_df_to_tsv(df, cur_root / f"{split}_raw_joint_st.tsv")
    
    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )

    # Generate config YAM
    gen_config_yaml_raw(
        cur_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_raw_joint_st.yaml",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help=(
                            "Maximum number of sentences to use to estimate"
                            "global mean and variance"
                        ))
    parser.add_argument("--tgt-lang", help="target language")
    args = parser.parse_args()

    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "src_lang", "tgt_text", "tgt_lang", "speaker"]
    process(args, MANIFEST_COLUMNS)



if __name__ == "__main__":
    main()
