import argparse
import logging
import os
import os.path as op
import yaml
import pandas as pd
import csv
import psutil
from tqdm import tqdm
from pathlib import Path
from examples.speech_to_text.data_utils import (
    filter_manifest_df,
    gen_config_yaml_raw,
    gen_vocab,
)

log = logging.getLogger(__name__)


def process(input_dir, output_dir, src_lang="en", tgt_lang="de", split= "train", vocab_type="bpe", vocab_size="10000"):
    os.makedirs(output_dir, exist_ok=True)
    txt_dir = op.join(input_dir, split)
    wav_dir = input_dir
    segments = []
    
    with open(op.join(txt_dir, f"{split}.tsv"),encoding="utf-8") as f:
        lines = f.readlines()
        wav_dir = lines[0].strip()
        for i, line in enumerate(lines[1:]):
            wav_name = line.split("\t")[0]
            duration = int(line.split("\t")[1].strip())
            sp, filename = wav_name.split("/")[0], wav_name.split("/")[-1].split(".")[0]
            utt_id = f"librispeech_{sp}_{filename}"
            segment = {
                "id": utt_id, 
                "n_frames": duration,
                "speaker": "spk.unk",
                "wav": op.join(wav_dir, wav_name),
            }
            segments.append(segment)

    with open(op.join(txt_dir, f"{split}.wrd"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            segments[i]["src_text"] = line.strip()
            segments[i]["src_lang"] = src_lang

            segments[i]["tgt_text"] = line.strip()
            segments[i]["tgt_lang"] = tgt_lang

    print("before load,memory cost {}".format( psutil.Process(os.getpid()).memory_info().rss))
    
    audio_info = {
        "id" : [], 
        "audio": [], 
        "n_frames": [], 
        "src_text": [],
        "src_lang": [],
        "tgt_text": [],
        "tgt_lang": [],
        "speaker": [],
        }

    test_wavs = []
    test_texts = []

    for segment in tqdm(segments):
        audio_info["id"].append(segment["id"])
        audio_info["audio"].append(segment["wav"])
        audio_info["n_frames"].append(segment["n_frames"])
        audio_info["src_text"].append(segment["src_text"])
        audio_info["src_lang"].append(segment["src_lang"])
        audio_info["tgt_text"].append(segment["tgt_text"])
        audio_info["tgt_lang"].append(segment["tgt_lang"])
        audio_info["speaker"].append(segment["speaker"])
        if "test" in split:
            test_wavs.append(segment["wav"])
            test_texts.append(segment["src_text"])

    df = pd.DataFrame.from_dict(audio_info)
    df = filter_manifest_df(df, is_train_split= split == "train", min_n_frames=1000, max_n_frames=480000)
    audio_path = op.join(output_dir, f"{split}_raw_audio.tsv")
    df.to_csv(
        audio_path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

    
    if split == "train":
        # Generate vocab
        v_size_str = "" if vocab_type == "char" else str(vocab_size)
        spm_filename_prefix = f"spm_{vocab_type}{v_size_str}"
        gen_vocab(
            Path(op.join(txt_dir, f"{split}.{src_lang}")),
            Path(op.join(output_dir, spm_filename_prefix)),
            vocab_type,
            vocab_size,
        )

        # Generate config YAM
        gen_config_yaml_raw(
            Path(output_dir),
            op.join(output_dir, spm_filename_prefix + ".model"),
            yaml_filename=f"config_wave.yaml",
            prepend_tgt_lang_tag=False,
        )

    if len(test_wavs):
        test_path = op.join(output_dir, "test")
        os.makedirs(test_path, exist_ok=True)
        with open(op.join(test_path, f"{split}.wav_list"), "w") as f:
            for wav_file in test_wavs:
                f.write(wav_file + "\n")

        with open(op.join(test_path, f"{split}.{src_lang}"), "w") as f:
            for src_text in test_texts:
                f.write(src_text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", type=str, default="en", help= "source lang")
    parser.add_argument("--tgt-lang", type=str, default="de", help= "target lang")
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("input_dir", type=str,  help= "raw data dir")
    parser.add_argument("output_dir", type=str, help = "dest dir")
    args = parser.parse_args()

    splits=["train", "valid", "test-clean", "test-other"]
    for split in splits:
        indir = args.input_dir
       
        process(indir, args.output_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang, split=split, vocab_type=args.vocab_type, vocab_size=args.vocab_size)



if __name__ == "__main__":
    main()