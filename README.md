# wav2vec-S: Adapting Pre-trained Speech Models for Streaming

Source code for ACL 2024 findings paper: [Adapting Offline Speech Translation Models for Streaming with Future-Aware Distillation and Inference](https://aclanthology.org/2024.findings-acl.681/)

## [wav2vec-S](#wav2vec-s-adapting-pre-trained-speech-models-for-streaming)
- [Requirements and Installation](#requirements-and-installation)
- [Streaming Pre-training](#streaming-pre-training)
- [Fine-tuning on streaming ST task](#fine-tuning-on-streaming-st-task)
- [Fine-tuning on streaming ASR task](#fine-tuning-on-streaming-asr-task)
- [Checkpoints](#checkpoints)

## Requirements and Installation
- PyTorch version >= 1.10.0
- Python version >= 3.8
- To install fairseq and develop locally:
```shell script
cd fairseq
pip install -e ./
```
- To install simuleval for streaming inference:
```shell script
cd simuleval
pip install -e ./
```
- To install warprnnt-pytorch for computing RNN-T loss:
```shell script
export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

cd warp_transducer
mkdir build && cd build
cmake ..
make install
cd ../pytorch_binding
pip install -e .
```
## Streaming Pre-training

Given a directory containing wav files to be used for pre-training (we recommend splitting each file into separate file 10 to 30 seconds in length). Our streaming pre-training is based on [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), and the datasets and pre-trained models can be downloaded [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec).

### Prepare training data manifest

First, install the `soundfile` library:

```shell script
pip install soundfile
```

Next, run:

```shell script
python examples/wav2vec/wav2vec_manifest.py /path/to/waves \
    --dest /manifest/path --ext $ext --valid-percent $valid
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

### Train a wav2vec-S base model

This configuration was used for the base model trained on the Librispeech dataset in the wav2vec-S paper. 

Note that the input is expected to be single channel, sampled at 16 kHz

```shell script
fairseq-hydra-train \
    task.data=/path/to/data \
    model.load_pretrained_model_from=/path/to/wav2vec_small.pt \
    --config-dir /path/to/wav2vec-S/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec-S_base_librispeech
```

Note: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 64/k


### Train a wav2vec-S large model

This configuration was used for the large model trained on the Libri-light dataset in the wav2vec-S paper.

```shell script
fairseq-hydra-train \
    task.data=/path/to/data \
    model.load_pretrained_model_from=/path/to/wav2vec_vox_new.pt \
    --config-dir /path/to/wav2vec-S/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec-S_large_librivox
```

Note: you can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

## Fine-tuning on streaming ST task


### Data Processing
Take German for example.
Firstly, download [MuST-C v1.0](https://ict.fbk.eu/must-c/) archive `MUSTC_v1.0_en-de.tar.gz` to the `${MUSTC_ROOT}` path, and uncompress it:
```shell script
LANG=de
MUSTC_ROOT=/path/data/en-${LANG}$
tar -xzvf MUSTC_v1.0_en-de.tar.gz
```
Then, run the script to prepare data manifest.
```shell script
sh wav2vec_s_scripts/preprocess/preprocess_st_data.sh
```
### Fine-tuning
Train an offline text-to-text translation model following [CAAT](https://github.com/danliu2/caat).
Then, generate sequence distillation training data:
```shell script
sh wav2vec_s_scripts/preprocess/gen_distillation_data.sh
```

Pretrain encoder with speech recognition task
```shell script
sh wav2vec_s_scripts/train/train_wav2vec_s_offline_asr_base.sh
```

Train final streaming st base model
```shell script
sh wav2vec_s_scripts/train/train_wav2vec_s_caat_simulst_base.sh
```

### Evaluation
Evaluation on the streaming ST task
```shell script
bash wav2vec_s_scripts/eval/eval_wav2vec_s_caat_st.sh
```

## Fine-tuning on streaming ASR task
### Data Processing
Firstly, download [LibriSpeech](https://www.openslr.org/12) and organize the data into the following formats:
```
train
├── train.wrd
└── train.tsv
valid/
├── valid.wrd
└── valid.tsv
test-clean/
├── test-clean.wrd
└── test-clean.tsv
test-other/
├── test-other.wrd
└── test-other.tsv
```
`valid` includes the data from the `dev-clean` and `dev-other` subsets.

Here's some examples from the `.tsv` file.
```
/path/to/LibriSpeech
train-clean-100/103/1240/103-1240-0000.flac	225360
train-clean-100/103/1240/103-1240-0001.flac	255120
...
train-other-500/985/126228/985-126228-0050.flac	134320
train-other-500/985/126228/985-126228-0051.flac	269440
```
Here's some examples from the `.wrd` file.
```
CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM
...
PERHAPS SUE WAS THUS VENTURESOME WITH MEN BECAUSE SHE WAS CHILDISHLY IGNORANT OF THAT SIDE OF THEIR NATURES WHICH WORE OUT WOMEN'S HEARTS AND LIVES
JUDE AND THE LANDLADY OFFERED TO GET IT NO SHE SAID RUNNING BACK IT IS MY HANDKERCHIEF I KNOW WHERE I LEFT IT JUDE FOLLOWED HER BACK SHE HAD FOUND IT AND CAME HOLDING IT IN HER HAND SHE LOOKED INTO HIS EYES WITH HER OWN TEARFUL ONES AND HER LIPS SUDDENLY PARTED AS IF SHE WERE GOING TO AVOW SOMETHING
```

Then, run the script to prepare data manifest.
```shell script
sh wav2vec_s_scripts/preprocess/preprocess_librispeech_data.sh
```
### Fine-tuning
Train streaming asr base model
```shell script
sh wav2vec_s_scripts/train/train_wav2vec_s_caat_simulasr_base.sh
```

### Evaluation
Evaluation on the streaming ASR task
```shell script
bash wav2vec_s_scripts/eval/eval_wav2vec_s_caat_asr.sh
```

## Checkpoints

Ours models are released here:

<table>
	<tr>
	    <td >Model</td>
	    <td>Stage</td>
	    <td>Dataset</td>
	    <td>Dictionary</td>
	</tr >
	<tr >
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-base.pt">wav2vec-S Base</a></td>
	    <td rowspan="2">pre-training</td>
	    <td>LibriSpeech 960h</td>
	    <td></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-lv60k-large.pt">wav2vec-S Large</a></td>
	    <td>Libri-Light 60kh</td>
	    <td></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulst-en-de-base.pt">wav2vec-S-CAAT Streaming ST En-De Base</a></td>
	    <td rowspan="6">fine-tuning</td>
	    <td rowspan="2">MuST-C En-De</td>
	    <td rowspan="2"><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-de/config_raw_joint_st.yaml">config.yaml</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-de/spm_unigram10000_st.model">sentencepiece.model</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-de/spm_unigram10000_st.vocab">sentencepiece.vocab</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-de/spm_unigram10000_st.txt">sentencepiece.txt</a></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulst-en-de-large.pt">wav2vec-S-CAAT Streaming ST En-De Large</a></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulst-en-es-base.pt">wav2vec-S-CAAT Streaming ST En-Es Base</a></td>
	    <td rowspan="2">MuST-C En-Es</td>
	    <td rowspan="2"><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-es/config_raw_joint_st.yaml">config.yaml</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-es/spm_unigram10000_st.model">sentencepiece.model</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-es/spm_unigram10000_st.vocab">sentencepiece.vocab</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-es/spm_unigram10000_st.txt">sentencepiece.txt</a></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulst-en-es-large.pt">wav2vec-S-CAAT Streaming ST En-Es Large</a></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulasr-en-librispeech-base.pt">wav2vec-S-CAAT Streaming ASR Base</a></td>
	    <td rowspan="2">LibriSpeech 960h</td>
	    <td rowspan="2"><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-librispeech/config_raw_asr.yaml">config.yaml</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-librispeech/spm_unigram10000.model">sentencepiece.model</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-librispeech/spm_unigram10000.vocab">sentencepiece.vocab</a><br><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/vocab-en-librispeech/spm_unigram10000.txt">sentencepiece.txt</a></td>
	</tr>
	<tr>
	    <td><a href="http://nlp.xmu.edu.cn/biaofu/models/wav2vec-S/wav2vec-S-caat-simulasr-en-librispeech-large.pt">wav2vec-S-CAAT Streaming ASR Large</a></td>
	</tr>
</table>



## Citation

If the paper or the code helps you, please cite the paper in the following format :
```
@inproceedings{fu-etal-2024-wav2vec,
    title = "wav2vec-{S}: Adapting Pre-trained Speech Models for Streaming",
    author = "Fu, Biao and Fan, Kai and Liao, Minpeng and Chen, Yidong and Shi, Xiaodong and Huang, Zhongqiang",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.681",
    pages = "11465--11480",
}
```