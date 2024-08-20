#!/bin/bash

DECISION_STEP=4 # 2 4 10 20

LANG=de
MUSTC_ROOT=/path/to/data/en-${LANG}

SOURCE_FILE=/path/to/tst-COMMON.wav_list
TARGET_FEIL=/path/to/tst-COMMON.${LANG}

W2V2_MODEL_PATH=/path/to/wav2vec-S-base.pt
MODEL_FILE=/path/to/model_output_path/wav2vec-S-caat-simulst-en-de-base.pt

RESULT_FILE=/path/to/model_output_path/results

if [ ! -f ${MODEL_FILE} ];then
    python fairseq/scripts/average_checkpoints.py \
    --inputs /path/to/model_output_path --output ${MODEL_FILE} \
    --num-epoch-checkpoints 10
fi

export PYTHONPATH=/path/to/wav2vec-S/fairseq

simuleval --agent ./rain/simul/speech_fullytransducer_agent.py \
    --task-type st --data-bin ${MUSTC_ROOT} \
    --user-dir /path/to/wav2vec-S/fairseq \
    --source ${SOURCE_FILE} \
    --target ${TARGET_FEIL} \
    --data-type speech --model-path ${MODEL_FILE} \
    --w2v2-model-path ${W2V2_MODEL_PATH} \
    --output ${RESULT_FILE} --port 12345 --timeout 100 \
    --max-len-a 0.048 --len-scale 0.7 --len-penalty 0 --max-len-b -5 \
    --intra-beam 5 --inter-beam 1 --decoder-step-read 256 --eager \
    --step-read-block ${DECISION_STEP}