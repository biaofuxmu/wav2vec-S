
DATA_ROOT=/path/to/librispeech/data-bin

PRETRAIN_WAV2VEC=/path/to/wav2vec-S-lv60k-large.pt

SAVE_DIR=/path/to/model_output_path

echo ${SAVE_DIR}

export PYTHONPATH=/path/to/wav2vec-S/fairseq

fairseq-train ${DATA_ROOT} \
    --config-yaml config_wave.yaml \
    --train-subset train_raw_audio \
    --valid-subset valid_raw_audio \
    --max-audio-positions 3200000 \
    --transducer-downsample 64 --step-mode "random" \
    --main-context 16 --right-context 8 \
    --user-dir rain \
    --max-update 100000 \
    --delay-func diag_positive \
    --delay-scale 1.0 \
    --w2v2-model-path ${PRETRAIN_WAV2VEC} \
    --transducer-label-smoothing 0.1 --transducer-ce-scale 1. \
    --task w2v2_transducer --task-type asr --bpe-dropout 0.1 \
    --arch w2v2_caat \
    --tokens-per-step 6000 \
    --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 \
    --lr-scheduler tri_stage --phase-ratio '[0.1, 0.0, 0.9]' \
    --final-lr-scale 0.05 \
    --criterion fake_loss \
    --clip-norm 25.0 \
    --save-dir ${SAVE_DIR} \
    --max-tokens 1440000 \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --decoder-attention-heads 16 --decoder-ffn-embed-dim 4096 \
    --decoder-layers 12 --jointer-layers 12 --jointer-embed-dim 1024 \
    --jointer-attention-heads 16 --jointer-ffn-embed-dim 4096 \
    --update-freq 2 --skip-invalid-size-inputs-valid-test \
    --log-interval 100 --save-interval 1 --log-format simple  --num-workers 24 \
    --fp16 --min-loss-scale 1e-6 --ddp-backend=no_c10d \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --use-linear-layer --freeze-w2v2-enc 0 \
    --best-checkpoint-metric loss --no-epoch-checkpoints \
    --keep-best-checkpoints 10 --freeze-finetune-updates 10000 \

