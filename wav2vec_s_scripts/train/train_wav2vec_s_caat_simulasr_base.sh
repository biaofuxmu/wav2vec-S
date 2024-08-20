
DATA_ROOT=/path/to/librispeech/data-bin

PRETRAIN_WAV2VEC=/path/to/wav2vec-S-base.pt

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
    --max-update 200000 \
    --delay-func diag_positive \
    --delay-scale 1.0 \
    --w2v2-model-path ${PRETRAIN_WAV2VEC} \
    --transducer-label-smoothing 0.1 --transducer-ce-scale 1. \
    --task w2v2_transducer --task-type asr --bpe-dropout 0.1 \
    --arch w2v2_caat \
    --tokens-per-step 6000 \
    --dropout 0.1 --activation-dropout 0.1 --attention-dropout 0.1 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 \
    --lr-scheduler tri_stage --phase-ratio '[0.1, 0.3, 0.6]' \
    --final-lr-scale 0.05 \
    --criterion fake_loss \
    --clip-norm 25.0 \
    --save-dir ${SAVE_DIR} \
    --max-tokens 3200000 \
    --encoder-embed-dim 768 --decoder-embed-dim 768 \
    --decoder-attention-heads 12 --decoder-ffn-embed-dim 3072 \
    --jointer-layers 6 --jointer-embed-dim 768 \
    --jointer-attention-heads 12 --jointer-ffn-embed-dim 3072 \
    --update-freq 1 --skip-invalid-size-inputs-valid-test \
    --log-interval 100 --save-interval 1 --log-format simple  --num-workers 24 \
    --fp16 --min-loss-scale 1e-6 --ddp-backend=no_c10d \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --use-linear-layer --freeze-w2v2-enc 0 \
    --best-checkpoint-metric loss --no-epoch-checkpoints \
    --keep-best-checkpoints 10 --freeze-finetune-updates 10000 \

