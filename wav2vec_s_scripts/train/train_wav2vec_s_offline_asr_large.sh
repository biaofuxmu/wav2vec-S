
LANG=de #es

MUSTC_ROOT=/path/to/dataset/en-${LANG}

PRETRAIN_WAV2VEC=/path/to/wav2vec-S-lv60k-large.pt

SAVE_DIR=/path/to/model_output_path

export PYTHONPATH=/path/to/wav2vec-S/fairseq

fairseq-train ${MUSTC_ROOT} \
    --config-yaml config_wave.yaml \
    --train-subset train_raw_joint_st \
    --valid-subset dev_raw_joint_st \
    --max-audio-positions 3200000 \
    --main-context 16 --right-context 8 \
    --user-dir rain \
    --max-epoch 80 \
    --ddp-backend no_c10d \
    --task w2v2_s2s --task-type asr \
    --arch online_w2v2_transformer_offline \
    --dropout 0.3 --activation-dropout 0.1 \
    --share-decoder-input-output-embed --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0002 --lr-scheduler inverse_sqrt  \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay 0.0001 --seed 1 \
    --save-dir ${SAVE_DIR} \
    --use-linear-layer \
    --decoder-layers 12 \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --decoder-attention-heads 16 --decoder-ffn-embed-dim 4096 \
    --max-tokens 4800000 --update-freq 2 \
    --log-interval 100 --save-interval 1 --log-format simple --fp16 \
    --w2v2-model-path ${PRETRAIN_WAV2VEC} \
    --tensorboard-logdir ${SAVE_DIR}/tensorboard_logs \
    --report-accuracy \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 10 \
    --patience 10 \
