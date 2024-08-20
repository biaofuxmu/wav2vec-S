
DATA_PATH=/path/to/data
PRETRAINED_MODEL=/path/to/wav2vec_vox_new.pt

fairseq-hydra-train \
    task.data=${DATA_PATH} \
    model.load_pretrained_model_from=${PRETRAINED_MODEL} \
    --config-dir /path/to/wav2vec-S/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec-S_large_librivox