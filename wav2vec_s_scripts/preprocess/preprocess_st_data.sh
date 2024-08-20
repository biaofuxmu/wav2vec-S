MUSTC_ROOT=/path/to/data
lang=de

export PYTHONPATH=/path/to/wav2vec-S/fairseq

python fairseq/examples/speech_to_text/prep_mustc_data_raw.py \
    --data-root ${MUSTC_ROOT} \
    --task st \
    --tgt-lang ${lang} \
    --vocab-type unigram --vocab-size 10000

python fairseq/examples/speech_to_text/seg_mustc_data.py \
    --data-root ${MUSTC_ROOT}/ \
    --task st \
    --lang ${lang} \
    --output ${MUSTC_ROOT}/en-${lang}/data/simul_test \
    --split tst-COMMON