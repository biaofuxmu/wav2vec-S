

INPUT_DIR=/path/to/librispeech
OUTPUT_DIR=/path/to/librispeech/data-bin

echo "processing audio in ${INPUT_DIR}"

python wav2vec_s_scripts/preprocess/process_librispeech_raw_data.py \
    --src-lang en --vocab-type unigram --vocab-size 10000 ${INPUT_DIR} $OUTPUT_DIR
