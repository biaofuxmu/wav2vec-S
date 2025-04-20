## Use wav2vec-S with 🤗Transformers

Wav2Vec-S is also available with the [🤗Transformers library](https://github.com/huggingface/transformers)

### Pretrained Models

| Huggingface | ModelScope  |
| ---- | ---- |
|[wav2vec-S Base 🤗](https://huggingface.co/biaofu-xmu/wav2vec-S-Base) | [wav2vec-S Base 🤖](https://www.modelscope.cn/models/BiaoFuXMU/wav2vec-S-Base) |
|[wav2vec-S Large 🤗](https://huggingface.co/biaofu-xmu/wav2vec-S-Large) | [wav2vec-S Large 🤖](https://www.modelscope.cn/models/BiaoFuXMU/wav2vec-S-Large) |
|[wav2vec-S-Base-ft-960h 🤗](https://huggingface.co/biaofu-xmu/wav2vec-S-Base-ft-960h) | [wav2vec-S-Base-ft-960h 🤖](https://www.modelscope.cn/models/BiaoFuXMU/wav2vec-S-Base-ft-960h) |
|[wav2vec-S-Large-ft-960h 🤗](https://huggingface.co/biaofu-xmu/wav2vec-S-Large-ft-960h) | [wav2vec-S-Large-ft-960h 🤖](https://www.modelscope.cn/models/BiaoFuXMU/wav2vec-S-Large-ft-960h) |

### Usage example

```python
# !pip install transformers
# !pip install soundfile
from transformers import Wav2Vec2FeatureExtractor, DynamicCache
import torch
from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel
import soundfile as sf

model_path = "biaofu-xmu/wav2vec-S-Base"

feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2VecSModel.from_pretrained(model_path)
model.to(torch.bfloat16).cuda()
model.eval()

wav_file = './example.flac'
waveform, sampling_rate = sf.read(wav_file)

inputs = feat_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").to('cuda').to(torch.bfloat16)

# streaming inference
main_context = model.config.main_context
right_context = model.config.right_context

# The number of blocks to wait for each inference which can control the delay
wait_block = 1

init_frames = main_context * wait_block + right_context

step_frames = main_context * wait_block

block_size = init_frames

speech_segment_size = 320

# first speech block 
segment_size = speech_segment_size * (init_frames + 1)

print(len(waveform))

processed_frames = 0
past_key_values = DynamicCache()

finish_read = False
hidden_state = None

while segment_size <= len(waveform):
    if segment_size >= len(waveform):
        finish_read = True

    if finish_read:
        model.encoder.right_context = 0
    else:
        model.encoder.right_context = right_context

    segment = waveform[:segment_size]
    block_inputs = feat_extractor(segment, sampling_rate=sampling_rate, return_tensors="pt").to('cuda').to(torch.bfloat16)
    cnn_features = model.extract_cnn_features(block_inputs.input_values)
    current_frame = cnn_features.size(1)

    # The model only outputs the hidden states corresponding to the main context. 
    # The hidden states of the right context will be discarded, 
    # but since it overlaps with the next block, its hidden states will be calculated in the next block.
    block_outputs = model.forward_encoder(
        cnn_features[:,:processed_frames+block_size], 
        past_key_values=past_key_values, 
        use_cache=True
    )

    if hidden_state is None:
        hidden_state = block_outputs.last_hidden_state
    else:
        hidden_state = torch.cat((hidden_state, block_outputs.last_hidden_state), dim=1)
    
    past_key_values = block_outputs.past_key_values
    processed_frames = past_key_values.get_seq_length()

    if finish_read:
        break

    # next speech block
    segment_size = min(speech_segment_size * (current_frame + step_frames + 1), len(waveform))
```

