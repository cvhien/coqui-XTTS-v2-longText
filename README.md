# coqui-XTTS-v2-longText

This python script to execute the model in case of a long text. 

Aim to read English articles with a clear voice, high accuracy and good performance.

My coding is still in newbie.

### Features

- Supports long text input from a file. 

There are limitations with high CPU consumption and slow speed,

with sentence preprocess (missing end sentence quotations + incorrect split -> incorrect tones), model usage.

There are may warnings about:

"Limitation "XTTS can only generate text with a maximum of 400 tokens."?

[!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio."

### Reference 

https://github.com/coqui-ai/TTS/tree/dev#installation

## Model

https://huggingface.co/coqui/XTTS-v2/tree/main

### Demo

speech-long.wav is for input-long.txt which is gotten from: https://theconversation.com/the-long-feared-middle-east-war-is-here-this-is-how-israel-could-now-hit-back-at-iran-240432

speech-long.wav is for input-short.txt which is gotten some lines from: https://www.abc.net.au/news/2024-10-04/daylight-saving-2024-pocket-guide/104384164

### Install

Install TTS with guidance from the original website, 

Download huggingface.co/coqui/XTTS-v2 model,

Test is ok with python 3.11.

### Usage

Execute: python my-coquilXTTS-v2.py

model_path="~/workspace/models/coqui/XTTS-v2" # Local model path

speaker_wav=model_path + "/samples/samples_en_sample.wav" #default

num_sentences = 2   # Adjust this value according to your needs, not ok for big number results a too long text

input="./input-short.txt" # text input file which is need to convert

outputFile="./speech.wav" # audio speech, result file

### Code
Using the model directly:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/data/TTS-public/_refclips/3.wav",
    gpt_cond_len=3,
    language="en",
)
```

### License

### Contact
