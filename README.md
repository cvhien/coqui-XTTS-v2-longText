# coqui-XTTS-v2-longText

Python script to convert long text file to audio wav file, tts, with voice cloning.

Aim to read English articles with a clear voice, high accuracy and good performance.

### Description

Supports long text input from a file, uses coqui-ai/TTS coqui/XTTS-v2 model.
There are limitations with high CPU consumption and quite slow speed.

It splits the text into smaller chunks, creates wav file.
There may have issues with not yet optimized model usage, spliting text or truncate audio with too long sentence or chunk size,
current setting is maxChar = 250.

There are may warnings about:
```
"Limitation "XTTS can only generate text with a maximum of 400 tokens."?
[!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio."
```

### Reference 

https://github.com/coqui-ai/TTS/tree/dev#installation

https://github.com/dynamiccreator/voice-text-reader.git voice-text-reader

## Model

https://huggingface.co/coqui/XTTS-v2/tree/main

### Demo

speech-long.wav is for input-long.txt which is gotten from: https://theconversation.com/the-long-feared-middle-east-war-is-here-this-is-how-israel-could-now-hit-back-at-iran-240432

speech-long.wav is for input-short.txt which is gotten some lines from: https://www.abc.net.au/news/2024-10-04/daylight-saving-2024-pocket-guide/104384164

### Install

Install TTS with guidance from the original website, plus some modules:
```
pip install -r requirements.txt
```

Download huggingface.co/coqui/XTTS-v2 model,

Run OK with python 3.11.

### Usage

Prepared your desired voice on a .wav file, or it will be the default "samples/en_sample.wav" in the model huggingface 
download folder. I make voice file with similar parameters of file in samples folder.

```
python my-coquilXTTS-v2.py
```
```
python my-coquilXTTS-v2.py -m /Volumes/OTHER/models/coqui/XTTS-v2 -t input-short.txt -sp ~/Oprah-Z3AtmPS1Wic.wav
```

```
All options:
  -h, --help            show this help message and exit
  -t TEXT, --text TEXT  The path of the text file to be read.
  -m MODEL, --model MODEL
                        The model path used for speak generation.
  -sp SPEAKER_FILE, --speaker_file SPEAKER_FILE
                        The path of the speaker file for voice cloning.
```

(Use the TTS from cli:
```
tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" --model_path "/Volumes/OTHER/models/coqui/XTTS-v2"  --config_path /Volumes/OTHER/models/coqui/XTTS-v2/config.json --language_idx en --speaker_idx 'Narelle Moon' --text "Cuttle is a web-based 2D parametric computer-aided design CAD tool. It is easy to learn, has a full-featured free tier, and—most importantly—it Just Works. It has a clean and usable interface, several high-quality tutorials for getting started, and a bunch of project templates of varying degrees of complexity, which serve as both a teaching tool" --out_path ./speech.wav
```
)

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
```

### License
