##https://github.com/coqui-ai/TTS/tree/dev#installation
##https://huggingface.co/coqui/XTTS-v2/tree/main
## Limitation "XTTS can only generate text with a maximum of 400 tokens."?
## "[!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio."

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np
import sys
import re
from pathlib import Path
from textwrap import TextWrapper

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import PunktTokenizer

model_path="/Volumes/OTHER/models/coqui/XTTS-v2" # Local model path
speaker_wav=model_path + "/samples/en_sample.wav" # default speaker
input="input-long.txt" # text input file which is need to convert
outputFile="./speech.wav" # audio speech, result file
maxChar = 300

def load_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as reader:
            ## May add pre_process steps below. Remove empty lines,
            ## Store to a filename contains '_processed'
            processed_file = "_processed.txt"   # Use a default filename
            path = Path(file_path)
            processed_file = path.with_stem(''.join(path.stem) + "_processed")
            with open(processed_file, 'w', encoding='utf-8') as writer:
                for line in reader:
                    if line.strip():
                        writer.write(line)
            with open(processed_file, 'r', encoding='utf-8') as f:
                text = f.read()
                return text
    except FileNotFoundError:
        print("The file does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def capture_sentences(text, maxChar = 250, file_path = ""):
    """
        Tokenize sentence from text, remove trailing chars,
        split 1 sentence into 2 if len > maxChar,
        make chunks of combine sentence with len about <= maxChar
    """
    current_chunk = ""
    chunks = [] #list of combined sentences with total length under limit maxChar
    if not text:
        return np.array(chunks)

    # Tokenize sentence from string text file
    sentences = []
    sentences_tk = PunktTokenizer().tokenize(text.strip())  #nltk.sent_tokenize(text.strip())
#    print(f"--sentences_tk: {sentences_tk}")
    wrapper = TextWrapper(width=maxChar/2)
    for idx in range (0, len(sentences_tk)):
        sentences_tk[idx] = sentences_tk[idx].replace("\n", " ")
        if len(sentences_tk[idx]) > maxChar:
            wraps = wrapper.wrap(sentences_tk[idx])
            sentences.extend([wrap for wrap in wraps])
        else:
            sentences.append(sentences_tk[idx])
#    print(f"--sentences: {sentences}")

    for idx in range (0, len(sentences)):
        sentences[idx] = sentences[idx].replace("\n", " ")
#        print(f"--sentences[{idx}]-len: {len(sentences[idx])} - {sentences[idx]}")

        ## Combine sentence, need to check
        ## chunks.append(sentences[idx])    #without combination
        if len(current_chunk) + len(sentences[idx]) > maxChar:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += " " + sentences[idx]
#        print(f"--current_chunk-len-idx,{idx}: {len(current_chunk)} - {current_chunk}")
    if current_chunk:
        chunks.append(current_chunk)

#    print(f"\n--chunks: {chunks}")

    ## Save chunks to a filename contains '_chunks'
    processed_file = "_chunks.txt" # Use a default filename
    if file_path and Path(file_path):
        path = Path(file_path)
        processed_file = path.with_stem(''.join(path.stem) + "_chunks")
    with open(processed_file, 'w', encoding='utf-8') as writer:
        for idx in range (0, len(chunks)):
                writer.write(str(idx) + "\n")
                writer.write(chunks[idx] + "\n")

    return np.array(chunks)

def is_one_dimensional(array):
    """
    Checks if a NumPy array is one-dimensional.

    Args:
        array (numpy.ndarray): The input NumPy array.

    Returns:
        bool: True if the array is one-dimensional, False otherwise.
    """
    return len(array.shape) == 1

def merge_array(*array):
    """
    Merges a single-dimensional NumPy array into a single-dimensional array.

    Args:
        array (numpy.ndarray): The input single-dimensional NumPy array.

    Returns:
        numpy.ndarray: A single-dimensional NumPy array containing the all values.
    """
    return np.concatenate((array))

config = XttsConfig()
config.load_json(model_path + str("/config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
##model.cuda()

# Capture sentences from input
text = load_text_file(input)
sentenceArray = capture_sentences(text, maxChar, input)
print(f"--Got sentenceArray with size: {len(sentenceArray)}")

# Convert sentence to audio
#sentence="Israel has killed the leader of the militant group Hezbollah in a airstrike in Beirut, marking a further escalation of hostilities in the region."
def convertSentenceToAudioValue(sentence):
    """
    Convert a sentence to audio value
    """
    outputs = model.synthesize(
        sentence,
        config,
        speaker_wav=speaker_wav,
        gpt_cond_len=3,
        language="en",
    )
    return outputs["wav"]

audio_array_list = []
count = 0
if len(sentenceArray):
    for sentence in np.nditer(sentenceArray):
        print(f"--Converting sentenceArray idx: {count}/{len(sentenceArray)}")
        print(f"--sentence: {sentence}")
        audioArray = convertSentenceToAudioValue(str(sentence))
        if not is_one_dimensional(audioArray):
            print(f"Invalid sentenceArray NumPy array, not single-dimensional, array no: {count}")
            count = count + 1
            continue
        audio_array_list.append((audioArray))
        count = count + 1

combinedAudioArray = merge_array(*audio_array_list)
torchaudio.save(outputFile, torch.tensor(combinedAudioArray).unsqueeze(0), 24000)
