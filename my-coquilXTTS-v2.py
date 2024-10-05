##https://github.com/coqui-ai/TTS/tree/dev#installation
##https://huggingface.co/coqui/XTTS-v2/tree/main
## Limitation "XTTS can only generate text with a maximum of 400 tokens."?
## [!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio.

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np
import sys
import re

model_path="/Users/st/workspace/models/coqui/XTTS-v2" # Local model path
speaker_wav=model_path + "/samples/samples_en_sample.wav" #default
num_sentences = 2  # Adjust this value according to your needs, not ok for big number results a too long text
input="./input-long.txt" # text input file which is need to convert
outputFile="./speech.wav" # audio speech, result file

def load_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
            return text
    except FileNotFoundError:
        print("The file does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def capture_sentences(file_path):
    try:
        # Initialize an empty list to store the captured sentences
        sentences = []

        # Read the text file line by line
        with open(file_path, 'r') as file:
            for line in file:
                # Use regular expression to split the line into sentences
                # Missing sentence mark, incorrect split in some cases, eg: 3.29am! -> todo
                sentence = re.split(r'[.!?]', line)

                # Remove leading/trailing whitespaces and empty strings
                sentence = [s.strip() for s in sentence if s.strip()]
                sentences.extend(sentence)
        return np.array(sentences)

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def combined_sentences(sentences, num_sentences = 2):
    # -> Improve to check max char length and reduce the number of combined sentences
    separator = '.' # space char: ' '
    combined_sentence_array = np.array([
        separator.join(sentences[i:i + num_sentences])
            for i in range(0, len(sentences), num_sentences)
        ])
    return combined_sentence_array

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
#model.cuda()

# Capture sentences from input
sentenceArray = capture_sentences(input)
print(f"--Got sentenceArray with size: {len(sentenceArray)}")
combinedSentenceArray = combined_sentences(sentenceArray, num_sentences)
print(f"--Got combinedSentenceArray with size: {len(combinedSentenceArray)}")

# Convert sentence to audio
#sentence="Does Hezbollah represent Lebanon?"
#sentence="Israel has killed the leader of the militant group Hezbollah in a airstrike in Beirut, marking a further escalation of hostilities in the region."
def convertSentenceToAudioValue(sentence):
    # Convert a sentence to audio value
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
for sentence in np.nditer(combinedSentenceArray):
    print(f"--Converting sentenceArray idx: {count}")
    audioArray = convertSentenceToAudioValue(str(sentence))
    if not is_one_dimensional(audioArray):
        print(f"Invalid sentenceArray NumPy array, not single-dimensional, array no: {count}")
        count = count + 1
        continue
    audio_array_list.append((audioArray))
    count = count + 1

combinedAudioArray = merge_array(*audio_array_list)
torchaudio.save(outputFile, torch.tensor(combinedAudioArray).unsqueeze(0), 24000)
