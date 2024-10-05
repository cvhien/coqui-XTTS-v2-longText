##https://github.com/coqui-ai/TTS/tree/dev#installation
##https://huggingface.co/coqui/XTTS-v2/tree/main
## Limitation "XTTS can only generate text with a maximum of 400 tokens."?
## [!] Warning: The text length exceeds the character limit of 250 for language 'en', this might cause truncated audio.

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import sys
import numpy as np

model_path="~/workspace/models/coqui/XTTS-v2" # Local model path
speaker_wav=model_path + "/samples/samples_en_sample.wav" #default
n_lines_per_capture = 2  # Adjust this value according to your needs, not ok for big number result a long text
input="./input-short.txt" # text input file which is need to convert
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

def splitFileToStringArray(filename, n_lines_per_capture = 2):
    try:
        with open(filename, 'r') as file_handler:
            lines = file_handler.readlines()

            # Ensure each line is stripped of newline characters and other whitespace for clean processing
            cleaned_lines = [line.strip() for line in lines]

            # Now create a new array where each nth element (starting from 0) represents an n_lines_per_capture block of consecutive lines from your original file
            capture_array = np.array([
                ''.join(cleaned_lines[i:i + n_lines_per_capture])
                for i in range(0, len(cleaned_lines), n_lines_per_capture)
            ])

            print("--Capture text Array with size: " + str(len(capture_array)))
#            print(capture_array)
            return capture_array

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

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
    Merges a single-dimensional NumPy array into a combined single-dimensional array.

    Args:
        array (numpy.ndarray): The input single-dimensional NumPy array.

    Returns:
        numpy.ndarray: A single-dimensional NumPy array containing the all values.
    """
    # Use np.cumsum to calculate cumulative sums
    return np.concatenate((array))

config = XttsConfig()
config.load_json(model_path + str("/config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
#model.cuda()

#sentence1="Does Hezbollah represent Lebanon?"
#sentence3="Israel has killed the leader of the militant group Hezbollah in a airstrike in Beirut, marking a further escalation of hostilities in the region."
#sentence="Hello!"

sentenceArray = splitFileToStringArray(input, n_lines_per_capture)
print("--Got sentenceArray with size: " + str(len(sentenceArray)))

def convertSentenceToAudioValue(sentence):
    # ConvertSentenceToAudioValue
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
for sentence in np.nditer(sentenceArray):
    print(f"--sentenceArray idx: {count}")
    audioArray = convertSentenceToAudioValue(str(sentence))
    if not is_one_dimensional(audioArray):
        print("The NumPy array is not valid for single-dimensional operations.")
        print(f"Invalid sentenceArray no: {count}")
        continue
    audio_array_list.append((audioArray))
    count = count + 1
audioArrayCombined = merge_array(*audio_array_list)
torchaudio.save(outputFile, torch.tensor(audioArrayCombined).unsqueeze(0), 24000)
