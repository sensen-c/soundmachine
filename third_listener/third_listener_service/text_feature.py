import unicodedata
import re
import wave
import json
from vosk import Model, KaldiRecognizer

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def convert_audio_to_text(audio_file_path,model):
    # Open the audio file
    with wave.open(audio_file_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio file must be WAV format mono PCM with 16kHz sample rate")

        rec = KaldiRecognizer(model, wf.getframerate())

        # Read and process audio data
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())

        # Append the final result
        results.append(rec.FinalResult())

    # Combine results and extract text
    text = " ".join(json.loads(r).get("text", "") for r in results)

    print(f"Text: {text}")
    return [normalizeString(text)]







