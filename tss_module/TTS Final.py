import base64
import json
import torch
import torchaudio
import time
import whisper
import numpy as np
from pydub import AudioSegment
from io import BytesIO
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load TTS model
config = XttsConfig()
config.load_json("config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, eval=True, vocab_path="vocab.json")
model.cuda()

# Load Whisper model
whisper_model = whisper.load_model("medium")  # You can use "small", "medium", "large", etc.

def encode_audio_to_base64(audio_segment):
    """Encodes an audio segment to Base64."""
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def chunk_base64_audio(base64_audio, chunk_size=1024):
    """Chunks the base64 string into smaller parts for streaming."""
    return [base64_audio[i:i + chunk_size] for i in range(0, len(base64_audio), chunk_size)]

def text_to_speech(text, speaker_audio_path="sample_0.wav"):
    """Converts text to speech using XTTS."""
    start_time = time.time()

    # Generate speech
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=speaker_audio_path,
        gpt_cond_len=3,
        language="ar",
    )

    end_time = time.time()
    print(f"Synthesis Time: {end_time - start_time:.2f} seconds")

    # Save synthesized audio
    audio_path = "response_audio.wav"
    torchaudio.save(audio_path, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
    return audio_path

def transcribe_with_whisper(audio_path):
    """Transcribes the audio using Whisper and returns word timestamps."""
    result = whisper_model.transcribe(audio_path, word_timestamps=True)

    words_data = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            words_data.append({
                "word": word_info["word"],
                "start_time": word_info["start"],
                "end_time": word_info["end"]
            })

    return words_data

def generate_ndjson(audio_path, words_data, chunk_size=1024):
    """Generates NDJSON with base64 chunks and Whisper-detected timestamps."""
    audio = AudioSegment.from_wav(audio_path)
    ndjson_entries = []

    for word_info in words_data:
        word = word_info["word"]
        start_time = word_info["start_time"]
        end_time = word_info["end_time"]

        # Extract the word's audio segment
        start_time_ms = int(start_time * 1000)
        end_time_ms = int(end_time * 1000)
        word_audio_segment = audio[start_time_ms:end_time_ms]

        # Encode the segment to base64
        base64_encoded_audio = encode_audio_to_base64(word_audio_segment)

        # Chunk the base64 audio
        base64_chunks = chunk_base64_audio(base64_encoded_audio, chunk_size)

        # Character-level timestamps (split word into characters)
        characters = list(word)
        num_chars = len(characters)
        char_times = np.linspace(start_time, end_time, num_chars + 1)
        character_start_times = char_times[:-1].tolist()
        character_end_times = char_times[1:].tolist()

        # Create NDJSON entry
        entry = {
            "base64_audio_chunks": base64_chunks,
            "alignment": {
                "word": word,
                "start_time_seconds": start_time,
                "end_time_seconds": end_time,
                "characters": characters,
                "character_start_times_seconds": character_start_times,
                "character_end_times_seconds": character_end_times
            },
            "normalized_alignment": {
                "word": word,
                "start_time_seconds": start_time,
                "end_time_seconds": end_time,
                "characters": characters,
                "character_start_times_seconds": character_start_times,
                "character_end_times_seconds": character_end_times
            }
        }
        ndjson_entries.append(entry)

    return ndjson_entries

def save_ndjson(output_data, file_path):
    """Saves the NDJSON output to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in output_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Example Usage
text = "مرحبا بك في عالم الذكاء الاصطناعي"  # Replace with your text
audio_path = text_to_speech(text)  # Generate speech
ndjson_file = "output_audio_chunks.ndjson"

# Step 1: Transcribe the generated speech using Whisper
start_time = time.time()
words_data = transcribe_with_whisper(audio_path)
print(f"Transcription Time: {time.time() - start_time:.2f} seconds")

# Step 2: Generate NDJSON using Whisper-detected timestamps
start_time = time.time()
ndjson_data = generate_ndjson(audio_path, words_data)
save_ndjson(ndjson_data, ndjson_file)
print(f"NDJSON saved at: {ndjson_file}. Processing Time: {time.time() - start_time:.2f} seconds")
