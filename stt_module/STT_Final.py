import base64
import io
import time
from pydub import AudioSegment
import torch
import torchaudio
from transformers import pipeline

# Initialize device for GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Speech-to-Text pipeline
pipe = pipeline("automatic-speech-recognition", model="facebook/seamless-m4t-v2-large")

def process_audio_chunks(base64_chunks):
    transcriptions = []
    for idx, encoded_audio in enumerate(base64_chunks):
        try:
            received_audio_bytes = base64.b64decode(encoded_audio)
            received_buffer = io.BytesIO(received_audio_bytes)

            audio = AudioSegment.from_file(received_buffer).set_frame_rate(16000).set_channels(1)
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)

            waveform, sample_rate = torchaudio.load(buffer)
            transcription = pipe(waveform.squeeze(0).numpy(), generate_kwargs={"tgt_lang": "arb"})
            transcriptions.append(transcription['text'])
        except Exception as e:
            print(f"Error processing chunk {idx+1}: {str(e)}")
    
    full_transcription = " ".join(transcriptions)
    return full_transcription
