# transcription.py
"""
Speech-to-text (STT) transcription providers.
Provides a common interface so providers can be swapped via config.
"""

import io
import time
import requests
from abc import ABC, abstractmethod

class Transcriber(ABC):
    """Abstract base class for transcription providers."""
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe raw audio bytes to text."""
        pass

# class WhisperTranscriber(Transcriber):
#     """OpenAI Whisper transcription (requires `whisper` package)."""
#     def __init__(self, model_size: str = "base"):
#         import whisper
#         self.model = whisper.load_model(model_size)
#     def transcribe(self, audio_bytes: bytes) -> str:
#         # Save audio bytes to temporary file for whisper
#         from tempfile import NamedTemporaryFile
#         with NamedTemporaryFile(suffix=".wav") as tmp:
#             tmp.write(audio_bytes)
#             tmp.flush()
#             result = self.model.transcribe(tmp.name)
#         return result["text"]

class DeepgramTranscriber(Transcriber):
    """Deepgram API transcription."""
    def __init__(self, api_key: str):
        self.api_key = api_key
    def transcribe(self, audio_bytes: bytes) -> str:
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"
        }
        response = requests.post("https://api.deepgram.com/v1/listen",
                                 headers=headers, data=audio_bytes)
        response.raise_for_status()
        data = response.json()
        # Extract transcript (assumes single channel, highest confidence)
        transcript = data['channels'][0]['alternatives'][0]['transcript']
        return transcript

# class AssemblyAITranscriber(Transcriber):
#     """AssemblyAI API transcription."""
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#     def transcribe(self, audio_bytes: bytes) -> str:
#         # Step 1: upload audio
#         upload_response = requests.post(
#             "https://api.assemblyai.com/v2/upload",
#             headers={"authorization": self.api_key},
#             data=audio_bytes
#         )
#         upload_response.raise_for_status()
#         audio_url = upload_response.json()['upload_url']

#         # Step 2: request transcript
#         transcript_response = requests.post(
#             "https://api.assemblyai.com/v2/transcript",
#             headers={
#                 "authorization": self.api_key,
#                 "content-type": "application/json"
#             },
#             json={"audio_url": audio_url}
#         )
#         transcript_response.raise_for_status()
#         transcript_id = transcript_response.json()['id']

#         # Step 3: poll until transcription is completed
#         while True:
#             status_response = requests.get(
#                 f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
#                 headers={"authorization": self.api_key}
#             )
#             status_response.raise_for_status()
#             status_json = status_response.json()
#             if status_json['status'] == 'completed':
#                 return status_json['text']
#             if status_json['status'] == 'error':
#                 raise RuntimeError(f"Transcription error: {status_json}")
#             time.sleep(1)  # wait and retry

# Factory function to select transcriber based on config
def get_transcriber(config):
    provider = config.STT_PROVIDER
    if provider == "whisper":
        return WhisperTranscriber(model_size="base")
    elif provider == "deepgram":
        return DeepgramTranscriber(api_key=config.DEEPGRAM_API_KEY)
    elif provider == "assemblyai":
        return AssemblyAITranscriber(api_key=config.ASSEMBLYAI_API_KEY)
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
