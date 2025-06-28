# tts.py
"""
Text-to-speech (TTS) providers.
Each class implements a synthesize(text) -> audio bytes method.
"""

import requests
from abc import ABC, abstractmethod

class TextToSpeech(ABC):
    """Abstract base class for TTS providers."""
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes."""
        pass

class ElevenLabsTTS(TextToSpeech):
    """ElevenLabs voice synthesis API."""
    def __init__(self, api_key: str, voice_id: str = None):
        self.api_key = api_key
        self.voice_id = voice_id  # e.g. a specific voice UUID
    def synthesize(self, text: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        json_payload = {"text": text}
        response = requests.post(url, headers=headers, json=json_payload)
        response.raise_for_status()
        return response.content  # audio bytes

class GoogleCloudTTS(TextToSpeech):
    """Google Cloud Text-to-Speech (requires google-cloud-texttospeech)."""
    def __init__(self):
        from google.cloud import texttospeech
        self.client = texttospeech.TextToSpeechClient()
    def synthesize(self, text: str) -> bytes:
        from google.cloud import texttospeech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        # Example: English voice, neutral gender
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content  # raw PCM 16-bit

# class AmazonPollyTTS(TextToSpeech):
#     """Amazon Polly TTS via boto3."""
#     def __init__(self, region: str, access_key: str, secret_key: str):
#         import boto3
#         self.client = boto3.client(
#             'polly',
#             region_name=region,
#             aws_access_key_id=access_key,
#             aws_secret_access_key=secret_key
#         )
#     def synthesize(self, text: str) -> bytes:
#         response = self.client.synthesize_speech(
#             Text=text,
#             VoiceId='Joanna',  # or another available voice
#             OutputFormat='pcm'
#         )
#         # The AudioStream is a StreamingBody; read all
#         return response['AudioStream'].read()

# Factory function to select TTS provider
def get_tts_engine(config):
    provider = config.TTS_PROVIDER
    if provider == "elevenlabs":
        return ElevenLabsTTS(api_key=config.ELEVENLABS_API_KEY, voice_id=None)
    elif provider == "google":
        return GoogleCloudTTS()
    # elif provider == "amazon_polly":
    #     return AmazonPollyTTS(
    #         region=config.AMAZON_POLLY_REGION,
    #         access_key=config.AMAZON_POLLY_ACCESS_KEY,
    #         secret_key=config.AMAZON_POLLY_SECRET_KEY
    #     )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
