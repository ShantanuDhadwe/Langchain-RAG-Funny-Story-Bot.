# audio_handler.py
import os
from typing import IO, Optional, Union
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import speech_recognition as sr
import time

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


elevenlabs_client = None
if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("ElevenLabs client initialized.")
    except Exception as e:
        print(f"Error initializing ElevenLabs client: {e}")
        elevenlabs_client = None
else:
    print("WARNING: ELEVENLABS_API_KEY not found. TTS via ElevenLabs disabled.")

def text_to_speech_elevenlabs(text: str) -> Optional[BytesIO]: # Return BytesIO or None
    if not elevenlabs_client or not text:
        print("ElevenLabs client not initialized or no text provided.")
        return None
    
    try:
        print(f"ElevenLabs TTS: Converting '{text[:30]}...'")
        response_iterator = elevenlabs_client.text_to_speech.stream(
            voice_id="pNInz6obpgDQGcFmaJgB", # Adam
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128", # Standard mp3 format
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75, style=0.0, use_speaker_boost=True)
        )
        
        audio_stream = BytesIO()
        for chunk in response_iterator:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)
        print("ElevenLabs TTS: Stream converted to BytesIO.")
        return audio_stream
    except Exception as e:
        print(f"Error during ElevenLabs TTS: {e}")
        return None

def speech_to_text_from_mic(timeout: int = 7, phrase_time_limit: Optional[int] = 10) -> Union[str, None]:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("STT: Adjusting for ambient noise. Please wait a moment...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("STT: Listening... Speak now.")
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("STT: Processing speech...")
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"STT Recognized: {text}")
                    return text
                except sr.UnknownValueError:
                    print("STT: Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"STT: Could not request results; {e}")
                    return None
            except sr.WaitTimeoutError:
                print("STT: No speech detected within the timeout period")
                return None
    except Exception as e:
        print(f"STT: Error accessing microphone: {e}")
        return None


# if __name__ == "__main__":
#     print("Testing STT (speak for 5s):")
#     recognized = speech_to_text_from_mic(timeout=5, phrase_time_limit=5)
#     if recognized:
#         print(f"You said: {recognized}")
#         print("Testing TTS with what you said:")
#         audio_data = text_to_speech_elevenlabs(f"I heard you say: {recognized}")
#         if audio_data:
#             print("TTS successful. In a Streamlit app, you would use st.audio(audio_data).")
#             # For direct play test here (if you have playsound or similar and elevenlabs.play)
#             # from elevenlabs import play
#             # play(audio_data) # This would consume the BytesIO stream
#     else:
#         print("STT failed.")