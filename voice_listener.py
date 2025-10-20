#!/usr/bin/env python
# coding: utf-8

# In[2]:


# D:\ai_glasses\voice_listener.py

import threading
import pyaudio
import wave
import whisper
import time
from config import * # Import all settings from config.py

# Global flag to signal the main thread that a command was heard
voice_command_heard = threading.Event() 

def listen_for_command(whisper_model):
    """
    Runs continuously in a separate thread to listen for the command phrase.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=1024,
                    input_device_index=MIC_DEVICE_INDEX)

    print(f"[Thread] Voice Listener ACTIVE. Command: '{COMMAND_PROMPT}'")

    while True:
        try:
            frames = []

            # 1. Record a short chunk of audio
            for i in range(0, int(RATE / 1024 * RECORD_SECONDS)):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            # 2. Save the audio chunk
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 3. Transcribe using Whisper
            result = whisper_model.transcribe(WAVE_OUTPUT_FILENAME)
            transcribed_text = result["text"].strip().lower()

            # 4. Check for command phrase
            if COMMAND_PROMPT in transcribed_text:
                print(f"[ALERT] Command detected: {transcribed_text.upper()}")
                voice_command_heard.set()  # Set the flag to notify the main thread
                # Wait for the main thread to process the command
                time.sleep(5) 

        except Exception as e:
            # Handle PyAudio stream errors (less frequent now)
            print(f"[Thread Error] Audio stream problem: {e}")
            time.sleep(1)

    # Cleanup (This part is rarely reached in a continuous thread)
    stream.stop_stream()
    stream.close()
    p.terminate()


# In[ ]:




