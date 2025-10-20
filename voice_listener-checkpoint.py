# D:\ai_glasses\voice_listener.py

import pyaudio
import wave
import whisper
import time
import threading
import os  # CRUCIAL: Used for os.system() and pathing
from config import * # Global flag to signal the main thread that a command was heard
voice_command_heard = threading.Event() 

# Helper to find the local ffmpeg path
def get_ffmpeg_path():
    """Returns the absolute path to ffmpeg.exe in the current working directory."""
    return os.path.join(os.getcwd(), "ffmpeg.exe")


def listen_for_command(whisper_model):
    """
    Runs continuously in a separate thread to listen for the command phrase.
    Bypasses problematic Whisper audio loading by using os.system() with explicit path.
    """
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=1024,
                        input_device_index=MIC_DEVICE_INDEX)
    except OSError as e:
        print(f"[Init Error] Failed to open microphone stream: {e}")
        # Terminate PyAudio and return if the stream can't be opened
        p.terminate()
        return

    print(f"[Thread] Voice Listener ACTIVE. Command: '{COMMAND_PROMPT}'")
    
    ffmpeg_exe_path = get_ffmpeg_path()

    while True:
        try:
            frames = []
            
            # 1. Record a short chunk of audio
            for i in range(0, int(RATE / 1024 * RECORD_SECONDS)):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            
            # 2. Save the audio chunk to a temporary file
            input_file = "raw_audio.wav"
            wf = wave.open(input_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 3. EXPLICIT FFMPEG CONVERSION (The Fix)
            temp_output_file = "temp_converted.wav"
            
            # Construct command using absolute paths and double quotes for safety
            conversion_command = (
                f'"{ffmpeg_exe_path}" -i "{input_file}" '
                f'-acodec pcm_s16le -ac 1 -ar {RATE} "{temp_output_file}" -y'
            )
            
            # Execute the command directly (silencing output with > nul 2>&1 on Windows)
            # os.system is run within the thread
            # NOTE: os.system() can still fail if Windows Defender blocks it, but this is the most direct call.
            os.system(conversion_command) 
            
            # 4. Transcribe using Whisper (using the converted file)
            # This file is guaranteed to be in a format Whisper can load without issue.
            result = whisper_model.transcribe(temp_output_file)
            
            transcribed_text = result["text"].strip().lower()

            # 5. Check for command phrase
            if COMMAND_PROMPT in transcribed_text:
                print(f"\n[ALERT] Command detected: {transcribed_text.upper()}")
                voice_command_heard.set()
                time.sleep(5) 

        except Exception as e:
            print(f"[Thread Error] Audio processing failed: {e}")
            time.sleep(1)

    # Cleanup 
    stream.stop_stream()
    stream.close()
    p.terminate()