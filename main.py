#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# D:\ai_glasses\main.py

import cv2
import pyttsx3
import easyocr
from ultralytics import YOLO
import threading
import time
import os
import sys

import whisper

# Import components from our project files
from config import *
from voice_listener import listen_for_command, voice_command_heard

# --- Helper Functions ---

def initialize_systems():
    """Initializes all AI models and hardware (camera/TTS/Whisper/OCR)."""
    global cap, engine, reader, whisper_model, last_spoken_time, last_proximity_alert_time

    print("--- Initializing AI Glasses Systems ---")

    # 1. Models
    whisper_model = whisper.load_model("tiny")
    model = YOLO("yolov8n.pt") 

    # 2. Camera (Using the configured index)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}. Exiting.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 3. TTS Engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)

    # 4. OCR Reader
    reader = easyocr.Reader(['en'], gpu=False) 

    # 5. Timer initialization
    last_spoken_time = time.time() - SPEAK_INTERVAL
    last_proximity_alert_time = time.time() - PROXIMITY_ALERT_INTERVAL

    print("Initialization Complete. Starting voice listener...")

    # Start the voice listener thread
    listener_thread = threading.Thread(target=listen_for_command, args=(whisper_model,), daemon=True)
    listener_thread.start()

    print("---------------------------------------")
    return model

def speak(text, is_alert=False):
    """Handles audio output."""
    global last_spoken_time, last_proximity_alert_time

    if is_alert:
        last_proximity_alert_time = time.time()
    else:
        last_spoken_time = time.time()

    engine.say(text)
    engine.runAndWait()


# --- Main Logic ---

def main_loop(model):
    """The continuous video processing loop."""
    global last_spoken_time, last_proximity_alert_time
    OCR_MODE = False

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream ended. Restarting...")
                time.sleep(5)
                continue

            annotated_frame = frame.copy()
            current_time = time.time()

            # --- CHECK VOICE COMMAND ---
            if voice_command_heard.is_set():
                OCR_MODE = True
                voice_command_heard.clear() # Reset the flag

            # --- OCR MODE EXECUTION ---
            if OCR_MODE:
                print("[OCR] Running text detection...")
                gray = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
                results = reader.readtext(gray)

                if results:
                    full_text = " ".join([text for (bbox, text, prob) in results])
                    speak(f"Reading: {full_text}")
                else:
                    speak("No text detected.")

                OCR_MODE = False
                print("[OCR] Resuming real-time detection.")
                time.sleep(1) # Small pause before resuming vision

            # --- REAL-TIME VISION MODE ---
            else:
                results = model(annotated_frame, verbose=False) # verbose=False for headless
                detected_objects = set()
                proximity_alert = False
                img_height = annotated_frame.shape[0]

                for r in results:
                    for box in r.boxes:
                        label = model.names[int(box.cls)]

                        # 1. Proximity Check
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_height_ratio = (y2 - y1) / img_height

                        if box_height_ratio > PROXIMITY_THRESHOLD:
                            proximity_alert = True

                        detected_objects.add(label)

                # 2. Audio Logic
                if proximity_alert and (current_time - last_proximity_alert_time > PROXIMITY_ALERT_INTERVAL):
                    speak("DANGER! Obstacle right in front!", is_alert=True)
                    print("[ALERT] High Proximity Warning.")

                elif detected_objects and (current_time - last_spoken_time > SPEAK_INTERVAL) and not proximity_alert:
                    narration_text = ", ".join(list(detected_objects))
                    speak(f"Detected: {narration_text}")
                    print(f"[INFO] Speaking: {narration_text}")

            # Note: No cv2.imshow for headless deployment, we rely only on audio/logs

        except KeyboardInterrupt:
            print("\nShutting down by user interrupt...")
            break
        except Exception as e:
            print(f"\nCRITICAL ERROR in main loop: {e}")
            time.sleep(5) # Wait before attempting to continue

    # --- Cleanup ---
    if cap.isOpened():
        cap.release()
    print("Application closed gracefully.")


if __name__ == "__main__":
    # Ensure the system doesn't crash if the OS module isn't loaded correctly (for safety)
    try:
        import os
    except ImportError:
        print("Warning: os module not available, functionality may be limited.")

    # Main execution flow
    yolo_model = initialize_systems()
    main_loop(yolo_model)


# In[ ]:




