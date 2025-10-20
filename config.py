


# D:\ai_glasses\config.py

# --- YOLO/CV Settings ---
CAMERA_INDEX = 0  # 0 for default webcam, adjust for CSI/USB camera on device
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Audio/Voice Command Settings ---
COMMAND_PROMPT = "read this"  # The phrase to trigger OCR (lowercase)
RATE = 16000                 # Sample rate for microphone (Whisper standard)
CHANNELS = 1
RECORD_SECONDS = 3.0         # Duration to record after wake event

# --- Calibrated System Thresholds ---
# Use the values you determined during your testing and tuning phase!
# Example placeholder values used below:
PROXIMITY_THRESHOLD = 0.45   # Bounding box height ratio for DANGER alert (~1 meter)
SPEAK_INTERVAL = 4.0         # Seconds delay between standard object alerts
PROXIMITY_ALERT_INTERVAL = 1.0 # Seconds delay between high-priority alerts

# --- Deployment Paths ---
WAVE_OUTPUT_FILENAME = "command_audio.wav"
# You MUST find your microphone's index using the diagnostic script and update this:
MIC_DEVICE_INDEX = 1  # <--- REPLACE 1 WITH YOUR ACTUAL DEVICE INDEX







