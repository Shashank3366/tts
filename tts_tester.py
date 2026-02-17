import os
import csv
import time
import requests
import librosa
import soundfile as sf
import numpy as np
from docx import Document
import random
from pathlib import Path

# --- Configuration ---
# API Keys (Provided by user)
ELEVENLABS_API_KEY = "71094e7fb933cc803d51c65f7bb820ddba86f02e6f03ebb046285211c57c8af8"
SARVAM_API_KEY = "sk_45ue4dhi_9QtYuVMELaO07cY8IwiIMBdy"

# URLs
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
SARVAM_URL = "https://api.sarvam.ai/text-to-speech" 

# Settings
INPUT_FILE = "english_test_texts.docx"
OUTPUT_DIR = "output"
REPORT_FILE = "tts_comparison_report.csv"

# Voice IDs
# 11Labs Voice ID (Example: '21m00Tcm4TlvDq8ikWAM' - Rachel, or use a specific one if known)
# Using a common pre-made voice ID for 11Labs (Rachel)
ELEVENLABS_VOICE_ID = "xYWUvKNK6zWCgsdAK7Wi" 

# Sarvam specifics
# Ensure we use a valid language code or speaker if required. 
# Sarvam details: target_language_code (e.g., 'hi-IN', 'en-IN'), speaker_id (if applicable)
SARVAM_TARGET_LANGUAGE = "en-IN" # Assuming English input for Indian context or general

# ---------------------

def setup_directories():
    """Creates output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Subdirectories for organization
    os.makedirs(os.path.join(OUTPUT_DIR, "11labs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "sarvam"), exist_ok=True)

def read_text_from_docx(file_path):
    """Reads text from a .docx file. Returns a list of non-empty strings."""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
        return full_text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def call_11labs_tts(text, index):
    """
    Calls 11Labs API.
    Returns: Path to saved audio file or None if failed.
    """
    url = ELEVENLABS_URL.format(voice_id=ELEVENLABS_VOICE_ID)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            filename = os.path.join(OUTPUT_DIR, "11labs", f"text_{index}_11labs.mp3")
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        else:
            print(f"11Labs API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"11Labs Exception: {e}")
        return None

def call_sarvam_tts(text, index):
    """
    Calls Sarvam AI (Bulbul) API.
    Returns: Path to saved audio file or None if failed.
    """
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "inputs": [text],
        "target_language_code": SARVAM_TARGET_LANGUAGE,
        "speaker": "neha", # Example speaker, check docs for valid ones like 'meera', 'pavithra', etc. Using 'meera' as default for EN/HI often.
        "pace": 1.0,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v3"
    }

    try:
        response = requests.post(SARVAM_URL, json=data, headers=headers)
        if response.status_code == 200:
            # Sarvam returns base64 string or direct audio? 
            # Usually Sarvam returns a JSON with base64 audio 'audios': ['base64string']
            try:
                response_json = response.json()
                if "audios" in response_json and len(response_json["audios"]) > 0:
                    import base64
                    audio_data = base64.b64decode(response_json["audios"][0])
                    filename = os.path.join(OUTPUT_DIR, "sarvam", f"text_{index}_sarvam.wav")
                    with open(filename, 'wb') as f:
                        f.write(audio_data)
                    return filename
                else:
                    print(f"Sarvam API Response format error or empty: {response_json}")
                    return None
            except ValueError:
                # If not JSON, maybe direct audio?
                print("Sarvam response was not JSON. Checking if direct audio...")
                # Fallback if API changes to direct stream
                filename = os.path.join(OUTPUT_DIR, "sarvam", f"text_{index}_sarvam.wav")
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename

        else:
            error_msg = f"Sarvam API Error: {response.status_code} - {response.text}"
            print(error_msg)
            with open("error_log.txt", "a") as err_log:
                err_log.write(f"Index {index}: {error_msg}\n")
            return None
    except Exception as e:
        print(f"Sarvam Exception: {e}")
        with open("error_log.txt", "a") as err_log:
            err_log.write(f"Index {index}: Sarvam Exception: {e}\n")
        return None

def analyze_audio(file_path):
    """
    Analyzes audio file for Duration, Average Pitch, and RMS Energy.
    Returns: dictionary of metrics.
    """
    if not file_path or not os.path.exists(file_path):
        return {"duration": 0, "avg_pitch": 0, "rms_energy": 0}

    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        avg_rms = np.mean(rms)
        
        # Pitch (F0)
        # Using piptrack for simplicity, or pyin for accuracy. pyin is slower.
        # Let's use simple pitch estimation.
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Filter out zero pitches (silence)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            avg_pitch = np.mean(pitch_values)
        else:
            avg_pitch = 0
            
        return {
            "duration": round(duration, 2),
            "avg_pitch": round(float(avg_pitch), 2),
            "rms_energy": round(float(avg_rms), 4)
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {"duration": 0, "avg_pitch": 0, "rms_energy": 0}

def main():
    setup_directories()
    
    # 1. Read Texts
    print(f"Reading texts from {INPUT_FILE}...")
    texts = read_text_from_docx(INPUT_FILE)
    if not texts:
        print("No texts found or file missing. Exiting.")
        return

    results = []

    print(f"Found {len(texts)} text segments. Randomly selecting one text for testing...")
    
    if texts:
        selected_text = random.choice(texts)
        print(f"Selected Text: \"{selected_text}\"")
        texts = [selected_text]
    else:
        print("No texts found.")
        return

    # 2. Process each text
    for i, text in enumerate(texts):
        print(f"\n--- Processing Text {i+1}/{len(texts)} ---")
        print(f"Text preview: {text[:50]}...")
        
        # 11Labs
        print("Calling 11Labs...")
        file_11labs = call_11labs_tts(text, i)
        metrics_11labs = analyze_audio(file_11labs)
        
        # Sarvam
        print("Calling Sarvam AI...")
        file_sarvam = call_sarvam_tts(text, i)
        metrics_sarvam = analyze_audio(file_sarvam)
        
        # Collect Result
        row = {
            "Text": text,
            "11Labs_File": file_11labs,
            "Sarvam_File": file_sarvam,
            "11Labs_Duration": metrics_11labs["duration"],
            "Sarvam_Duration": metrics_sarvam["duration"],
            "11Labs_Pitch": metrics_11labs["avg_pitch"],
            "Sarvam_Pitch": metrics_sarvam["avg_pitch"],
            "11Labs_RMS": metrics_11labs["rms_energy"],
            "Sarvam_RMS": metrics_sarvam["rms_energy"]
        }
        results.append(row)
        
        # Be nice to APIs
        time.sleep(1) 

    # 3. Generate Report
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    print(f"\nGenerating report at {report_path}...")
    
    fieldnames = [
        "Text", 
        "11Labs_File", "Sarvam_File", 
        "11Labs_Duration", "Sarvam_Duration", 
        "11Labs_Pitch", "Sarvam_Pitch", 
        "11Labs_RMS", "Sarvam_RMS"
    ]
    
    with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("Done! Check the output directory for results.")

if __name__ == "__main__":
    main()
