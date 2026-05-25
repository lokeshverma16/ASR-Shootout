#!/usr/bin/env python3
import os
import re
import time
import argparse
import pandas as pd
import requests
from dotenv import load_dotenv
from jiwer import wer
from rapidfuzz import fuzz

# Load environment variables from .env file
load_dotenv()

# Text normalization helper for fair WER evaluation
def normalize_text(text):
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Standardize whitespace
    text = " ".join(text.split())
    return text

# Devanagari/Roman mapping for Bangalore localities to handle script differences in transcription
LOCALITY_MAPPING = {
    "koramangala": ["koramangala", "कोरमंगला", "कोरामंगला"],
    "indiranagar": ["indiranagar", "इंदिरानगर", "इन्दिरानगर"],
    "whitefield": ["whitefield", "व्हाइटफील्ड", "वाइटफील्ड"],
    "electronic city": ["electronic city", "इलेक्ट्रॉनिक सिटी", "इलेक्ट्रोनिक सिटी", "इलेक्ट्रॉनिकसिटी"],
    "marathahalli": ["marathahalli", "मराठाहल्ली", "मर्थहल्ली"],
    "jayanagar": ["jayanagar", "जयनगर"],
    "rajajinagar": ["rajajinagar", "राजाजीनगर"],
    "hebbal": ["hebbal", "हेब्बल"],
    "yelahanka": ["yelahanka", "येलाहंका"],
    "banashankari": ["banashankari", "बनाशंकरी", "बनशंकरी"],
    "hsr layout": ["hsr layout", "एचएसआर लेआउट", "एच एस आर लेआउट", "एचएसआर"],
    "btm layout": ["btm layout", "बीटीएम लेआउट", "बी टी एम लेआउट", "बीटीएम"],
    "majestic": ["majestic", "मैजेस्टिक", "मेजेस्टिक"],
    "silk board": ["silk board", "सिल्क बोर्ड", "सिल्कबोर्ड"],
    "bellandur": ["bellandur", "बेलंदूर", "बेलंदुर"],
    "sarjapur": ["sarjapur", "सरजापुर"],
    "bommanahalli": ["bommanahalli", "बोम्मनहल्ली", "बम्मनहल्ली"],
    "kr puram": ["kr puram", "केआर पुरम", "के आर पुरम", "केआरपुरम"],
    "peenya": ["peenya", "पीन्या", "पीण्या"],
    "yeshwanthpur": ["yeshwanthpur", "यशवंतपुर", "यश्वंतपुर"]
}

# Fuzzy named entity matcher for target localities
def is_locality_detected(transcription, target_locality, threshold=80):
    if not transcription or not target_locality:
        return False
    
    trans_norm = normalize_text(transcription)
    loc_key = target_locality.lower().strip()
    
    # Get all possible spelling variants (Roman & Devanagari)
    variants = LOCALITY_MAPPING.get(loc_key, [loc_key])
    
    for variant in variants:
        variant_norm = normalize_text(variant)
        
        # 1. Direct substring check
        if variant_norm in trans_norm:
            return True
            
        # 2. Check token n-grams to handle multi-word localities
        v_words = variant_norm.split()
        n = len(v_words)
        if n == 0:
            continue
            
        trans_words = trans_norm.split()
        if len(trans_words) < n:
            # Check partial ratio of entire string if transcription is shorter than locality
            if fuzz.partial_ratio(variant_norm, trans_norm) >= threshold:
                return True
            continue
            
        # Slide across transcription words to compare n-grams
        for i in range(len(trans_words) - n + 1):
            ngram = " ".join(trans_words[i:i+n])
            if fuzz.ratio(variant_norm, ngram) >= threshold:
                return True
                
        # 3. Fallback: check if the partial ratio is high
        if fuzz.partial_ratio(variant_norm, trans_norm) >= threshold:
            return True
            
    return False

# Deepgram ASR via REST API
def transcribe_deepgram(audio_path, api_key):
    # Using Nova-2 model; language 'hi' works well for Hindi/Hinglish
    url = "https://api.deepgram.com/v1/listen?model=nova-2&language=hi&smart_format=true"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    with open(audio_path, "rb") as f:
        start_time = time.time()
        response = requests.post(url, headers=headers, data=f, timeout=30)
        latency = time.time() - start_time
        
        response.raise_for_status()
        res_data = response.json()
        transcript = res_data['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript, latency

# OpenAI or Groq Whisper ASR via REST API
def transcribe_whisper(audio_path, api_key, is_groq=False):
    if is_groq:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        model = "whisper-large-v3"
    else:
        url = "https://api.openai.com/v1/audio/transcriptions"
        model = "whisper-1"
        
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "audio/wav")
        }
        data = {
            "model": model,
            "language": "hi"  # Hinting Hindi helps with Hinglish phonetic matches
        }
        start_time = time.time()
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        latency = time.time() - start_time
        
        response.raise_for_status()
        transcript = response.json()["text"]
        return transcript, latency

# Sarvam AI ASR via REST API
def transcribe_sarvam(audio_path, api_key):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": api_key
    }
    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "audio/wav")
        }
        data = {
            "model": "saaras:v3",
            "mode": "transcribe"
        }
        start_time = time.time()
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        latency = time.time() - start_time
        
        response.raise_for_status()
        res_data = response.json()
        transcript = res_data.get("transcript", "")
        return transcript, latency

def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark Pipeline for Locality Transcription")
    parser.add_argument("--metadata", default="metadata.csv", help="Path to metadata CSV file")
    parser.add_argument("--audio-dir", default="audio", help="Directory containing audio files")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save benchmark results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of audio files to process (for quick testing)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for metadata file
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file '{args.metadata}' not found. Please create it first.")
        return

    df = pd.read_csv(args.metadata)
    if args.limit:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} audio configurations from {args.metadata}.")

    # Verify API Keys
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    sa_key = os.getenv("SARVAM_API_KEY")
    
    # Support Groq API key for Whisper (100% free Developer Tier) as a fallback to OpenAI
    groq_key = os.getenv("GROQ_API_KEY")
    whisper_key = groq_key if groq_key else os.getenv("OPENAI_API_KEY")
    is_groq = True if groq_key else False

    models_to_run = []
    if dg_key:
        models_to_run.append("deepgram")
    else:
        print("Warning: DEEPGRAM_API_KEY not found. Skipping Deepgram.")

    if whisper_key:
        models_to_run.append("whisper")
    else:
        print("Warning: Neither GROQ_API_KEY nor OPENAI_API_KEY found. Skipping Whisper.")

    if sa_key:
        models_to_run.append("sarvam")
    else:
        print("Warning: SARVAM_API_KEY not found. Skipping Sarvam AI.")

    if not models_to_run:
        print("Error: No API keys configured in environment. Please set them in a .env file.")
        print("Check .env.example for details.")
        return

    print(f"Configured models to run: {', '.join(models_to_run)}")
    if "whisper" in models_to_run:
        print(f"  Whisper provider: {'Groq (whisper-large-v3)' if is_groq else 'OpenAI (whisper-1)'}")

    # Check audio files
    missing_files = []
    for idx, row in df.iterrows():
        audio_path = os.path.join(args.audio_dir, row['audio_file'])
        if not os.path.exists(audio_path):
            missing_files.append(row['audio_file'])

    if missing_files:
        print("\n=== WARNING: MISSING AUDIO FILES ===")
        print(f"The following {len(missing_files)} file(s) are missing from the '{args.audio_dir}' folder:")
        for mf in missing_files:
            print(f"  - {mf}")
        print("\nPlease record these files and place them in the folder before running.")
        print("====================================\n")
        
        # If all files are missing, we should stop
        if len(missing_files) == len(df):
            print("Stopping: No audio files available to benchmark.")
            return

    results = []

    # Process files
    for idx, row in df.iterrows():
        audio_file = row['audio_file']
        audio_path = os.path.join(args.audio_dir, audio_file)
        
        if not os.path.exists(audio_path):
            print(f"Skipping {audio_file} (not found)...")
            continue

        ref_text = row['expected_text']
        locality = row['locality']
        condition = row.get('condition', 'unknown')

        print(f"\nProcessing [{idx+1}/{len(df)}]: {audio_file} (Locality: {locality}, Condition: {condition})")
        
        row_result = {
            "audio_file": audio_file,
            "expected_text": ref_text,
            "locality": locality,
            "condition": condition
        }

        # Run Deepgram
        if "deepgram" in models_to_run:
            try:
                print("  Transcribing with Deepgram...")
                transcript, latency = transcribe_deepgram(audio_path, dg_key)
                norm_ref = normalize_text(ref_text)
                norm_trans = normalize_text(transcript)
                
                row_result["deepgram_transcript"] = transcript
                row_result["deepgram_latency"] = latency
                row_result["deepgram_wer"] = wer(norm_ref, norm_trans) if norm_ref else 0.0
                row_result["deepgram_locality_correct"] = is_locality_detected(transcript, locality)
            except Exception as e:
                print(f"  Deepgram Error: {e}")
                row_result["deepgram_transcript"] = "ERROR"
                row_result["deepgram_latency"] = None
                row_result["deepgram_wer"] = None
                row_result["deepgram_locality_correct"] = False

        # Run Whisper
        if "whisper" in models_to_run:
            try:
                print(f"  Transcribing with Whisper ({'Groq' if is_groq else 'OpenAI'})...")
                transcript, latency = transcribe_whisper(audio_path, whisper_key, is_groq)
                norm_ref = normalize_text(ref_text)
                norm_trans = normalize_text(transcript)
                
                row_result["whisper_transcript"] = transcript
                row_result["whisper_latency"] = latency
                row_result["whisper_wer"] = wer(norm_ref, norm_trans) if norm_ref else 0.0
                row_result["whisper_locality_correct"] = is_locality_detected(transcript, locality)
            except Exception as e:
                print(f"  Whisper Error: {e}")
                row_result["whisper_transcript"] = "ERROR"
                row_result["whisper_latency"] = None
                row_result["whisper_wer"] = None
                row_result["whisper_locality_correct"] = False

        # Run Sarvam
        if "sarvam" in models_to_run:
            try:
                print("  Transcribing with Sarvam AI...")
                transcript, latency = transcribe_sarvam(audio_path, sa_key)
                norm_ref = normalize_text(ref_text)
                norm_trans = normalize_text(transcript)
                
                row_result["sarvam_transcript"] = transcript
                row_result["sarvam_latency"] = latency
                row_result["sarvam_wer"] = wer(norm_ref, norm_trans) if norm_ref else 0.0
                row_result["sarvam_locality_correct"] = is_locality_detected(transcript, locality)
            except Exception as e:
                print(f"  Sarvam Error: {e}")
                row_result["sarvam_transcript"] = "ERROR"
                row_result["sarvam_latency"] = None
                row_result["sarvam_wer"] = None
                row_result["sarvam_locality_correct"] = False

        results.append(row_result)

    # Create DataFrame and save raw outputs
    res_df = pd.DataFrame(results)
    raw_csv_path = os.path.join(args.output_dir, "benchmark_raw_results.csv")
    res_df.to_csv(raw_csv_path, index=False)
    print(f"\nSaved raw results to: {raw_csv_path}")

    # Generate Summary Metrics
    print("\n" + "="*50)
    print("               BENCHMARK SUMMARY")
    print("="*50)

    summary_stats = []

    for model in models_to_run:
        # Filter rows without errors
        valid_rows = res_df[res_df[f"{model}_transcript"] != "ERROR"]
        if valid_rows.empty:
            print(f"\nModel {model.upper()}: No successful runs to summarize.")
            continue
            
        avg_wer = valid_rows[f"{model}_wer"].mean()
        avg_latency = valid_rows[f"{model}_latency"].mean()
        loc_acc = valid_rows[f"{model}_locality_correct"].sum() / len(valid_rows)
        
        print(f"\nModel: {model.upper()}")
        print(f"  Average WER:                 {avg_wer:.2%}")
        print(f"  Locality Detection Accuracy: {loc_acc:.2%}")
        print(f"  Average Latency:             {avg_latency:.2f}s")

        summary_stats.append({
            "Model": model.capitalize(),
            "Avg WER": f"{avg_wer:.2%}",
            "Locality Detection Acc": f"{loc_acc:.2%}",
            "Avg Latency (s)": f"{avg_latency:.2f}s"
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary results to: {summary_csv_path}")

if __name__ == "__main__":
    main()
