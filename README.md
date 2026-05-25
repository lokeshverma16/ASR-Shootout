# ASR Benchmarking Shootout: Indian Conversational Speech

This repository contains a reproducible benchmarking pipeline designed to evaluate Automatic Speech Recognition (ASR) systems for transcribing Hinglish conversational speech and extracting named entities (localities).

---

## Repository Structure

```text
gravity1/
├── audio/                   # Folder to place your .wav audio recordings
├── outputs/                 # Directory where benchmark results are saved
│   ├── benchmark_raw_results.csv  # Raw transcripts, individual latency, and metrics
│   └── benchmark_summary.csv      # Aggregated mean latency, WER, and locality accuracy
├── .env.example             # Template file for API keys
├── metadata.csv             # Dataset registry mapping audio files to ground truth & localities
├── README.md                # This instructions file
├── report.md                # Final markdown report template with benchmarks and findings
├── requirements.txt         # Python dependencies
└── run_benchmark.py         # Main benchmarking execution script
```

---

## Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Create a virtual environment and install the required libraries:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Setup Credentials
Copy the environment template and insert your API keys:
```bash
cp .env.example .env
```
Open `.env` and configure:
*   `DEEPGRAM_API_KEY`: For the Deepgram baseline.
*   `SARVAM_API_KEY`: For the India-focused Sarvam AI (Saaras-v3) model.
*   `GROQ_API_KEY` (Recommended - Free): For Whisper-large-v3, or `OPENAI_API_KEY` for Whisper-1.

### 4. Add Recordings
1. Record your conversational locality audio files using your phone mic.
2. Save them as `.wav` files inside the `audio/` directory.
3. Map the files to their expected transcriptions and target localities in `metadata.csv`.

---

## Running the Benchmark

Execute the pipeline script:
```bash
python run_benchmark.py
```

### Optional Arguments:
*   `--limit N`: Only process the first `N` samples in the registry (useful for testing).
*   `--audio-dir PATH`: Change the path to your audio directory (defaults to `audio`).
*   `--metadata PATH`: Path to a custom metadata CSV (defaults to `metadata.csv`).

Example (Test with a single file first):
```bash
python run_benchmark.py --limit 1
```

---

## Evaluation Metrics

1.  **Word Error Rate (WER):** Standard transcription error rate, calculated using normalized reference and hypothesis text (lowercased, punctuation removed) to prevent formatting bias.
2.  **Locality Detection Accuracy:** A custom binary metric indicating whether the target locality name was correctly captured. Uses fuzzy substring matching with a RapidFuzz threshold of `80%` to accommodate dialect spelling variants (e.g. *"Koramangala"* vs *"Kormangla"*).
3.  **Latency:** The round-trip HTTP response time for each transcription call.
