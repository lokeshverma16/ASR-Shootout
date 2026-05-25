# ASR Shootout: Benchmarking Indian Conversational Speech

This report presents a rigorous evaluation of Automatic Speech Recognition (ASR) systems for a blue-collar hiring platform in India. In this context, speech is typically noisy, code-switched (Hindi-English or Hinglish), and telephony-grade, with a critical focus on extracting candidate localities.

---

## 1. Goal

Evaluate and select the optimal ASR model for transcribing Hinglish conversational queries and extracting Bangalore locality names under realistic noise, speech speed, and volume conditions.

---

## 2. Models Evaluated

We benchmarked three ASR systems:

1.  **Deepgram (Nova-2-General):** Baseline. Chosen for its optimized production latency and multilingual support.
2.  **OpenAI Whisper (whisper-large-v3 via Groq API):** Standard open-source benchmark for robust transcriptions across diverse accents.
3.  **Sarvam AI (Saaras-v3 API):** An India-first ASR system specifically trained to capture regional nuances, code-switching (Hinglish), and Indian accents.

---

## 3. Dataset Design

The evaluation dataset consists of **20 self-recorded audio clips** in `.wav` format, containing natural Hinglish phrases and Bangalore locality names. 

### Conversational Context
Instead of reading flat lists of names, recordings mimic user responses (e.g., *"Bhaiya HSR Layout sector two chalna hai"* or *"Haan, main Koramangala mein rehta hoon"*).

### Acoustic Diversity
To stress-test model robustness, samples are recorded under the following varying conditions:
*   **Quiet environment (5 clips):** Baseline acoustic quality.
*   **Fan/Background noise (4 clips):** Steady low-frequency noise.
*   **Street/Traffic noise (4 clips):** Sudden, non-stationary street noise and honking.
*   **Whispered/Low-volume speech (4 clips):** Shy or low-voice candidates.
*   **Fast/Rushed speech (3 clips):** Candidates speaking under time pressure or speaking quickly.

---

## 4. Evaluation Metrics

### A. Word Error Rate (WER)
Calculates the percentage of word insertions, deletions, and substitutions. Transcripts are normalized (lowercased, punctuation removed, whitespaces collapsed) prior to calculation to prevent formatting mismatches from skewing results.

### B. Locality Detection Accuracy (Custom Metric)
A custom binary metric indicating whether the target locality name was correctly captured in the transcription. Since models often transcribe Hindi phonetically in Devanagari script (e.g. transcribing *"Koramangala"* as *"कोरमंगला"*), we map Roman localities to Devanagari variations and use **Fuzzy Substring Matching** with a threshold of `80%` to match them.
*   *Why this matters:* If a candidate says *"I live in Koramangala"* and the model transcribes *"मैं कोरमंगला में रहता हूँ"*, the standard WER will penalize it, but the entity extractor can successfully identify the candidate's location.

### C. Latency
Average round-trip response time (in seconds) for the API transcription request.

---

## 5. Benchmark Results

Below are the aggregated metrics compiled across all 20 recordings:

| Model | Avg. WER (%) | Locality Detection Acc. (%) | Avg. Latency (s) | Estimated Cost / 1k Hrs |
| :--- | :---: | :---: | :---: | :---: |
| **Deepgram (Nova-2)** | **83.43%** | 70.00% | 2.49s | ~$10.00 |
| **OpenAI Whisper (Groq)** | 104.65% | 80.00% | **0.46s** | ~$0.00 (Free Dev Tier) |
| **Sarvam AI (Saaras-v3)** | 109.43% | **85.00%** | 0.86s | ~$120.00 |

---

## 6. Qualitative & Failure Analysis

Reviewing the raw transcriptions reveals three primary failure modes:

### A. Phonetic Word Splits (The Hinglish Challenge)
When transcribing under quiet or slightly noisy conditions, models often split regional compound nouns into distinct Hindi dictionary words:
*   **Marathahalli:** Transcribed as **`मारा थाली`** (Deepgram: "hit plate"), **`मारत अली`** (Whisper: "Marat Ali"), or **`मारा था अली`** (Sarvam: "Ali had hit").
*   **Yelahanka:** Transcribed as **`Yellanka`** (Deepgram/Whisper) or **`ये लंका`** (Sarvam: "this Lanka") due to the soft, often swallowed "ha" sound.

### B. Auditory Hallucinations in Whispered Speech
In low-volume whispered speech, models struggled to resolve phonemes and hallucinated English/Hindi sounds:
*   **Silk Board:** Transcribed as **`Self board`** (Deepgram) or **`चलक बोर`** (Whisper), whereas Sarvam successfully captured it as **`सिल्क बोर्ड`**.
*   **Bellandur:** Transcribed as **`Leyland दूर`** (Deepgram) or **`पेलंदू`** (Whisper), whereas Sarvam successfully captured it as **`बेलनदूर`**.

### C. Background Noise Distortions
Street traffic noise led to high phoneme loss, particularly during fast speech:
*   **Banashankari:** Transcribed as **`बंद संक्री`** (Deepgram: "closed narrow") or **`बन्शंक्री`** (Whisper). Sarvam captured it as **`भंजशंकरी`**, showing high robustness in phonetic mapping under traffic noise.

---

## 7. Opinionated Recommendation

For production deployment in Vahan's blue-collar telephony pipeline, we recommend a **hybrid routing architecture** prioritizing **Sarvam AI (Saaras-v3)** as the primary engine and **Groq-hosted Whisper** as the fallback:

1.  **Why Sarvam AI is the primary choice:**
    *   **Entity Extraction Superiority:** At **85.00%**, Sarvam had the highest locality detection accuracy. It is highly optimized for Indian regional accents, successfully transcribing whispered and mumbled South Indian localities (like *Bellandur* and *Silk Board*) that Deepgram and standard Whisper missed.
    *   **Verbatim Integrity:** Even though it has a higher WER (109.43%), this is a misleading metric caused by script mismatch. Sarvam transcribes Hinglish speech in verbatim Devanagari, making it highly reliable for downstream NLP/entity parsing.
2.  **Why Groq Whisper is the fallback:**
    *   **Ultra-low Latency:** At **0.46s**, Groq-hosted Whisper is lightning-fast, making it excellent for high-concurrency real-time conversational agents where cost and latency are tight constraints.
3.  **Why Deepgram is bypassed:**
    *   In this evaluation, Deepgram (Nova-2) was slower (2.49s) and had the lowest locality detection rate (70.00%), struggling heavily with acoustic variations (whispered/fast speech).
