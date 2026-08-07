# CLARA: Clip-Level Multimodal Alignment with VLM-Derived Rationales for Hateful Video Detection

## ⚠️ Ethics Statement

**This repository contains sensitive content that may be disturbing to some readers.**


The videos and data provided for annotation in this project have been sourced from publicly accessible social media platforms in compliance with all applicable laws and regulations. 

The content within these videos does not reflect the opinions, beliefs, or viewpoints of the research group or its members.

This dataset is intended solely for research purposes and must be treated with strict confidentiality. No personally identifiable information is included, and all procedures align with relevant legal and ethical standards.

---

This is the official implementation of **CLARA**: Clip-Level Multimodal Alignment with VLM-Derived Rationales for Hateful Video Detection (MM'26).

---

## Overview

CLARA consists of three key components:

- **Clip-level MoE-based Multimodal Encoding**  
  Flexible alignment of audio, visual, and textual modalities.

- **Local–Global Contrastive Learning**  
  Captures temporal consistency between short and long segments.

- **Rationale-Guided Transformer**  
  Enhances semantic understanding using VLM-generated rationales.

---

##  Project Structure

```bash
CLARA/
├── data_preprocess/
│   ├── get_raw_features/
│   │   ├── get_audio.py
│   │   ├── get_frame.py
│   │   ├── get_frames_for_rationale.py
│   │   ├── get_ocr.py
│   │   ├── get_rationale_llava.py
│   │   ├── get_rationale_qwen.py
│   │   ├── get_transcription.py
│   │   ├── get_video_clip.py
│   ├── extract_video_emb.py
├── model/
├── utils/
├── main.py
├── run_clara.sh
├── README.md
```

---

## Pipeline

The overall CLARA pipeline is:


```text
1.  Get video transcription (for clip segmentation)

Raw video
   ↓
get_audio.py
   ↓
{video_id}.wav
   ↓
get_transcription.py
   ↓
{video_id}.json
(Whisper transcription with segment timestamps)


2. Clip-level preprocessing

{video_id}.json + {video_id}.wav
   ↓
get_video_clip.py
   ↓
{video_id}/clipinfo.json
   ↓
get_frame.py
   ↓
clip-level sampled frames
   ↓
get_ocr.py
   ↓
clip-level OCR text


3. VLM rationale generation

Raw video
   ↓
get_frames_for_rationale.py
   ↓
frames for VLM reasoning
   ↓
get_rationale_qwen.py / get_rationale_llava.py
   ↓
VLM-derived rationales


4. Extract Video Embeddings

Audio + clip-level frames + transcripts + OCR + VLM rationales
   ↓
extract_video_emb.py
   ↓
{video_id}.pt
(clip-level multimodal and rationale embeddings)


5. Training and Evaluation

{video_id}.pt
   ↓
main.py
   ↓
Training & Evaluation
```

---


### Step 1: Get Video Transcription

First, extract the audio from each raw video and obtain timestamped transcriptions using Whisper. The transcription is subsequently used to determine the clip-level segmentation.

```bash
python data_preprocess/get_raw_features/get_audio.py
python data_preprocess/get_raw_features/get_transcription.py
```
For each video, this step produces:
```text
{video_id}.wav
   └── extracted audio

{video_id}.json
   └── Whisper transcription
        ├── segment id
        ├── start time
        ├── end time
        └── text
```


### Step 2: Clip-Level Preprocessing

Generate clip metadata from the timestamped transcription and audio, then extract clip-level frames and OCR text.

```bash
python data_preprocess/get_raw_features/get_video_clip.py
python data_preprocess/get_raw_features/get_frame.py
python data_preprocess/get_raw_features/get_ocr.py
```



For each video, `get_video_clip.py` generates:

```text
{video_id}/
└── clipinfo.json
     └── T clips
          ├── start / end / duration
          ├── transcript
          ├── silence indicator
          └── allocated frames
               ├── 40-frame budget
               ├── 60-frame budget
               ├── 80-frame budget
               ├── 100-frame budget
               └── 120-frame budget
```

The subsequent frame and OCR extraction produces clip-level data organised as:

```text
{video_id}/
├── clipinfo.json
│
├── clip_000/
│   ├── frames/
│   │   ├── frame_40/
│   │   │   └── frame_*.jpg
│   │   ├── frame_60/
│   │   ├── frame_80/
│   │   ├── frame_100/
│   │   └── frame_120/
│   │
│   └── ocr_text/
│       └── ocr_clip.json
│
├── clip_001/
│   └── ...
│
└── clip_XXX/
    └── ...
```

---

### Step 3: Generate VLM Rationales

Sample frames for VLM reasoning and generate video-level rationales using Qwen and/or LLaVA.

```bash
python data_preprocess/get_raw_features/get_frames_for_rationale.py
python data_preprocess/get_raw_features/get_rationale_qwen.py
python data_preprocess/get_raw_features/get_rationale_llava.py
```
Each rationale is subsequently represented using eight semantic fields:
```text
VLM rationale
   ├── objective summary
   ├── visual description
   ├── textual description
   ├── cross-modal relation
   ├── contextually important elements
   ├── final decision
   ├── reasons
   └── notes
```
---

### Step 4: Extract Video Embeddings

Extract clip-level multimodal embeddings and encode the VLM-derived rationales.

```bash
python data_preprocess/extract_video_emb.py
```
By default, BERT embeddings for transcripts and OCR text are extracted using `google-bert/bert-base-uncased`. For Chinese transcript and OCR text, set `bert_ckpt` to `google-bert/bert-base-chinese`.

For each video, the script generates a `{video_id}.pt` file containing clip-level multimodal embeddings, VLM-derived rationale embeddings, and corresponding availability masks.

```text
one video
   │
   ├── T clips
   │    ├── audio       → [T, 1280]
   │    ├── visual      → [T, 768]
   │    ├── transcript  → [T, D]
   │    └── OCR         → [T, D]
   │
   └── VLM rationale
        └── 8 fields    → [8, D]
```

`T` denotes the number of clips in a video, and `D` denotes the embedding dimension of the corresponding text encoder (e.g., `D = 768` for BERT).



---

## Step 5: Training and Evaluation

```bash
bash run_clara.sh
```

This will:
- Load precomputed embeddings
- Train the CLARA model
- Evaluate performance on test data

---
##  Datasets

## Datasets

The datasets used in this study can be accessed from their official repositories:

- **HateMM:** https://github.com/hate-alert/HateMM
- **MultiHateClip:** https://github.com/Social-AI-Studio/MultiHateClip
- **DeHate:** https://github.com/Multimodal-Intelligence-Lab-MIL/DeHate

Please follow the respective dataset licences and access requirements provided by the original authors.
