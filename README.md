
# Real-Time Emotion Analysis of Audio and Lyrics Using Multi-Label Classification
Presentation link: https://prezi.com/view/dxElOn50mgcNJ76rphEx/

This project implements a multimodal mood detection system that analyzes both audio and lyrics of music in real time. By combining acoustic features and natural language processing, the system detects and interprets the emotional tone of a musical piece.

## Features

Audio-based emotion recognition using MFCC features and a deep neural network
Lyrics-based emotion classification using fine-tuned BERT
Whisper ASR for automatic lyric transcription from vocals
Rule-based fusion strategy to combine predictions from audio and text

Two modes of use:
newfinal.py → CLI version for recording, transcribing, and analyzing emotions
appy.py → Streamlit GUI version with visual outputs

## System Components
1. Audio Model

Input: MFCC features extracted from .wav files
Model: Deep Neural Network (emotion_model_final.h5)
Output: Probabilities across mood classes

2. Lyrics Model

Input: Transcription from Whisper
Model: Fine-tuned BERT (bert_multilabel.pt)
Output: Probabilities for 28 GoEmotions labels, mapped to broader categories

3. Decision Fusion

Combines predictions from audio and lyrics
Uses confidence thresholds and heuristic rules for final mood classification

## Model Weights
This repository does not include:

bert_multilabel.pt (BERT weights)
emotion_model_final.h5 (audio model)

These files are excluded due to GitHub size limits.
You can generate them by running the notebooks inside:
scripts/Model Pretraining notebooks/

## Installation
Clone the repository:

```git clone https://github.com/shravanthibalasubramanian/Mood-and-Lyric-Based-Song-Sentiment-Analysis.git ```
```
cd Mood-and-Lyric-Based-Song-Sentiment-Analysis
```
Install dependencies:
```pip install -r requirements.txt```


Or manually:
pip install torch transformers torchaudio librosa sounddevice soundfile matplotlib tensorflow pandas streamlit

## Usage
on bash
```python scripts/newfinal.py```

Streamlit App
on bash
```
streamlit run appy.py
```

## References
Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” 2019
Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision (Whisper),” 2022
GoEmotions Dataset
Speech Emotion Recognition Dataset (Kaggle)
