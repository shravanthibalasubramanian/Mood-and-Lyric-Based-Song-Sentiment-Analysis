import os
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import sounddevice as sd
import soundfile as sf
import whisper
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
import tempfile

MODEL_PATH = r"C:\Users\shrav\Desktop\music_mood_project\models\lyrics_emotion_model\bert_multilabel.pt"
AUDIO_DEEP_MODEL_PATH = r"C:\Users\shrav\Desktop\music_mood_project\models\emotion_model_final.h5"
LABEL_ENCODER_PATH = r"C:\Users\shrav\Desktop\music_mood_project\models\label_encoder.pkl"

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

MAPPING_TO_OVERALL = {
    'admiration': 'happy',
    'amusement': 'happy',
    'anger': 'angry',
    'annoyance': 'angry',
    'approval': 'happy',
    'caring': 'happy',
    'confusion': 'neutral',
    'curiosity': 'neutral',
    'desire': 'happy',
    'disappointment': 'sad',
    'disapproval': 'angry',
    'disgust': 'disgust',
    'embarrassment': 'sad',
    'excitement': 'happy',
    'fear': 'fear',
    'gratitude': 'happy',
    'grief': 'sad',
    'joy': 'happy',
    'love': 'happy',
    'nervousness': 'fear',
    'optimism': 'happy',
    'pride': 'happy',
    'realization': 'neutral',
    'relief': 'happy',
    'remorse': 'sad',
    'sadness': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels=28):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

@st.cache_resource
def load_models():
    lyrics_model = BertForMultiLabelClassification(num_labels=28)
    lyrics_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    lyrics_model.to(device)
    lyrics_model.eval()

    audio_model = load_model(AUDIO_DEEP_MODEL_PATH)
    le_audio = pd.read_pickle(LABEL_ENCODER_PATH)

    whisper_model = whisper.load_model("base")

    return lyrics_model, audio_model, le_audio, whisper_model

lyrics_model, audio_model, le_audio, whisper_model = load_models()

def record_audio(duration=20, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_audio.name, audio, fs, subtype='PCM_16')
    st.success(f"Recording saved.")
    return temp_audio.name

def play_audio(filename):
    data, fs = sf.read(filename, dtype='float32')
    sd.play(data, fs)
    sd.wait()
    st.success("Playback finished.")

def transcribe_audio(filename):
    st.info("Transcribing with Whisper...")
    try:
        result = whisper_model.transcribe(filename)
        text = result["text"].strip()
        st.success("Transcription complete!")
        st.text_area("Transcript:", text)
        return text
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return None

def extract_mfcc(audio_path, n_mfcc=13, max_len=130, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return mfcc

def predict_lyrics_emotions(text):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = lyrics_model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()

    top5_idx = probs.argsort()[-5:][::-1]
    st.subheader("Top 5 Lyrics-Based Emotions (GoEmotions):")
    for idx in top5_idx:
        st.write(f"{EMOTION_LABELS[idx]}: {probs[idx]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(EMOTION_LABELS, probs, color=plt.cm.plasma(np.linspace(0, 1, len(EMOTION_LABELS))))
    plt.xticks(rotation=90)
    plt.xlabel('GoEmotions Categories')
    plt.ylabel('Probability')
    plt.title('Lyrics-based Emotion Probabilities (GoEmotions)')
    st.pyplot(fig)

    overall_scores = {emotion: 0.0 for emotion in set(MAPPING_TO_OVERALL.values())}
    for idx, prob in enumerate(probs):
        label = EMOTION_LABELS[idx]
        mapped = MAPPING_TO_OVERALL.get(label, 'neutral')
        overall_scores[mapped] += prob

    sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    top_mood = sorted_overall[0][0]
    st.subheader(f"Final Overall Mood (Lyrics): {top_mood.capitalize()}")

    return overall_scores

def predict_audio_mood(audio_path):
    mfcc_features = extract_mfcc(audio_path, n_mfcc=13, max_len=130)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    prediction = audio_model.predict(mfcc_features)
    
    if prediction.ndim > 1 and prediction.shape[1] > 1:
        probs = prediction[0]
    else:
        probs = prediction.flatten()

    if hasattr(le_audio, 'inverse_transform'):
        class_labels = le_audio.inverse_transform(np.arange(len(probs)))
    else:
        class_labels = le_audio.classes_

    top_idx = np.argmax(probs)
    top_mood = class_labels[top_idx]
    st.subheader(f"Audio-Based Overall Mood: {top_mood.capitalize()}")

    df_probs = pd.DataFrame({
        'Emotion': class_labels,
        'Probability': probs
    }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
    st.table(df_probs)

    return top_mood, dict(zip(class_labels, probs))

def overall_analysis(lyrics_scores, audio_scores, audio_mood):
    lyrics_top_mood = max(lyrics_scores, key=lyrics_scores.get)
    lyrics_confidence = lyrics_scores[lyrics_top_mood]
    audio_confidence = max(audio_scores.values())

    st.subheader("Final Combined Analysis")
    st.write(f"Lyrics Top Moods: {', '.join([f'{k}: {v:.2f}' for k, v in lyrics_scores.items()])}")
    st.write(f"Audio Mood: {audio_mood} (confidence: {audio_confidence:.2f})")

    if audio_mood == lyrics_top_mood:
        combined_mood = audio_mood
        reason = "Audio and lyrics agree."
    elif audio_confidence < 0.35:
        combined_mood = lyrics_top_mood
        reason = f"Audio confidence ({audio_confidence:.2f}) too low; trusting lyrics."
    elif lyrics_confidence >= 0.8 and audio_mood in ['disgust', 'angry', 'fear']:
        combined_mood = lyrics_top_mood
        reason = f"Lyrics are strong ({lyrics_confidence:.2f}) and audio is negative."
    else:
        combined_mood = audio_mood
        reason = "Defaulting to audio mood."

    st.success(f"Combined Overall Mood: {combined_mood.capitalize()}")
    st.caption(f"Reason: {reason}")

st.title("Music Mood Analyzer")

if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

if st.button("Record Audio (20 seconds)"):
    st.session_state.audio_file = record_audio()

if st.session_state.audio_file:
    st.audio(st.session_state.audio_file)
    if st.button("Play Recording"):
        play_audio(st.session_state.audio_file)

    if st.button("Transcribe & Analyze"):
        text = transcribe_audio(st.session_state.audio_file)
        if text:
            lyrics_scores = predict_lyrics_emotions(text)
        else:
            st.warning("No transcript available, skipping lyrics emotion detection.")
            lyrics_scores = {'neutral': 1.0}

        audio_mood, audio_mood_scores = predict_audio_mood(st.session_state.audio_file)
        overall_analysis(lyrics_scores, audio_mood_scores, audio_mood)
