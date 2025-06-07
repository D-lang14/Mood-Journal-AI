import streamlit as st
from transformers import pipeline
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load model
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_classifier = load_emotion_model()

def analyze_emotion(text):
    results = emotion_classifier(text)[0]
    top_emotion = max(results, key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score']

def get_journaling_tip(emotion):
    tips = {
        "joy": "Keep embracing positive moments and express gratitude in your journal today!",
        "anger": "Try to write about what triggered your anger and ways to calm down.",
        "sadness": "Allow yourself to express your feelings and consider writing about what comforts you.",
        "fear": "Reflect on what worries you and how you can face those fears step by step.",
        "surprise": "Note the unexpected things and how they made you feel.",
        "love": "Write about the people or things you love and why they matter to you.",
        "neutral": "Write freely about your day and your general mood.",
    }
    return tips.get(emotion.lower(), "Write honestly about how you feel today.")

def get_motivational_message(emotion):
    messages = {
        "joy": "Your happiness is your strength—keep shining!",
        "anger": "Channel your anger into positive action and growth.",
        "sadness": "This too shall pass. Better days are coming.",
        "fear": "Bravery is not absence of fear but acting despite it.",
        "surprise": "Embrace the unexpected; it often leads to growth.",
        "love": "Love fuels the soul. Keep nurturing it.",
        "neutral": "Every day is a fresh start. Make the most of it!",
    }
    return messages.get(emotion.lower(), "Keep being you. Every feeling is valid.")

DATA_FILE = "mood_journal_entries.csv"

def save_entry(text, emotion, confidence):
    entry = {
        "date": datetime.now().isoformat(),
        "text": text,
        "emotion": emotion,
        "confidence": confidence
    }
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(DATA_FILE, index=False)

def load_entries():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["date", "text", "emotion", "confidence"])

# App UI
st.set_page_config(page_title="Mood Journal AI", layout="centered")
st.title("📝 Daily Mood Journal with AI Therapist")

st.write("Write how you feel today. I'll analyze your mood, give journaling tips, and motivational feedback!")

user_input = st.text_area("Your mood journal entry:", height=150)

# Session state to prevent auto rerun errors
if "submitted" not in st.session_state:
    st.session_state.submitted = False
    st.session_state.result = None

if st.button("Analyze My Mood") and user_input.strip():
    emotion, confidence = analyze_emotion(user_input)
    save_entry(user_input, emotion, confidence)
    st.session_state.submitted = True
    st.session_state.result = (emotion, confidence, user_input)

if st.session_state.submitted and st.session_state.result:
    emotion, confidence, _ = st.session_state.result
    st.subheader(f"Detected Emotion: {emotion} ({confidence:.0%} confidence)")
    st.markdown("### Journaling Tip:")
    st.write(get_journaling_tip(emotion))
    st.markdown("### Motivational Message:")
    st.info(get_motivational_message(emotion))
    st.success("✅ Entry saved!")

# Mood history
st.markdown("---")
st.subheader("📈 Mood History")

entries = load_entries()

if not entries.empty:
    st.dataframe(entries.reset_index(drop=True))

    # Convert date
    entries['date_only'] = pd.to_datetime(entries['date']).dt.date

    # 📊 Mood counts
    mood_counts = entries['emotion'].value_counts().sort_index()
    total_entries = mood_counts.sum()
    mood_percentages = (mood_counts / total_entries * 100).sort_values()

    # 🎯 Most common mood (mode)
    mood_mode = entries['emotion'].mode().iloc[0]

    # 🖤 Dark Theme Chart
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    mood_percentages.plot(kind='line', marker='o', color='lime', ax=ax)

    ax.set_title("Mood Percentage Distribution")
    ax.set_xlabel("Mood")
    ax.set_ylabel("Percentage (%)")
    for i, (label, value) in enumerate(mood_percentages.items()):
        ax.text(i, value + 1, f"{value:.1f}%", color='white', ha='center')

    st.pyplot(fig)

    # ➕ Show mode below chart
    st.markdown(f"### 🏆 Most Frequent Mood: **{mood_mode.upper()}**")
    st.caption("Based on percentage of all journal entries.")

else:
    st.info("No entries yet. Once you add, they will appear here with a mood percentage chart.")
