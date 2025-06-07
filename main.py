import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Cache model loading
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_classifier = load_emotion_model()

# Analyze input text and get emotion
def analyze_emotion(text):
    results = emotion_classifier(text)[0]
    top_emotion = max(results, key=lambda x: x['score'])
    return top_emotion['label'], top_emotion['score']

# Tip based on emotion
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

# Motivation based on emotion
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

# Save entry to CSV
def save_entry(text, emotion, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "text": text, "emotion": emotion, "confidence": confidence}
    file_path = "mood_history.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    
    df.to_csv(file_path, index=False)

# Plot line chart of mood entries
def show_history_chart():
    file_path = "mood_history.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['count'] = 1

        fig, ax = plt.subplots(figsize=(10, 4))
        mood_counts = df.groupby([df['timestamp'].dt.date, 'emotion']).size().unstack().fillna(0)
        mood_counts.plot(ax=ax, kind='line', marker='o')

        plt.title("📈 Mood Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Entries")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("No mood history available yet. Start journaling!")

# -------- Streamlit UI --------

st.set_page_config(page_title="AI Mood Journal", layout="centered")
st.title("📝 Daily Mood Journal with AI Therapist")

# Session state for rerun control
if "new_entry_added" not in st.session_state:
    st.session_state["new_entry_added"] = False

st.write("Write how you feel today. I'll analyze your mood, give journaling tips, and motivational feedback!")

user_input = st.text_area("Your mood journal entry:", height=150)

if st.button("Analyze My Mood") and user_input.strip():
    with st.spinner("Analyzing your mood..."):
        emotion, confidence = analyze_emotion(user_input)
        st.subheader(f"Detected emotion: {emotion} ({confidence:.0%} confidence)")
        st.markdown("### Journaling Tip:")
        st.write(get_journaling_tip(emotion))
        st.markdown("### Motivational Message:")
        st.info(get_motivational_message(emotion))

        save_entry(user_input, emotion, confidence)
        st.success("✅ Entry saved! See mood trend below ⬇️")

        st.session_state["new_entry_added"] = True

st.markdown("---")
st.header("📊 Your Mood History")
show_history_chart()
