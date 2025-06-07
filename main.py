# main.py
import streamlit as st
from transformers import pipeline

# Load HuggingFace emotion analysis pipeline (uses CPU by default)
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_classifier = load_emotion_model()

def analyze_emotion(text):
    results = emotion_classifier(text)[0]  # list of emotions with scores
    # Pick emotion with highest score
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

# Streamlit UI
st.title("📝 Daily Mood Journal with AI Therapist")

st.write("Write how you feel today. I'll analyze your mood, give journaling tips, and some motivational feedback!")

user_input = st.text_area("Your mood journal entry:", height=150)

if st.button("Analyze My Mood") and user_input.strip():
    with st.spinner("Analyzing your mood..."):
        emotion, confidence = analyze_emotion(user_input)
        st.subheader(f"Detected emotion: {emotion} ({confidence:.0%} confidence)")
        st.markdown(f"### Journaling Tip:")
        st.write(get_journaling_tip(emotion))
        st.markdown(f"### Motivational Message:")
        st.info(get_motivational_message(emotion))
elif user_input.strip() == "":
    st.write("Please enter how you feel to get started!")

