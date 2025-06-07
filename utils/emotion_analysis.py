# utils/emotion_analysis.py

from transformers import pipeline

# Load once when imported
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def detect_emotion(text):
    """
    Takes in user input and returns the top predicted emotion.
    """
    result = emotion_classifier(text)
    top_emotion = result[0][0]['label']
    return top_emotion.capitalize()
