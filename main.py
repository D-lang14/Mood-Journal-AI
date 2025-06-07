# main.py

import streamlit as st
from utils.emotion_analysis import detect_emotion
import openai
import os
from datetime import datetime
import pandas as pd

# 💡 Set your OpenAI API key (add it as secret or env var in Codespaces)
openai.api_key = os.getenv("sk-abcd1234efgh5678abcd1234efgh5678abcd1234")

# 💅 Page Config
st.set_page_config(page_title="AI Mood Journal", page_icon="💬", layout="centered")
st.title("🧠 Daily Mood Journal with AI Therapist")

# ✍️ User mood input
user_input = st.text_area("How are you feeling today?", placeholder="Type your thoughts here...", height=200)

# 📤 When user clicks submit
if st.button("Analyze My Mood"):
    if not user_input.strip():
        st.warning("Please enter something first.")
    else:
        with st.spinner("Analyzing your mood..."):

            # 🧠 Emotion detection (via HuggingFace model)
            emotion = detect_emotion(user_input)

            # 🤖 Prompt OpenAI to generate reflection & motivation
            prompt = f"""
You are an empathetic AI therapist. The user wrote: "{user_input}"
1. Analyze their mood in a short sentence.
2. Ask a journaling reflection question.
3. Provide a motivational message in a comforting tone.
Only return 3 short bullet points.
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                result = response.choices[0].message.content

                # 🖼️ Show results
                st.subheader("🧠 Detected Emotion:")
                st.info(f"**{emotion}**")

                st.subheader("💬 AI Therapist Response")
                st.write(result)

                # 📦 Optional: Save to CSV
                save = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": user_input,
                    "emotion": emotion,
                    "response": result
                }
                df = pd.DataFrame([save])
                df.to_csv("data/mood_log.csv", mode='a', header=not os.path.exists("data/mood_log.csv"), index=False)

            except Exception as e:
                st.error("Something went wrong. Check your API key or internet connection.")
                st.exception(e)
