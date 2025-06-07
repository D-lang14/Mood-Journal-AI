# 🧠 AI Mood Journal

AI Mood Journal is a Streamlit-based web app that uses an AI-powered emotion classifier to help you reflect on your feelings, get helpful journaling tips, and stay motivated — all while visually tracking your mood history.

## 🚀 How to Run (GitHub Codespaces or Locally)

1. Clone this repo
2. Open terminal and run:
```
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the app:
```
python -m streamlit run main.py
# or
streamlit run main.py
```

## 🌟 Features
- 🤖 AI Mood Detection: Detects your mood using HuggingFace's distilroberta-base emotion model.
- ✍️ Journaling Tips: Provides emotion-specific journaling prompts to encourage mindful reflection.
- 💬 Motivational Messages: Personalized encouragement based on how you feel.
- 📈 Mood History Tracker: Line chart showing how frequently you experience different emotions.
- 💾 CSV Log: All journal entries are saved in mood_history.csv.

## 📸 Screenshot
*(Add one later once deployed)*

## 🧪 Model Used
- Model: j-hartmann/emotion-english-distilroberta-base
- Task: Text classification (emotion detection)

## 🤗 Supported Emotions:
- Joy
- Sadness
- Anger
- Love
- Fear
- Surprise
- Neutral

## 📦 Future Improvements
- Add user login to track individual mood history
- Add emojis and weekly trends
- Export entries to PDF or markdown
- Cloud sync or Google Drive backup

## 🙌 Credits
Built with 💙 by Disha Patil

Powered by 🤗 HuggingFace Transformers and Streamlit
