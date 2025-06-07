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
![Screenshot 2025-06-07 184920](https://github.com/user-attachments/assets/bcbffa1e-4cf5-4eaa-80f6-5914cc75b82b)
![Screenshot 2025-06-07 184936](https://github.com/user-attachments/assets/023a3739-dcb5-489a-9d2e-4a862bab6c88)
![Screenshot 2025-06-07 191425](https://github.com/user-attachments/assets/d4eafff1-d1be-4ebd-b99e-c122e107dcc8)

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
