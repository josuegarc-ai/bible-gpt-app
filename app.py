# Streamlit UI for Bible GPT v2.2 (Updated for OpenAI Client use)

import streamlit as st
from datetime import datetime
import openai
import requests
import json
import re
import os

# === CONFIG ===
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"
BIBLE_API = "https://bible-api.com/"
VALID_TRANSLATIONS = ["web", "kjv", "asv", "bbe", "oeb-us"]

# === HELPERS ===
def ask_gpt(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.4,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a biblical mentor and teacher. You explain Scripture clearly, compassionately, and apply it to modern life with spiritual insight."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def fetch_verse(passage, translation="web"):
    if translation not in VALID_TRANSLATIONS:
        st.error(f"Invalid translation. Use one of {VALID_TRANSLATIONS}.")
        return ""
    url = f"{BIBLE_API}{passage}?translation={translation}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to fetch verse.")
        return ""
    return r.json().get("text", "")

# === UI ===
st.set_page_config(page_title="TrueVine Bible GPT", layout="wide")
st.title("✝️ TrueVine Bible GPT")

mode = st.sidebar.selectbox("Select Mode", [
    "Bible Lookup",
    "Chat with GPT",
    "Practice Questions (basic)",
    "Verse of the Day",
    "Study Plan",
    "Faith Journal",
    "Prayer Starter",
    "Quick Devotional",
    "Small Group Guide",
    "Tailored Learning Path"
])

if mode == "Bible Lookup":
    passage = st.text_input("Enter a passage (e.g., John 3:16)")
    translation = st.selectbox("Translation", VALID_TRANSLATIONS)
    if st.button("Get Verse"):
        verse = fetch_verse(passage, translation)
        st.markdown(f"**📖 {passage} ({translation}):**\n\n{verse}")
        summary = ask_gpt(f"Summarize and explain this Bible verse clearly: '{verse}' ({passage}). Include a daily life takeaway.")
        st.markdown(f"**💡 Summary:**\n\n{summary}")

elif mode == "Chat with GPT":
    user_input = st.text_area("Ask Bible GPT a question")
    if st.button("Ask") and user_input:
        st.markdown(ask_gpt(user_input))

elif mode == "Verse of the Day":
    import random
    books = ["Matthew", "Mark", "Luke", "John", "Romans", "Psalms", "Proverbs"]
    ref = f"{random.choice(books)} {random.randint(1,5)}:{random.randint(1,20)}"
    verse = fetch_verse(ref)
    st.markdown(f"**🌅 {ref}:**\n\n{verse}")
    reflection = ask_gpt(f"Reflect on this verse: '{verse}' ({ref}).")
    st.markdown(f"**💬 Reflection:**\n\n{reflection}")

elif mode == "Study Plan":
    topic = st.text_input("Topic (e.g., grace, forgiveness)")
    goal = st.text_input("Timeframe or goal (e.g., 14 days)")
    if st.button("Generate Plan"):
        prompt = f"Create a {goal} Bible study on {topic}, day by day with verses, devotions, reflection and prayers."
        st.text_area("📅 Study Plan", ask_gpt(prompt), height=400)

elif mode == "Faith Journal":
    entry = st.text_area("Write your journal entry")
    if st.button("Save Journal"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"journal_{ts}.txt", "w") as f:
            f.write(entry)
        st.success("Saved!")
        if st.checkbox("Generate spiritual insight"):
            insight = ask_gpt(f"Provide spiritual insight on: {entry}")
            st.markdown(f"**💡 Insight:**\n\n{insight}")

elif mode == "Prayer Starter":
    topic = st.text_input("What would you like to pray about?")
    if st.button("Generate Prayer"):
        st.text_area("🕊️ Prayer", ask_gpt(f"Write a heartfelt prayer for: {topic}"), height=250)

elif mode == "Quick Devotional":
    theme = st.text_input("Devotional topic")
    if st.button("Generate Devotional"):
        st.text_area("🧠 Devotional", ask_gpt(f"Give a 30-second devotional on '{theme}' with verse, reflection, and prayer."), height=300)

elif mode == "Small Group Guide":
    topic = st.text_input("Discussion topic or passage")
    if st.button("Create Guide"):
        prompt = f"Create a small group guide on {topic} with: Icebreaker, 3–5 discussion questions, leader notes, and a closing prayer."
        st.text_area("👥 Guide", ask_gpt(prompt), height=400)

elif mode == "Tailored Learning Path":
    user_type = st.radio("Learner type", ["child", "adult"])
    goal = st.text_input("Learning goal")
    level = st.selectbox("Bible knowledge level", ["beginner", "intermediate", "advanced"])
    style = st.selectbox("Preferred style", ["storytelling", "questions", "memory games", "reflection"])
    if st.button("Generate Path"):
        prompt = (
            f"Create a 5-lesson Bible learning path for a {user_type}. Goal: {goal}. Knowledge: {level}. Style: {style}.\n"
            f"Each lesson should have title, verse, explanation, interactive element, and prayer."
        )
        st.text_area("📘 Learning Path", ask_gpt(prompt), height=500)

else:
    st.info("This mode will be expanded in future versions.")
