import streamlit as st
import openai
import requests
import json
import re
import os
from datetime import datetime
from duckduckgo_search import DDGS
import random

# ================= CONFIG =================
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-4o"
bible_api_base = "https://bible-api.com/"
valid_translations = ["web", "kjv", "asv", "bbe", "oeb-us"]

# =============== UTILITIES ===============
def fetch_bible_verse(passage: str, translation: str = "web") -> str:
    if translation not in valid_translations:
        raise ValueError(f"Unsupported translation. Choose from: {valid_translations}")
    url = f"{bible_api_base}{passage}?translation={translation}"
    response = requests.get(url)
    if response.status_code == 404:
        raise Exception(f"❌ Passage '{passage}' not found.")
    elif response.status_code != 200:
        raise Exception(f"❌ Error {response.status_code}: Unable to fetch passage.")
    data = response.json()
    return data.get("text", "").strip()

def ask_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.4,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a biblical mentor and teacher. You explain Scripture clearly, compassionately, and apply it to modern life with spiritual insight."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def search_sermons(passage):
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    results = []
    with DDGS() as ddgs:
        for pastor in pastors:
            query = f"{pastor} sermon on {passage} site:youtube.com"
            try:
                search_results = ddgs.text(query, max_results=1)
                if search_results:
                    url = search_results[0]['href']
                    results.append(f"[{pastor}]({url})")
                else:
                    results.append(f"{pastor}: ❌ No sermon found")
            except Exception as e:
                results.append(f"{pastor}: ❌ Error searching")
    return results

# =============== STREAMLIT UI ===============
st.set_page_config(page_title="TrueVine AI", layout="centered")
st.title("📖 TrueVine AI ✝️")

mode = st.selectbox("Choose a mode:", [
    "Bible Lookup",
    "Chat with GPT about faith",
    "Practice Questions",
    "Verse of the Day",
    "Bible Study Plan",
    "Faith Journal Companion",
    "Prayer Starter",
    "Fast Devotional",
    "Small Group Generator",
    "Tailored Learning Path"
])

if mode == "Bible Lookup":
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
    translation = st.selectbox("Choose translation:", valid_translations)
    audio = st.checkbox("🔊 Read verse aloud")
    if st.button("Lookup"):
        try:
            verse_text = fetch_bible_verse(passage, translation)
            st.subheader("🕊️ Verse Text")
            st.write(verse_text)
            summary = ask_gpt(f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway.")
            st.subheader("💡 AI Summary")
            st.write(summary)
            cross = ask_gpt(f"List 2-3 cross-referenced Bible verses related to: '{verse_text}' and explain their connection.")
            st.subheader("🔗 Cross References")
            st.write(cross)
            action = ask_gpt(f"Based on this verse: '{verse_text}', suggest one small, practical action someone can take today.")
            st.subheader("🔥 Action Step")
            st.write(action)
            sermons = search_sermons(passage)
            st.subheader("🎙️ Sermons")
            for result in sermons:
                st.markdown(result)
        except Exception as e:
            st.error(str(e))

elif mode == "Chat with GPT about faith":
    st.subheader("🙏 Chat with Bible GPT")
    user_input = st.text_input("Your question or thought:")
    if st.button("Send") and user_input:
        response = ask_gpt(user_input)
        st.subheader("✝️ Bible GPT Response")
        st.write(response)

elif mode == "Practice Questions":
    book = st.text_input("📚 Book of the Bible (e.g., Matthew):")
    style = st.selectbox("🎯 Question Style", ["multiple choice", "fill in the blank", "true or false"])
    if st.button("Start Practice"):
        score = 0
        for i in range(3):
            prompt = f"Generate a {style} Bible question from the book of {book} with 1 correct answer and 3 incorrect ones. Format the response in JSON with keys: 'question', 'correct', 'choices'."
            response = ask_gpt(prompt)
            try:
                q_data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group(0))
            except:
                st.warning("Skipping malformed question.")
                continue
            st.write(f"**Q{i+1}:** {q_data['question']}")
            user_answer = ""
            if style == "multiple choice":
                options = q_data['choices']
                user_answer = st.radio("Choices:", options, key=f"q{i}")
            elif style == "fill in the blank":
                user_answer = st.text_input("Your answer:", key=f"q{i}")
            elif style == "true or false":
                user_answer = st.radio("Your answer:", ["true", "false"], key=f"q{i}")
            if user_answer.lower() == q_data['correct'].lower():
                st.success("✅ Correct!")
                score += 1
            else:
                st.error(f"❌ Incorrect. Correct answer: {q_data['correct']}")
                explain = ask_gpt(f"Why is this answer incorrect for: '{q_data['question']}' (correct: {q_data['correct']})? Provide a mini Bible study.")
                st.write("📖 Teaching Moment:", explain)
        st.info(f"🏁 Final Score: {score}/3")

elif mode == "Verse of the Day":
    books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "Matthew", "Mark", "Luke", "John"]
    verse = f"{random.choice(books)} {random.randint(1, 5)}:{random.randint(1, 20)}"
    try:
        verse_text = fetch_bible_verse(verse)
        st.subheader(f"🌅 Verse of the Day ({verse})")
        st.write(verse_text)
        reflection = ask_gpt(f"Summarize and reflect on this daily verse: '{verse_text}' ({verse}). Include encouragement and life application.")
        st.subheader("💬 Reflection")
        st.write(reflection)
        action = ask_gpt(f"Based on this verse: '{verse_text}', suggest one small, practical action someone can take today.")
        st.subheader("🔥 Action Step")
        st.write(action)
    except Exception as e:
        st.error(str(e))

elif mode == "Bible Study Plan":
    topic = st.text_input("📘 What topic would you like to study?")
    goal = st.text_input("🎯 What's your goal or timeframe?")
    if st.button("Generate Plan"):
        prompt = (
            f"Create a thoughtful and theologically accurate Bible study plan about '{topic}' for the goal: '{goal}'.\n"
            f"Structure it day-by-day. Each day should include:\n"
            f"1. A relevant Bible verse or passage\n"
            f"2. A short devotional thought or explanation\n"
            f"3. A personal reflection or application question\n"
            f"4. A relevant prayer\n"
            f"End with a closing encouragement.\n"
        )
        plan = ask_gpt(prompt)
        st.subheader("📅 Study Plan")
        st.text_area("", plan, height=500)
        growth_summary = ask_gpt(f"Analyze this Bible study plan and summarize spiritual growth achieved and two areas for continued growth: {plan}")
        st.subheader("🌱 Growth Summary")
        st.write(growth_summary)

elif mode == "Faith Journal Companion":
    st.subheader("📝 Journal")
    journal_entry = st.text_area("Write your thoughts or prayer:")
    if st.button("Save Entry"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(journal_entry)
        st.success(f"Journal saved as {filename}")
        if st.checkbox("Would you like a spiritual insight?"):
            insight = ask_gpt(f"Provide spiritual insight based on this journal: {journal_entry}")
            st.subheader("💡 Insight")
            st.write(insight)

elif mode == "Prayer Starter":
    topic = st.text_input("🙏 What would you like to pray about today?")
    if st.button("Generate Prayer"):
        prayer = ask_gpt(f"Write a heartfelt prayer for: {topic}")
        st.subheader("🕊️ Prayer")
        st.write(prayer)

elif mode == "Fast Devotional":
    theme = st.text_input("⚡ Quick devotional topic (e.g., grace, strength):")
    if st.button("Generate Devotional"):
        devo = ask_gpt(f"Give a 30-second devotional on '{theme}' with a verse, reflection, and prayer.")
        st.subheader("🕊️ Devotional")
        st.write(devo)

elif mode == "Small Group Generator":
    topic = st.text_input("📘 Topic or passage for discussion:")
    if st.button("Generate Guide"):
        prompt = (
            f"Create a small group guide on '{topic}' with: \n"
            f"1. Icebreaker\n2. 3–5 questions\n3. Leader notes\n4. Closing prayer"
        )
        guide = ask_gpt(prompt)
        st.subheader("👥 Group Guide")
        st.write(guide)

elif mode == "Tailored Learning Path":
    st.subheader("📚 Personalized Bible Learning Path ✝️")
    user_type = st.selectbox("👤 Choose learner type", ["child", "adult"])
    goal = st.text_input("🎯 Learning goal")
    level = st.selectbox("📈 Bible knowledge level", ["beginner", "intermediate", "advanced"])
    style = st.selectbox("🧠 Preferred learning style", ["storytelling", "questions", "memory games", "reflection"])
    if st.button("Generate Learning Path"):
        prompt = (
            f"You are an experienced Bible teacher creating a Duolingo-style AI-driven learning path for a {user_type}.\n"
            f"Design a custom path based on these details:\n"
            f"- Goal: {goal}\n"
            f"- Knowledge Level: {level}\n"
            f"- Preferred Style: {style}\n"
            f"Include 5 lessons. For each, provide:\n"
            f"1. Title\n"
            f"2. Bible verse\n"
            f"3. Engaging explanation\n"
            f"4. Interactive element (like fill in the blank, multiple choice, or reflection)\n"
            f"5. Prayer or life application.\n"
            f"Make it simple, clear, and aligned to age and learning style."
        )
        response = ask_gpt(prompt)
        st.text_area("📘 Tailored Bible Path", response, height=600)
