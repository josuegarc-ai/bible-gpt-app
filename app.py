# Streamlit UI for Bible GPT v2.2 (Fully Synced with CLI Version)

import streamlit as st
from datetime import datetime
import openai
import requests
import json
import re
import os
import random
from googlesearch import search

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

def extract_json_from_response(response_text):
    try:
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)
        return json.loads(json_text)
    except:
        return None

def search_sermons(passage):
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    queries = [f"{pastor} sermon on {passage} site:youtube.com" for pastor in pastors]
    results = []
    for query, pastor in zip(queries, pastors):
        try:
            for url in search(query, num_results=1):
                results.append(f"[{pastor}]({url})")
        except Exception as e:
            results.append(f"{pastor}: ❌ No sermon found")
    return results

# === UI ===
st.set_page_config(page_title="TrueVine Bible GPT", layout="wide")
st.title("✝️ TrueVine Bible GPT")

mode = st.sidebar.selectbox("Select Mode", [
    "Bible Lookup",
    "Chat with GPT",
    "Practice Questions",
    "Verse of the Day",
    "Study Plan",
    "Faith Journal",
    "Prayer Starter",
    "Quick Devotional",
    "Small Group Guide",
    "Tailored Learning Path"
])

if mode == "Bible Lookup":
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16)")
    translation = st.selectbox("Translation", VALID_TRANSLATIONS)
    if st.button("Get Verse"):
        verse = fetch_verse(passage, translation)
        st.markdown(f"**📖 {passage} ({translation}):**\n\n{verse}")
        summary = ask_gpt(f"Summarize and explain this Bible verse clearly: '{verse}' ({passage}). Include a daily life takeaway.")
        st.markdown(f"**💡 Summary:**\n\n{summary}")
        cross_ref = ask_gpt(f"List 2-3 cross-referenced Bible verses related to: '{verse}' and explain their connection.")
        st.markdown(f"**🔗 Cross-References:**\n\n{cross_ref}")
        sermons = search_sermons(passage)
        st.markdown("**🎙️ Related Sermons from Selected Pastors:**")
        for s in sermons:
            st.markdown(f"- {s}")
        action = ask_gpt(f"Based on this verse: '{verse}', suggest one small, practical action someone can take today.")
        st.markdown(f"**🔥 Action Step:**\n\n{action}")

elif mode == "Chat with GPT":
    st.markdown("🙏 Start a conversation with Bible GPT")
    user_input = st.text_area("Type your thoughts or questions:")
    if st.button("Ask") and user_input:
        st.markdown(ask_gpt(user_input))

elif mode == "Practice Questions":
    book = st.text_input("📚 Choose a book of the Bible (e.g., Matthew, Genesis)")
    style = st.selectbox("🎯 Question style", ["multiple choice", "fill in the blank", "true or false"])
    if st.button("Start Practice"):
        score = 0
        for i in range(3):
            q_prompt = f"Generate a {style} Bible question from the book of {book} with 1 correct answer and 3 incorrect ones. Format the response in JSON with keys: 'question', 'correct', 'choices'."
            response = ask_gpt(q_prompt)
            q_data = extract_json_from_response(response)
            if not q_data:
                continue
            st.markdown(f"**Q{i+1}: {q_data['question']}**")
            if style == "multiple choice":
                choice = st.radio("Choose one:", q_data['choices'], key=i)
            else:
                choice = st.text_input("Your answer:", key=f"input_{i}")
            if st.button(f"Submit Answer {i+1}", key=f"btn_{i}"):
                if choice.strip().lower() == q_data['correct'].strip().lower():
                    st.success("✅ Correct!")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. Correct answer: {q_data['correct']}")
                    explain = ask_gpt(f"Why is this answer incorrect for: '{q_data['question']}' (correct: {q_data['correct']})? Provide a mini Bible study.")
                    st.markdown(f"📖 Teaching Moment:\n\n{explain}")
        st.markdown(f"**🏁 Final Score: {score}/3**")

elif mode == "Verse of the Day":
    books = ["Genesis", "Exodus", "Psalms", "Proverbs", "Matthew", "Mark", "Luke", "John"]
    ref = f"{random.choice(books)} {random.randint(1,5)}:{random.randint(1,20)}"
    verse = fetch_verse(ref)
    st.markdown(f"**🌅 {ref}:**\n\n{verse}")
    summary = ask_gpt(f"Summarize and reflect on this daily verse: '{verse}' ({ref}). Include encouragement and life application.")
    st.markdown(f"**💬 Reflection:**\n\n{summary}")
    action = ask_gpt(f"Based on this verse: '{verse}', suggest one small, practical action someone can take today.")
    st.markdown(f"**🔥 Action Step:**\n\n{action}")

elif mode == "Study Plan":
    topic = st.text_input("📘 What spiritual topic or theme would you like to study?")
    goal = st.text_input("🎯 What's your goal or timeframe? (e.g., 14 days, 1 month)")
    if st.button("Generate Study Plan"):
        prompt = (
            f"Create a thoughtful and theologically accurate Bible study plan about '{topic}' for the goal: '{goal}'.\n"
            f"Structure it day-by-day. Each day should include:\n"
            f"1. A relevant Bible verse or passage\n"
            f"2. A short devotional thought or explanation\n"
            f"3. A personal reflection or application question\n"
            f"4. For each devotional day in the study plan, provide a sincere, thoughtful and relevant prayer\n"
            f"End with a closing encouragement.\n"
            f"Keep it clean, simple, formatted for readability."
        )
        plan = ask_gpt(prompt)
        st.text_area("📅 Study Plan", plan, height=500)
        growth = ask_gpt(f"Analyze this Bible study plan and summarize spiritual growth achieved and two areas for continued growth: {plan}")
        st.markdown(f"🌱 **Growth Summary:**\n\n{growth}")

elif mode == "Faith Journal":
    entry = st.text_area("Write your thoughts or prayers")
    if st.button("Save Entry"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("journals", exist_ok=True)
        path = os.path.join("journals", f"journal_{ts}.txt")
        with open(path, "w") as f:
            f.write(entry)
        st.success(f"✅ Saved as journals/journal_{ts}.txt")
        if st.checkbox("Get spiritual growth insight"):
            insight = ask_gpt(f"Provide spiritual insight based on this journal: {entry}")
            st.markdown(f"💡 **Insight:**\n\n{insight}")

elif mode == "Prayer Starter":
    topic = st.text_input("🙏 What would you like to pray about today?")
    if st.button("Generate Prayer"):
        st.text_area("🕊️ Prayer", ask_gpt(f"Write a heartfelt prayer for: {topic}"), height=250)

elif mode == "Quick Devotional":
    theme = st.text_input("⚡ Devotional topic (e.g., grace, strength)")
    if st.button("Generate Devotional"):
        st.text_area("🧠 Devotional", ask_gpt(f"Give a 30-second devotional on '{theme}' with a verse, reflection, and prayer."), height=300)

elif mode == "Small Group Guide":
    topic = st.text_input("📘 Topic or passage for discussion")
    if st.button("Create Group Guide"):
        prompt = (
            f"Create a small group guide on '{topic}' with: \n"
            f"1. Icebreaker\n2. 3–5 discussion questions\n3. Leader notes\n4. Closing prayer."
        )
        st.text_area("👥 Group Guide", ask_gpt(prompt), height=500)

elif mode == "Tailored Learning Path":
    user_type = st.radio("👤 Choose learner type:", ["child", "adult"])
    goal = st.text_input("🎯 Learning goal")
    level = st.selectbox("📈 Bible knowledge level", ["beginner", "intermediate", "advanced"])
    style = st.selectbox("🧠 Preferred learning style", ["storytelling", "questions", "memory games", "reflection"])
    if st.button("Generate Path"):
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
            f"5. Prayer or life application."
        )
        st.text_area("📘 Tailored Bible Path", ask_gpt(prompt), height=600)

else:
    st.info("This mode will be expanded in future versions.")
