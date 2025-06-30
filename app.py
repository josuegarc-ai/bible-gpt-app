import os
import openai
import requests
import json
import re
import random
from datetime import datetime
from duckduckgo_search import DDGS
import streamlit as st

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
    if response.status_code != 200:
        raise Exception(f"❌ Error {response.status_code}: Unable to fetch passage.")
    data = response.json()
    return data.get("text", "").strip()

def ask_gpt_conversation(prompt: str) -> str:
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

def extract_json_from_response(response_text):
    try:
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)
        return json.loads(json_text)
    except:
        return None

# =============== SERMON SEARCH ===============
def search_sermons_online(passage):
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    results = []
    for pastor in pastors:
        query = f"{pastor} sermon on {passage} site:youtube.com"
        try:
            for result in DDGS().text(query, max_results=1):
                results.append({"pastor": pastor, "url": result["href"]})
        except Exception as e:
            results.append({"pastor": pastor, "url": f"❌ Error searching"})
    return results

# =============== MODES ===============
def run_bible_lookup():
    st.subheader("📖 Bible Lookup")
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
    translation = st.selectbox("Choose translation:", valid_translations)
    if st.button("Fetch Verse") and passage:
        try:
            verse_text = fetch_bible_verse(passage, translation)
            st.success(verse_text)
            summary_prompt = f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway."
            summary = ask_gpt_conversation(summary_prompt)
            st.markdown("**💡 AI Summary:**")
            st.info(summary)
            cross_prompt = f"List 2–3 cross-referenced Bible verses related to: '{verse_text}' and explain their connection."
            cross = ask_gpt_conversation(cross_prompt)
            st.markdown("**🔗 Cross References:**")
            st.markdown(cross)
            sermons = search_sermons_online(passage)
            st.markdown("**🎙️ Related Sermons:**")
            for item in sermons:
                st.markdown(f"- {item['pastor']}: {item['url']}")
        except Exception as e:
            st.error(str(e))

def run_chat_mode():
    st.subheader("💬 Chat Mode")
    user_input = st.text_area("Ask a question or share a thought:")
    if st.button("Send") and user_input:
        response = ask_gpt_conversation(user_input)
        st.markdown("**✝️ Bible GPT:**")
        st.write(response)

def run_practice_chat():
    st.subheader("🧠 Practice Chat")
    book = st.text_input("Enter Bible book:")
    style = st.selectbox("Choose question style:", ["multiple choice", "fill in the blank", "true or false"])
    if st.button("Start Practice") and book:
        score = 0
        for i in range(3):
            q_prompt = f"Generate a {style} Bible question from the book of {book} with 1 correct answer and 3 incorrect ones. Format the response in JSON with keys: 'question', 'correct', 'choices'."
            response = ask_gpt_conversation(q_prompt)
            q_data = extract_json_from_response(response)
            if not q_data:
                continue
            st.markdown(f"**Q{i+1}: {q_data['question']}**")
            user_answer = ""
            if style == "multiple choice":
                user_answer = st.radio("Choose:", q_data['choices'], key=i)
            else:
                user_answer = st.text_input("Your answer:", key=i+10)
            if st.button(f"Submit Answer {i+1}", key=i+20):
                if user_answer.lower() == q_data['correct'].lower():
                    score += 1
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. Correct answer: {q_data['correct']}")
                    explain = ask_gpt_conversation(f"Why is this answer incorrect for: '{q_data['question']}' (correct: {q_data['correct']})?")
                    st.markdown("**📖 Teaching Moment:**")
                    st.write(explain)
        st.markdown(f"**🏁 Final Score:** {score}/3")

def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    if st.button("Get Daily Verse"):
        books = ["Genesis", "Exodus", "Matthew", "John", "Psalms", "Romans"]
        verse = f"{random.choice(books)} {random.randint(1,5)}:{random.randint(1,20)}"
        try:
            verse_text = fetch_bible_verse(verse)
            st.markdown(f"**{verse}**")
            st.success(verse_text)
            prompt = f"Summarize and reflect on this daily verse: '{verse_text}' ({verse}). Include encouragement and life application."
            reflection = ask_gpt_conversation(prompt)
            st.markdown("**💬 Reflection:**")
            st.write(reflection)
        except Exception as e:
            st.error(str(e))

def run_study_plan():
    st.subheader("📅 Study Plan")
    topic = st.text_input("Topic (e.g., grace, forgiveness):")
    goal = st.text_input("Goal/timeframe (e.g., 14 days):")
    if st.button("Generate Plan") and topic:
        prompt = f"Create a Bible study plan on '{topic}' for {goal}. Include daily verse, reflection, application, and prayer."
        plan = ask_gpt_conversation(prompt)
        st.text_area("Study Plan", plan, height=500)
        summary = ask_gpt_conversation(f"Analyze this Bible study plan and summarize spiritual growth achieved and two areas for continued growth: {plan}")
        st.markdown("**🌱 Growth Summary:**")
        st.write(summary)

def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:")
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"journal_{ts}.txt", "w") as f:
            f.write(entry)
        st.success("Saved successfully.")
        if st.checkbox("Get insight based on entry"):
            insight = ask_gpt_conversation(f"Provide spiritual insight based on this journal: {entry}")
            st.markdown("**💡 Insight:**")
            st.write(insight)

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    topic = st.text_input("Prayer topic:")
    if st.button("Generate Prayer") and topic:
        prompt = f"Write a heartfelt prayer about: {topic}"
        prayer = ask_gpt_conversation(prompt)
        st.text_area("🕊️ Prayer", prayer, height=300)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    theme = st.text_input("Devotional theme:")
    if st.button("Generate Devotional") and theme:
        prompt = f"Give a 30-second devotional on '{theme}' with a verse, reflection, and prayer."
        devo = ask_gpt_conversation(prompt)
        st.text_area("🕊️ Devotional", devo, height=300)

def run_small_group_generator():
    st.subheader("👥 Small Group Guide")
    topic = st.text_input("Discussion topic or passage:")
    if st.button("Generate Guide") and topic:
        prompt = f"Create a small group guide on '{topic}' with: Icebreaker, 3–5 questions, leader notes, closing prayer."
        guide = ask_gpt_conversation(prompt)
        st.text_area("📘 Guide", guide, height=500)

def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path")
    user_type = st.selectbox("User type:", ["child", "adult"])
    goal = st.text_input("Learning goal:")
    level = st.selectbox("Bible knowledge level:", ["beginner", "intermediate", "advanced"])
    style = st.selectbox("Learning style:", ["storytelling", "questions", "memory games", "reflection"])
    if st.button("Generate Path") and goal:
        prompt = f"Create a Duolingo-style Bible learning path for a {user_type} with goal '{goal}', level '{level}', and style '{style}'."
        result = ask_gpt_conversation(prompt)
        st.text_area("📘 Learning Path", result, height=500)

# =============== MAIN UI ===============
mode = st.sidebar.selectbox("Choose a mode:", [
    "Bible Lookup", "Chat with GPT", "Practice Chat", "Verse of the Day",
    "Study Plan", "Faith Journal", "Prayer Starter", "Fast Devotional",
    "Small Group Generator", "Tailored Learning Path"
])

if mode == "Bible Lookup": run_bible_lookup()
elif mode == "Chat with GPT": run_chat_mode()
elif mode == "Practice Chat": run_practice_chat()
elif mode == "Verse of the Day": run_verse_of_the_day()
elif mode == "Study Plan": run_study_plan()
elif mode == "Faith Journal": run_faith_journal()
elif mode == "Prayer Starter": run_prayer_starter()
elif mode == "Fast Devotional": run_fast_devotional()
elif mode == "Small Group Generator": run_small_group_generator()
elif mode == "Tailored Learning Path": run_learning_path_mode()
else: st.error("Unknown mode")
