# ✅ Bible GPT — Full Version v2.8 — All Modes Fully Integrated
# Includes all 11+ modes including conversational chat, practice, faith journaling, Bible Beta, and more

import os
import openai
import requests
import json
import re
import random
from datetime import datetime
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

# =============== MODES ===============
def run_bible_lookup():
    st.subheader("🔍 Bible Lookup")
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
    translation = st.selectbox("Choose translation:", valid_translations)
    if st.button("Lookup") and passage:
        try:
            verse_text = fetch_bible_verse(passage, translation)
            st.success(verse_text)
            summary = ask_gpt_conversation(f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway.")
            st.info(summary)
            action_step = ask_gpt_conversation(f"Based on this verse: '{verse_text}', suggest one small, practical action someone can take today.")
            st.write("🔥 **Action Step:**", action_step)
        except Exception as e:
            st.error(str(e))

def run_chat_mode():
    st.subheader("💬 Chat with Bible GPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for user, bot in st.session_state.chat_history:
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bible GPT:** {bot}")
    user_input = st.text_input("Ask a question:", key="chat")
    if user_input:
        response = ask_gpt_conversation(user_input)
        st.session_state.chat_history.append((user_input, response))
        st.experimental_rerun()

def run_practice_chat():
    st.subheader("🧠 Practice Chat")
    book = st.text_input("Bible book (e.g., Matthew):")
    style = st.selectbox("Question style:", ["multiple choice", "fill in the blank", "true or false"])
    if st.button("Start Practice"):
        st.session_state.practice_qs = []
        for _ in range(3):
            prompt = f"Generate a {style} Bible question from the book of {book} with 1 correct answer and 3 incorrect ones. Format the response in JSON with keys: 'question', 'correct', 'choices'."
            data = extract_json_from_response(ask_gpt_conversation(prompt))
            if data:
                st.session_state.practice_qs.append(data)
    if "practice_qs" in st.session_state:
        score = 0
        for i, q_data in enumerate(st.session_state.practice_qs):
            st.markdown(f"**Q{i+1}:** {q_data['question']}")
            with st.form(f"qform_{i}"):
                if style == "multiple choice":
                    user_answer = st.radio("Choose:", q_data['choices'], key=f"q_{i}")
                else:
                    user_answer = st.text_input("Your answer:", key=f"input_{i}")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if user_answer.lower() == q_data['correct'].lower():
                        st.success("✅ Correct!")
                        score += 1
                    else:
                        st.error(f"❌ Incorrect. Correct: {q_data['correct']}")
                        explanation = ask_gpt_conversation(f"Why is this answer incorrect for: '{q_data['question']}' (correct: {q_data['correct']})?")
                        st.info(explanation)
        st.write(f"🏁 Final Score: {score}/{len(st.session_state.practice_qs)}")

def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    books = ["John", "Matthew", "Romans", "Psalms"]
    verse = f"{random.choice(books)} {random.randint(1, 5)}:{random.randint(1, 20)}"
    verse_text = fetch_bible_verse(verse)
    st.write(f"**{verse}:** {verse_text}")
    reflection = ask_gpt_conversation(f"Reflect on this daily verse: '{verse_text}' ({verse}).")
    st.info(reflection)

def run_study_plan():
    st.subheader("📘 Bible Study Plan")
    topic = st.text_input("Study Topic (e.g., grace):")
    goal = st.text_input("Study Goal or Timeframe (e.g., 14 days):")
    if st.button("Generate Plan"):
        prompt = f"Create a theologically sound daily study plan on '{topic}' for {goal}."
        plan = ask_gpt_conversation(prompt)
        st.text_area("📖 Study Plan:", plan, height=400)
        growth = ask_gpt_conversation(f"Analyze this Bible study plan and summarize spiritual growth achieved and two areas for continued growth: {plan}")
        st.success("🌱 Growth Summary:")
        st.write(growth)

def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts or prayers:")
    if st.button("Save Entry"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("journals", exist_ok=True)
        with open(f"journals/journal_{timestamp}.txt", "w") as f:
            f.write(entry)
        st.success(f"Saved as journals/journal_{timestamp}.txt")
    if st.button("Spiritual Insight") and entry:
        insight = ask_gpt_conversation(f"Provide spiritual insight based on this journal: {entry}")
        st.info(insight)

def run_tailored_learning():
    st.subheader("📚 Tailored Bible Learning Path")
    user_type = st.selectbox("Learner type:", ["child", "adult"])
    goal = st.text_input("Learning goal (e.g., know Jesus, full Bible):")
    level = st.selectbox("Knowledge level:", ["beginner", "intermediate", "advanced"])
    style = st.multiselect("Preferred styles:", ["storytelling", "questions", "memory games", "reflection"])
    if st.button("Generate Path"):
        prompt = f"Create a Duolingo-style 5-lesson Bible path for a {user_type} with goal '{goal}', level '{level}', styles: {', '.join(style)}. Include verse, explanation, activity, prayer."
        st.text_area("📘 Lessons:", ask_gpt_conversation(prompt), height=400)

def run_bible_beta():
    st.subheader("📖 Bible Beta — Read, Listen, and Highlight")
    passage = st.text_input("Passage to display (e.g., Psalm 23):")
    voice = st.selectbox("Choose Voice:", ["Default", "TD Jakes (AI)", "Morgan Freeman (AI)"])
    if st.button("Load Bible Passage") and passage:
        text = fetch_bible_verse(passage)
        st.text_area("📖 Bible Passage:", text, height=300)
        if st.button("Play AI Audio"):
            st.info(f"🔊 Playing in {voice} voice (simulated)")
        highlight = st.text_area("Highlight text to summarize:")
        if st.button("Summarize Highlight") and highlight:
            st.info(ask_gpt_conversation(f"Summarize and explain this passage: {highlight}"))

def run_growth_summary():
    st.subheader("📈 Growth Summary")
    recent_input = st.text_area("Paste any journal, study plan, or reflection text:")
    if st.button("Generate Summary") and recent_input:
        summary = ask_gpt_conversation(f"Provide a growth summary and two focus areas based on this text: {recent_input}")
        st.success(summary)

def run_cross_reference():
    st.subheader("🔗 Cross Reference Summary")
    verse = st.text_input("Enter Bible verse to cross reference (e.g., Matthew 6:33):")
    if st.button("Generate Cross-References") and verse:
        original_text = fetch_bible_verse(verse)
        prompt = f"List 3 cross-referenced Bible verses related to: '{original_text}' and explain their connection."
        st.info(ask_gpt_conversation(prompt))

def run_bible_timeline_quiz():
    st.subheader("🕰️ Bible Timeline Quiz")
    if st.button("Start Quiz"):
        q = ask_gpt_conversation("Generate a timeline-based multiple choice quiz about Bible events. Format as: Question, 4 choices, Correct Answer")
        st.text_area("Quiz Question:", q, height=300)

def run_sermon_search():
    st.subheader("🎙️ Sermon Search by Verse")
    verse = st.text_input("Enter Bible verse (e.g., Romans 8:28):")
    if st.button("Find Sermons") and verse:
        prompt = f"Find sermon titles and short summaries related to '{verse}'."
        st.text_area("Sermon Results:", ask_gpt_conversation(prompt), height=300)

# =============== MAIN =====================
MODE_FUNCS = {
    "Bible Lookup": run_bible_lookup,
    "Chat with GPT": run_chat_mode,
    "Practice Chat": run_practice_chat,
    "Verse of the Day": run_verse_of_the_day,
    "Bible Study Plan": run_study_plan,
    "Faith Journal": run_faith_journal,
    "Tailored Learning Path": run_tailored_learning,
    "Bible Beta": run_bible_beta,
    "Growth Summary": run_growth_summary,
    "Cross Reference Summary": run_cross_reference,
    "Bible Timeline Quiz": run_bible_timeline_quiz,
    "Sermon Search by Verse": run_sermon_search
}

st.title("📖 TrueVine AI — Bible GPT")
mode = st.sidebar.radio("Choose a mode:", list(MODE_FUNCS.keys()))
MODE_FUNCS[mode]()
