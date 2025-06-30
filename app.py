import streamlit as st
import openai
import requests
import pyttsx3
import json
import re
import os
from datetime import datetime
from duckduckgo_search import DDGS

# ================= CONFIG =================
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-4o"
bible_api_base = "https://bible-api.com/"
valid_translations = ["web", "kjv", "asv", "bbe", "oeb-us"]

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    tts_enabled = True
except Exception:
    tts_enabled = False

# =============== UTILITIES ===============
def speak(text: str):
    if tts_enabled:
        engine.say(text)
        engine.runAndWait()
    else:
        print("🔇 Text-to-speech not available.")

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

def extract_json_from_response(response_text):
    try:
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL).group(0)
        return json.loads(json_text)
    except:
        return None

# =============== UI ===============
st.title("📖 TrueVine AI — Bible Companion")
mode = st.sidebar.selectbox("Choose a Mode", [
    "Bible Lookup",
    "Chat Mode",
    "Practice Questions",
    "Verse of the Day",
    "Study Plan",
    "Faith Journal",
    "Prayer Starter",
    "Fast Devotional",
    "Small Group Guide",
    "Learning Path"
])

if mode == "Bible Lookup":
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16)")
    translation = st.selectbox("Choose translation", valid_translations)
    audio = st.checkbox("🔊 Read verse aloud")
    if st.button("Search") and passage:
        try:
            verse_text = fetch_bible_verse(passage, translation)
            st.markdown(f"**Verse Text:**\n{verse_text}")
            summary_prompt = f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway."
            summary = ask_gpt(summary_prompt)
            st.markdown(f"**💡 AI Summary:**\n{summary}")
            cross_prompt = f"List 2-3 cross-referenced Bible verses related to: '{verse_text}' and explain their connection."
            st.markdown("**🔗 Cross-References:**")
            st.markdown(ask_gpt(cross_prompt))
            action_prompt = f"Based on this verse: '{verse_text}', suggest one small, practical action someone can take today."
            st.markdown("**🔥 Action Step:**")
            st.markdown(ask_gpt(action_prompt))
            sermons = search_sermons(passage)
            st.markdown("**🎙️ Related Sermons:**")
            for s in sermons:
                st.markdown(f"- {s}")
            if audio:
                speak(verse_text)
        except Exception as e:
            st.error(str(e))

elif mode == "Chat Mode":
    st.subheader("🙏 Talk to Bible GPT")
    prompt = st.text_area("What's on your heart today?")
    if st.button("Send") and prompt:
        response = ask_gpt(prompt)
        st.text_area("✝️ Bible GPT's Response", response, height=400)

elif mode == "Practice Questions":
    book = st.text_input("Book of the Bible (e.g., Matthew)")
    style = st.selectbox("Question Style", ["multiple choice", "fill in the blank", "true or false"])
    if st.button("Generate Questions") and book:
        score = 0
        for i in range(3):
            q_prompt = f"Generate a {style} Bible question from the book of {book} with 1 correct answer and 3 incorrect ones. Format the response in JSON with keys: 'question', 'correct', 'choices'."
            q_data = extract_json_from_response(ask_gpt(q_prompt))
            if q_data:
                st.markdown(f"**Q{i+1}:** {q_data['question']}")
                if style == "multiple choice":
                    choice = st.radio("Choose your answer:", q_data['choices'], key=f"q{i}")
                    if st.button(f"Submit Answer {i+1}"):
                        if choice == q_data['correct']:
                            st.success("Correct!")
                            score += 1
                        else:
                            st.error(f"Incorrect. The correct answer is: {q_data['correct']}")
                            explain_prompt = f"Why is this answer incorrect for: '{q_data['question']}' (correct: {q_data['correct']})? Provide a mini Bible study."
                            st.markdown(ask_gpt(explain_prompt))
        st.markdown(f"**🏁 Final Score: {score}/3**")

elif mode == "Verse of the Day":
    if st.button("Get Verse"):
        import random
        books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
                 "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
                 "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
                 "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
                 "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke",
                 "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
                 "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy",
                 "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
                 "Jude", "Revelation"]
        verse = f"{random.choice(books)} {random.randint(1,5)}:{random.randint(1,20)}"
        try:
            verse_text = fetch_bible_verse(verse)
            st.markdown(f"**🌅 {verse}**\n{verse_text}")
            summary_prompt = f"Summarize and reflect on this daily verse: '{verse_text}' ({verse}). Include encouragement and life application."
            st.markdown("**💬 Reflection:**")
            st.markdown(ask_gpt(summary_prompt))
            action_prompt = f"Based on this verse: '{verse_text}', suggest one small, practical action someone can take today."
            st.markdown("**🔥 Action Step:**")
            st.markdown(ask_gpt(action_prompt))
        except Exception as e:
            st.error(str(e))

elif mode == "Faith Journal":
    st.subheader("📝 Faith Journal Companion")
    entry = st.text_area("Journal your thoughts or prayers")
    if st.button("Save Entry") and entry:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_{timestamp}.txt"
        path = os.path.join("journals", filename)
        os.makedirs("journals", exist_ok=True)
        with open(path, "w") as f:
            f.write(entry)
        st.success(f"Saved journal to {filename}")
        if st.checkbox("💡 Generate Spiritual Insight"):
            st.markdown(ask_gpt(f"Provide spiritual insight based on this journal: {entry}"))

elif mode == "Study Plan":
    topic = st.text_input("📘 Topic (e.g., forgiveness)")
    goal = st.text_input("🎯 Goal or timeframe (e.g., 14 days)")
    if st.button("Generate Plan") and topic and goal:
        plan_prompt = (
            f"Create a Bible study plan about '{topic}' for '{goal}'. Each day should include a Bible verse, explanation, application question, and prayer. End with encouragement."
        )
        plan = ask_gpt(plan_prompt)
        st.text_area("📅 Study Plan", plan, height=600)
        summary_prompt = f"Analyze this Bible study plan and summarize spiritual growth achieved and two areas for continued growth: {plan}"
        st.markdown("🌱 Growth Summary:")
        st.markdown(ask_gpt(summary_prompt))

elif mode == "Prayer Starter":
    topic = st.text_input("🙏 Topic for prayer")
    if st.button("Generate Prayer") and topic:
        prompt = f"Write a heartfelt prayer for: {topic}"
        st.markdown("🕊️ Prayer:")
        st.markdown(ask_gpt(prompt))

elif mode == "Fast Devotional":
    theme = st.text_input("⚡ Devotional theme (e.g., grace, hope)")
    if st.button("Get Devotional") and theme:
        prompt = f"Give a 30-second devotional on '{theme}' with a verse, reflection, and prayer."
        st.markdown("🕊️ Devotional:")
        st.markdown(ask_gpt(prompt))

elif mode == "Small Group Guide":
    topic = st.text_input("📘 Topic or passage")
    if st.button("Generate Guide") and topic:
        prompt = f"Create a small group guide on '{topic}' with: Icebreaker, 3–5 questions, leader notes, and closing prayer."
        st.text_area("👥 Group Guide", ask_gpt(prompt), height=500)

elif mode == "Learning Path":
    user_type = st.selectbox("👤 Learner Type", ["adult", "child"])
    goal = st.text_input("🎯 Learning Goal")
    level = st.selectbox("📈 Knowledge Level", ["beginner", "intermediate", "advanced"])
    style = st.text_input("🧠 Learning Style (e.g., storytelling, memory games)")
    if st.button("Generate Path") and goal and style:
        prompt = (
            f"You are a Bible teacher creating a Duolingo-style path for a {user_type}. Goal: {goal}. Level: {level}. Style: {style}."
            f" Provide 5 lessons, each with title, verse, engaging explanation, interaction, and prayer."
        )
        st.text_area("📘 Tailored Bible Path", ask_gpt(prompt), height=600)
