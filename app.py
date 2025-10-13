# ================================================================
# ✅ Bible GPT — v5.1 (Final Syntax & Block Fix)
# ================================================================

# ==== Core imports ====
import os
import re
import json
import random
import urllib.parse
import tempfile
import subprocess
import requests
import shutil
from datetime import datetime
import streamlit as st

import streamlit as st, os

st.write("🔑 API Key loaded:", bool(st.secrets.get("OPENAI_API_KEY")))
st.write("🔑 API Key env:", bool(os.getenv("OPENAI_API_KEY")))

# Page config should be set early in Streamlit apps
st.set_page_config(page_title="Bible GPT", layout="wide")

# ==== AI / NLP ====
from openai import OpenAI
import whisper

# ==== Web scraping ====
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ==== Media / YouTube ====
import yt_dlp
import imageio_ffmpeg

# ================================================================
# FFmpeg: supply binary via imageio-ffmpeg (no ffprobe required)
# ================================================================
_FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
_FFMPEG_DIR = os.path.dirname(_FFMPEG_BIN)

# Ensure PATH includes ffmpeg dir for subprocess/whisper/yt_dlp
os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
# yt_dlp prefers a directory path here:
os.environ["FFMPEG_LOCATION"] = _FFMPEG_DIR

# ================================================================
# CONFIG
# ================================================================
BIBLE_API_BASE = "https://bible-api.com/"
VALID_TRANSLATIONS = ["web", "kjv", "asv", "bbe", "oeb-us"]

# OpenAI
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
MODEL = "gpt-4o"

# ================================================================
# UTILITIES
# ================================================================
def fetch_bible_verse(passage: str, translation: str = "web") -> str:
    if translation not in VALID_TRANSLATIONS:
        raise ValueError(f"Unsupported translation. Choose from: {VALID_TRANSLATIONS}")

    passage_clean = passage.strip()
    encoded_passage = urllib.parse.quote(passage_clean)
    url = f"{BIBLE_API_BASE}{encoded_passage}?translation={translation}"

    try:
        resp = requests.get(url, timeout=12)
    except requests.RequestException as e:
        raise Exception(f"❌ Network error: {e}")

    if resp.status_code != 200:
        raise Exception(f"❌ Error {resp.status_code}: Unable to fetch passage. Check the reference formatting.")

    try:
        data = resp.json()
    except Exception:
        raise Exception("❌ Unexpected response format from Bible API.")

    text = data.get("text", "").strip()
    if not text:
        raise Exception("❌ Passage returned no text. Verify the book/chapter/verse.")
    return text


def ask_gpt_conversation(prompt: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": "You are a biblical mentor and teacher. You explain Scripture clearly, compassionately, and apply it to modern life with spiritual insight."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content.strip()


def extract_json_from_response(response_text: str):
    match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Found a JSON block in the response, but it was malformed.")
            return None

    try:
        start_index = -1
        first_brace = response_text.find('{')
        first_bracket = response_text.find('[')

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_index = first_brace
            end_char = '}'
        elif first_bracket != -1:
            start_index = first_bracket
            end_char = ']'

        if start_index != -1:
            end_index = response_text.rfind(end_char)
            if end_index > start_index:
                potential_json = response_text[start_index:end_index + 1]
                return json.loads(potential_json)
    except (json.JSONDecodeError, IndexError):
        pass

    st.error("No valid JSON block found in the AI response.")
    return None

# ================================================================
# ORIGINAL APP MODULES
# ================================================================
def search_sermons_online(passage: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    base_url = "https://www.youtube.com/results?search_query="
    results = []
    for pastor in pastors:
        query = f"{pastor} sermon on {passage}"
        search_url = base_url + urllib.parse.quote(query)
        try:
            response = requests.get(search_url, headers=headers, timeout=12)
            soup = BeautifulSoup(response.text, "html.parser")
            video_results = soup.find("a", {"id": "video-title"})
            if video_results:
                video_url = "https://www.youtube.com" + video_results['href']
                results.append({"pastor": pastor, "url": video_url})
            else:
                results.append({"pastor": pastor, "url": "❌ No result"})
        except Exception as e:
            results.append({"pastor": pastor, "url": f"❌ Error: {e}"})
    return results

def run_bible_lookup():
    st.subheader("📖 Bible Lookup")
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
    translation = st.selectbox("Choose translation:", VALID_TRANSLATIONS)
    if st.button("Fetch Verse") and passage:
        with st.spinner("Fetching and analyzing..."):
            try:
                verse_text = fetch_bible_verse(passage, translation)
                st.success(f"**{passage.strip().title()} ({translation.upper()})**\n\n> {verse_text}")
                summary = ask_gpt_conversation(f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway.")
                st.markdown("**💡 AI Summary:**")
                st.info(summary)
            except Exception as e:
                st.error(str(e))

def run_chat_mode():
    st.subheader("💬 Chat with GPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask a question or share a thought:")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a loving, biblically grounded mentor."}] + st.session_state.chat_history
        with st.spinner("Thinking..."):
            r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4)
            reply = r.choices[0].message.content.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def run_pixar_story_animation():
    st.subheader("🎥 Pixar-Studio Animated Bible Story")
    book = st.text_input("📘 Bible book (e.g., Exodus):")
    if st.button("🎬 Generate Pixar Story") and book:
        with st.spinner("Directing your story..."):
            story_prompt = f"Turn the Bible story from {book} into a Pixar-style film story for kids. Break it into 5 cinematic scenes, each with a title."
            response = ask_gpt_conversation(story_prompt)
            st.markdown("### 📚 Your Pixar-Style Story")
            st.markdown(response)

def run_practice_chat():
    st.subheader("🤠 Practice Chat")
    if "practice_state" not in st.session_state:
        st.session_state.practice_state = {"questions": [], "current": 0, "score": 0}
    S = st.session_state.practice_state
    if not S["questions"]:
        topic = st.text_input("Enter a Bible book for a 5-question quiz:", "John")
        if st.button("Start Practice"):
            with st.spinner("Generating questions..."):
                prompt = f"Generate 5 unique multiple-choice questions from {topic}. Format as a JSON list. Each object must have 'question' (str), 'choices' (list of 4 str), and 'correct' (str)."
                response = ask_gpt_conversation(prompt)
                questions_data = extract_json_from_response(response)
                if questions_data and isinstance(questions_data, list):
                    S["questions"] = questions_data
                    S["current"] = 0
                    S["score"] = 0
                    st.rerun()
                else:
                    st.error("Failed to generate valid questions. Please try again.")
    elif S["current"] < len(S["questions"]):
        q = S["questions"][S["current"]]
        st.markdown(f"**Question {S['current'] + 1}:** {q.get('question', 'No question text found.')}")
        choices = q.get('choices', [])
        if choices:
            ans = st.radio("Choose:", choices, key=f"q_{S['current']}")
            if st.button("Submit Answer", key=f"submit_{S['current']}"):
                if ans == q['correct']:
                    st.success("✅ Correct!")
                    S["score"] += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer was: {q['correct']}")
                S["current"] += 1
                st.rerun()
    else:
        st.success(f"**Quiz complete! Your score: {S['score']}/{len(S['questions'])}**")
        if st.button("Start New Quiz"):
            st.session_state.practice_state = {"questions": [], "current": 0, "score": 0}
            st.rerun()

def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:", height=200)
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("journal_entries", exist_ok=True)
        filename = f"journal_entries/journal_{ts}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(entry)
        st.success(f"Saved to {filename}.")

def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path")
    goal = st.text_input("What is your learning goal? (e.g., 'understand grace')")
    level = st.selectbox("Your Bible knowledge level:", ["Beginner", "Intermediate", "Advanced"])
    if st.button("Generate Path") and goal:
        with st.spinner("Designing your path..."):
            prompt = f"Design a 7-day Bible learning path for a {level} learner with the goal '{goal}'. For each day, provide a topic, scripture, and a reflection question."
            result = ask_gpt_conversation(prompt)
            st.text_area("📘 Your Learning Path", result, height=500)

def run_bible_beta():
    st.subheader("📘 Bible Chapter Reader")
    book = st.text_input("Book (e.g., Romans):")
    chapter = st.number_input("Chapter:", min_value=1, step=1, value=1)
    if st.button("Read Chapter") and book:
        with st.spinner(f"Loading {book} {chapter}..."):
            try:
                text = fetch_bible_verse(f"{book} {chapter}")
                st.text_area(f"📖 {book} {chapter}", value=text, height=400)
            except Exception as e:
                st.error(str(e))

def _convert_to_wav_if_needed(src_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wav_path = tmp_file.name
    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"ffmpeg failed: {e.stderr}")
    return wav_path

def download_youtube_audio(url: str) -> tuple[str, str, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        output_path = temp_file.name
    ydl_opts = {"format": "bestaudio/best", "outtmpl": output_path, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Untitled")
        uploader = info.get("uploader", "Unknown")
    return output_path, uploader, title

def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    yt_link = st.text_input("📺 YouTube Link:")
    if st.button("⏺️ Transcribe & Summarize") and yt_link:
        with st.spinner("Processing..."):
            try:
                audio_path, uploader, title = download_youtube_audio(yt_link)
                model = whisper.load_model("base")
                transcription = model.transcribe(audio_path)
                transcript_text = transcription.get("text", "")
                st.text_area("📝 Transcript", transcript_text, height=300)
                summary = ask_gpt_conversation(f"Summarize the key points of this sermon by {uploader} titled '{title}':\n{transcript_text[:2000]}")
                st.markdown("### 🧠 Sermon Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"Error: {e}")

def run_study_plan():
    st.subheader("📅 Personalized Bible Study Plan")
    goal = st.text_input("What is your study goal? (e.g., 'Understand forgiveness')")
    duration = st.slider("How many days should the plan last?", 7, 30, 14)
    if st.button("Generate Study Plan") and goal:
        with st.spinner("Creating your plan..."):
            prompt = f"Create a {duration}-day Bible study plan for the goal: '{goal}'. For each day, give a theme, 1-2 passages, and a reflection question."
            plan = ask_gpt_conversation(prompt)
            st.text_area("📘 Your Study Plan", plan, height=600)

def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    if st.button("Get Today's Verse"):
        with st.spinner("Finding inspiration..."):
            prompt = "Provide one encouraging Bible verse (e.g., Philippians 4:13 KJV) and write a short, 2-3 sentence reflection on it."
            response = ask_gpt_conversation(prompt)
            st.success(response)

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    theme = st.text_input("What is on your heart? (e.g., gratitude, anxiety)")
    if st.button("Generate Prayer") and theme:
        with st.spinner("Composing a prayer..."):
            prayer = ask_gpt_conversation(f"Write a short, heartfelt prayer starter on the theme of {theme}.")
            st.text_area("Your Prayer Starter", prayer, height=200)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    topic = st.text_input("What topic do you need encouragement on? (e.g., hope, perseverance)")
    if st.button("Generate Devotional") and topic:
        with st.spinner("Writing your devotional..."):
            devo = ask_gpt_conversation(f"Compose a 150-word devotional on {topic} with one primary verse.")
            st.text_area("Your Devotional", devo, height=300)

def run_small_group_generator():
    st.subheader("👥 Small Group Generator")
    passage = st.text_input("Which Bible passage are you studying?")
    if st.button("Create Guide") and passage:
        with st.spinner("Building your guide..."):
            guide = ask_gpt_conversation(f"Create a small group discussion guide for {passage} with 5 thoughtful questions and a key truth.")
            st.text_area("Your Group Guide", guide, height=400)

# ================================================================
# NEW LEARNING MODULE (DEFINITIVE, CORRECTED VERSION)
# ================================================================
def create_lesson_prompt(level_topic: str, lesson_number: int, user_learning_style: str, time_commitment: str) -> str:
    return f"""
You are an expert AI, Python coder, pastor, and theologian teacher. Your task is to generate a single, biblically sound Christian lesson for a learning app.

**Lesson Details:**
- **Level Topic:** "{level_topic}"
- **Lesson Number:** {lesson_number}
- **User Learning Style:** "{user_learning_style}"
- **Time Commitment:** "{time_commitment}"

**Output Format (Strict JSON):**
Your entire response MUST be a single JSON object, wrapped in triple backticks and 'json' specifier.

```json
{{
  "lesson_title": "A concise, engaging title for this lesson",
  "lesson_content_sections": [
    {{
      "type": "text",
      "content": "Paragraph 1 of the lesson content, biblically sound and engaging."
    }},
    {{
      "type": "knowledge_check",
      "question_type": "multiple_choice",
      "question": "What is the biblical definition of faith?",
      "options": ["A feeling of hope", "Trusting in human ability", "Confidence in what we hope for and assurance about what we do not see", "Blind belief"],
      "correct_answer": "Confidence in what we hope for and assurance about what we do not see",
      "biblical_reference": "Hebrews 11:1"
    }},
    {{
      "type": "text",
      "content": "Further lesson content, building on previous points."
    }},
    {{
      "type": "knowledge_check",
      "question_type": "true_false",
      "question": "Faith is primarily based on human reason.",
      "correct_answer": "False",
      "biblical_reference": "Romans 10:17"
    }}
  ],
  "summary_points": [
    "Key takeaway 1 from the lesson.",
    "Key takeaway 2 from the lesson."
  ]
}}
```
"""

def create_level_quiz_prompt(level_topic: str) -> str:
    return f"""
You are an expert AI, pastor, and theologian teacher. Your task is to create a 10-question final quiz for a Christian learning app, covering the level topic: "{level_topic}".
The quiz should include a mix of multiple choice, true/false, matching, and fill-in-the-blank questions.

Output Format (Strict JSON Array):
Your entire response MUST be a single JSON array of question objects.

```json
[
  {{
    "question_type": "multiple_choice",
    "question": "Which book details the exodus from Egypt?",
    "options": ["Genesis", "Exodus", "Leviticus", "Numbers"],
    "correct_answer": "Exodus",
    "biblical_reference": "Exodus 1:1"
  }},
  {{
    "question_type": "true_false",
    "question": "The greatest commandment is to love your neighbor.",
    "correct_answer": "False",
    "biblical_reference": "Matthew 22:36-40"
  }},
  {{
    "question_type": "fill_in_the_blank",
    "question": "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have ____________________.",
    "correct_answer": "everlasting life",
    "biblical_reference": "John 3:16"
  }},
  {{
    "question_type": "matching",
    "question": "Match the biblical figures with their roles:",
    "options": [
      {{"term": "Moses", "match": "Led Israelites out of Egypt"}},
      {{"term": "David", "match": "King of Israel"}},
      {{"term": "Paul", "match": "Apostle to the Gentiles"}}
    ],
    "correct_answer": {{
      "Moses": "Led Israelites out of Egypt",
      "David": "King of Israel",
      "Paul": "Apostle to the Gentiles"
    }},
    "biblical_reference": "Various"
  }}
]
```
"""

def display_knowledge_check_question(S):
    current_lesson_sections = S["levels"][S["current_level"]]["lessons"][S["current_lesson_index"]]["lesson_content_sections"]
    q = current_lesson_sections[S["current_section_index"]]

    st.markdown("---")
    st.markdown(f"#### ✅ Knowledge Check")
    st.markdown(f"**{q.get('question', 'Missing question text.')}**")

    user_answer = None
    input_key = f"kc_{S['current_level']}_{S['current_lesson_index']}_{S['current_section_index']}"

    if q.get('question_type') == 'multiple_choice':
        user_answer = st.radio("Select your answer:", q.get('options', []), key=input_key)
    elif q.get('question_type') == 'true_false':
        user_answer = st.radio("True or False?", ['True', 'False'], key=input_key)
    elif q.get('question_type') == 'fill_in_the_blank':
        user_answer = st.text_input("Fill in the blank:", key=input_key)

    if st.button("Submit Answer", key=f"submit_{input_key}"):
        is_correct = user_answer and str(user_answer).strip().lower() == str(q.get('correct_answer')).strip().lower()
        if is_correct:
            st.success("Correct! Moving on.")
            S["current_section_index"] += 1
            if "kc_answered_incorrectly" in S:
                del S["kc_answered_incorrectly"]
            st.rerun()
        else:
            S["kc_answered_incorrectly"] = True
            st.rerun()

    if S.get("kc_answered_incorrectly"):
        st.error(f"Not quite. The correct answer is: **{q.get('correct_answer')}**")
        st.info(f"See {q.get('biblical_reference')} for more context.")
        if st.button("Continue", key=f"continue_{input_key}"):
            del S["kc_answered_incorrectly"]
            S["current_section_index"] += 1
            st.rerun()

def run_level_quiz(S):
    level_data = S["levels"][S["current_level"]]
    quiz_questions = level_data.get("quiz_questions", [])
    q_index = S.get("current_question_index", 0)

    st.markdown("### 📝 Final Level Quiz")
    if not quiz_questions:
        st.warning("Quiz questions are not available.")
        return

    st.progress((q_index) / len(quiz_questions))
    st.markdown(f"**Score: {S.get('user_score', 0)}/{len(quiz_questions)}**")

    if q_index < len(quiz_questions):
        q = quiz_questions[q_index]
        st.markdown(f"**Question {q_index + 1}:** {q.get('question', '')}")

        user_answer = None
        q_key = f"quiz_{S['current_level']}_{q_index}"

        if q.get('question_type') == 'multiple_choice':
            user_answer = st.radio("Answer:", q.get('options', []), key=q_key)
        elif q.get('question_type') == 'true_false':
            user_answer = st.radio("Answer:", ["True", "False"], key=q_key)
        elif q.get('question_type') == 'fill_in_the_blank':
            user_answer = st.text_input("Answer:", key=q_key)

        if st.button("Submit Quiz Answer", key=f"submit_{q_key}"):
            if user_answer and str(user_answer).strip().lower() == str(q.get('correct_answer')).strip().lower():
                st.success("Correct!")
                S["user_score"] = S.get("user_score", 0) + 1
            else:
                st.error(f"Incorrect. The correct answer was: {q.get('correct_answer')}")
            S["current_question_index"] = q_index + 1
            st.rerun()
    else:
        st.success(f"### Quiz Completed! Final Score: {S.get('user_score', 0)}/{len(quiz_questions)}")
        if S.get('user_score', 0) >= len(quiz_questions) * 0.7:
            st.balloons()
            st.markdown(f"Congratulations! You passed Level {S['current_level'] + 1}!")
            if st.button("Next Level ▶️"):
                S["current_level"] += 1
                S["current_lesson_index"] = 0
                S["current_section_index"] = 0
                S["current_question_index"] = 0
                S["user_score"] = 0
                S["quiz_mode"] = False
                st.rerun()
        else:
            st.error("Please review the lessons and try the quiz again.")
            if st.button("Retake Quiz"):
                S["current_question_index"] = 0
                S["user_score"] = 0
                st.rerun()

def run_learn_module():
    st.subheader("📚 Learn Biblical Truths")
    if "learn_state" not in st.session_state:
        st.session_state.learn_state = {}
    S = st.session_state.learn_state

    if "levels" not in S:
        st.info("Welcome! Let's tailor your learning journey.")
        style = st.selectbox("Preferred learning style:", ["storytelling", "analytical", "practical application"])
        time = st.selectbox("Daily time commitment:", ["15 minutes", "30 minutes", "45 minutes"])
        if st.button("Start Learning Journey 🚀"):
            S.update({
                "levels": [
                    {"name": "Level 1: Foundations of Faith", "topic": "Faith and Grace"},
                    {"name": "Level 2: The Person of Christ", "topic": "Who Jesus Is"},
                    {"name": "Level 3: The Holy Spirit", "topic": "The Role of the Holy Spirit"}
                ],
                "current_level": 0,
                "current_lesson_index": 0,
                "current_section_index": 0,
                "user_learning_style": style,
                "time_commitment_per_day": time,
                "quiz_mode": False
            })
            st.rerun()
        return

    if S["current_level"] >= len(S["levels"]):
        st.success("🎉 You've completed all available levels!")
        return

    level_data = S["levels"][S["current_level"]]
    st.markdown(f"## {level_data['name']}")

    if S.get("quiz_mode"):
        run_level_quiz(S)
        return

    MAX_LESSONS = 5
    if S["current_lesson_index"] >= MAX_LESSONS:
        st.info("You've completed all lessons for this level.")
        if st.button("Start Final Quiz"):
            S["quiz_mode"] = True
            if "quiz_questions" not in level_data or not level_data["quiz_questions"]:
                with st.spinner("Generating quiz..."):
                    quiz_prompt = create_level_quiz_prompt(level_data["topic"])
                    quiz_resp = ask_gpt_conversation(quiz_prompt)
                    level_data["quiz_questions"] = extract_json_from_response(quiz_resp)
                    S["current_question_index"] = 0
                    S["user_score"] = 0
            st.rerun()
        return

    if "lessons" not in level_data:
        level_data["lessons"] = []
    if S["current_lesson_index"] >= len(level_data["lessons"]):
        with st.spinner("Generating your next lesson..."):
            lesson_prompt = create_lesson_prompt(level_data["topic"], S["current_lesson_index"] + 1, S["user_learning_style"], S["time_commitment_per_day"])
            lesson_resp = ask_gpt_conversation(lesson_prompt)
            lesson_data = extract_json_from_response(lesson_resp)
            if lesson_data:
                level_data["lessons"].append(lesson_data)
                S["current_section_index"] = 0
                st.rerun()
            else:
                st.error("Failed to generate lesson. Please try again.")
                return

    lesson = level_data["lessons"][S["current_lesson_index"]]
    sections = lesson.get("lesson_content_sections", [])
    st.markdown(f"### Lesson {S['current_lesson_index'] + 1}: {lesson.get('lesson_title', '')}")
    st.progress((S["current_section_index"]) / len(sections) if sections else 0)

    sec_index = S["current_section_index"]
    if sec_index < len(sections):
        section = sections[sec_index]
        if section.get("type") == "text":
            st.write(section.get("content"))
            if st.button("Continue", key=f"cont_{sec_index}"):
                S["current_section_index"] += 1
                st.rerun()
        elif section.get("type") == "knowledge_check":
            display_knowledge_check_question(S)
    else:
        st.success(f"Lesson {S['current_lesson_index'] + 1} complete!")
        if st.button("Next Lesson"):
            S["current_lesson_index"] += 1
            S["current_section_index"] = 0
            st.rerun()

# ================================================================
# MAIN UI
# ================================================================
st.title("✅ Bible GPT")

mode = st.sidebar.selectbox(
    "Choose a mode:",
    [
        "Learn Module", "Bible Lookup", "Chat with GPT", "Sermon Transcriber & Summarizer",
        "Practice Chat", "Verse of the Day", "Study Plan", "Faith Journal", "Prayer Starter",
        "Fast Devotional", "Small Group Generator", "Tailored Learning Path", "Bible Beta Mode",
        "Pixar Story Animation",
    ],
)

if mode == "Learn Module":
    run_learn_module()
elif mode == "Bible Lookup":
    run_bible_lookup()
elif mode == "Chat with GPT":
    run_chat_mode()
elif mode == "Sermon Transcriber & Summarizer":
    run_sermon_transcriber()
elif mode == "Practice Chat":
    run_practice_chat()
elif mode == "Verse of the Day":
    run_verse_of_the_day()
elif mode == "Study Plan":
    run_study_plan()
elif mode == "Faith Journal":
    run_faith_journal()
elif mode == "Prayer Starter":
    run_prayer_starter()
elif mode == "Fast Devotional":
    run_fast_devotional()
elif mode == "Small Group Generator":
    run_small_group_generator()
elif mode == "Tailored Learning Path":
    run_learning_path_mode()
elif mode == "Bible Beta Mode":
    run_bible_beta()
elif mode == "Pixar Story Animation":
    run_pixar_story_animation()
else:
    st.warning("This mode is under construction.")
