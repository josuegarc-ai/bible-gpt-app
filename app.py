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
    """
    Fetch a Bible passage from bible-api.com with URL encoding.
    Raises a clear error when the passage is actually not found.
    """
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
    """Stable, conservative GPT call for summaries and guidance."""
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
    """
    Extracts a JSON object or array from a string, supporting markdown code fences.
    """
    match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Found a JSON block in the response, but it was malformed.")
            return None
    
    # Fallback for responses that might not include the markdown block
    try:
        # Find the first '{' or '['
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
            # Find the last corresponding closing character
            end_index = response_text.rfind(end_char)
            if end_index > start_index:
                potential_json = response_text[start_index : end_index + 1]
                return json.loads(potential_json)
    except (json.JSONDecodeError, IndexError):
        pass  # If fallback fails, just proceed to the error

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
    """Generates the prompt for GPT to create a single lesson with embedded knowledge checks."""
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
