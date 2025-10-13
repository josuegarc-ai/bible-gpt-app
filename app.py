# ================================================================
# ✅ Bible GPT — v3.0 (Final Syntax Fix)
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

# ==== AI / NLP ====
import openai
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

# (We intentionally do NOT set FFPROBE_LOCATION – we are avoiding ffprobe.)

# ================================================================
# CONFIG
# ================================================================
BIBLE_API_BASE = "https://bible-api.com/"
VALID_TRANSLATIONS = ["web", "kjv", "asv", "bbe", "oeb-us"]

# OpenAI
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
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

    # bible-api.com returns 404 for unknown passages
    if resp.status_code != 200:
        raise Exception(f"❌ Error {resp.status_code}: Unable to fetch passage. "
                        f"Check the reference formatting, e.g. 'John 3:16' or 'Psalm 23:1-3'.")

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
        max_tokens=2500, # Increased max_tokens for lessons/quizzes
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biblical mentor and teacher. You explain Scripture clearly, "
                    "compassionately, and apply it to modern life with spiritual insight."
                ),
            },
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
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from markdown block: {e}")
            return None

    first_char_pos = -1
    if '{' in response_text:
        first_char_pos = response_text.find('{')
    if '[' in response_text:
        bracket_pos = response_text.find('[')
        if first_char_pos == -1 or bracket_pos < first_char_pos:
            first_char_pos = bracket_pos

    if first_char_pos != -1:
        potential_json = response_text[first_char_pos:]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            last_bracket = '}' if potential_json.startswith('{') else ']'
            last_bracket_pos = potential_json.rfind(last_bracket)
            if last_bracket_pos != -1:
                try:
                    return json.loads(potential_json[:last_bracket_pos+1])
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON from fallback: {e}")
                    return None
    
    st.error("No valid JSON object or array found in the response.")
    return None

# ================================================================
# SERMON SEARCH (YouTube result links via HTML scrape)
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
            scripts = soup.find_all("script")
            found = False

            for script in scripts:
                txt = script.text or ""
                if "var ytInitialData" in txt:
                    start = txt.find("var ytInitialData") + len("var ytInitialData = ")
                    end = txt.find("};", start) + 1
                    if start > -1 and end > start:
                        json_text = txt[start:end]
                        yt_data = json.loads(json_text)
                        contents = yt_data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"][
                            "sectionListRenderer"
                        ]["contents"][0]["itemSectionRenderer"]["contents"]

                        for item in contents:
                            if "videoRenderer" in item:
                                video_id = item["videoRenderer"]["videoId"]
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                results.append({"pastor": pastor, "url": video_url})
                                found = True
                                break
                        break

            if not found:
                results.append({"pastor": pastor, "url": "❌ No result"})
        except Exception as e:
            results.append({"pastor": pastor, "url": f"❌ Error: {e}"})

    return results

# ================================================================
# BIBLE LOOKUP MODE
# ================================================================
def run_bible_lookup():
    st.subheader("📖 Bible Lookup")
    passage = st.text_input("Enter a Bible passage (e.g., John 3:16):")
    translation = st.selectbox("Choose translation:", VALID_TRANSLATIONS)

    if st.button("Fetch Verse") and passage:
        try:
            verse_text = fetch_bible_verse(passage, translation)
            st.success(verse_text)
            summary = ask_gpt_conversation(
                f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). "
                "Include a daily life takeaway."
            )
            st.markdown("**💡 AI Summary:**")
            st.info(summary)
            cross = ask_gpt_conversation(
                f"List 2–3 cross-referenced Bible verses related to: '{verse_text}' and explain their connection."
            )
            st.markdown("**🔗 Cross References:**")
            st.markdown(cross)
            sermons = search_sermons_online(passage)
            st.markdown("**🎙️ Related Sermons:**")
            for item in sermons:
                st.markdown(f"- {item['pastor']}: {item['url']}")
        except Exception as e:
            st.error(str(e))

# ================================================================
# CHAT MODE
# ================================================================
def run_chat_mode():
    st.subheader("💬 Chat with GPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question or share a thought:")

    if st.button("Send") and user_input:
        if user_input.lower().strip() in ["exit", "quit", "end", "stop"]:
            full_context = "\n".join(
                [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history]
            )
            reflection = ask_gpt_conversation(
                "You are a Christ-centered, pastoral guide. "
                "Based on the following conversation, write a short, encouraging reflection that gently sends the user off. "
                "Do not pray for them directly. Instead, guide them to seek God's presence, remind them of Jesus' love, "
                "and create a related prayer for the user.\n\n" + full_context
            )
            st.markdown("**🙏 Final Encouragement:**")
            st.write(reflection)
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a loving, biblically grounded mentor."}]
        messages += st.session_state.chat_history

        r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4)
        reply = r.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg in st.session_state.chat_history:
        who = "✝️ Bible GPT" if msg["role"] == "assistant" else "🧍 You"
        st.markdown(f"**{who}:** {msg['content']}")

# ================================================================
# PIXAR STORY ANIMATION (kept from your earlier app)
# ================================================================
def run_pixar_story_animation():
    st.subheader("🎥 Pixar-Studio Animated Bible Story")
    st.info("Generate a biblically accurate Pixar-style short film scene-by-scene based on Scripture.")

    book = st.text_input("📘 Bible book (e.g., Exodus):")
    chapter = st.text_input("🔢 Chapter (optional):")
    tone = st.selectbox("🎭 Pixar tone:", ["Adventurous", "Heartwarming", "Funny", "Epic", "All Ages"])
    theme = st.text_input("💡 Lesson or theme (e.g., faith, obedience):")

    if st.button("🎬 Generate Pixar Story") and book:
        reference = f"{book} {chapter}".strip() if chapter else book
        story_prompt = (
            f"Turn the Bible story from {reference} into a Pixar-studio style film story for kids ages 4–10. "
            f"Tone: {tone}. Theme: {theme or 'faith'}. "
            "Break it into exactly 5 cinematic scenes with 1–2 sentences each. "
            "Each scene should show a clear moment, visually imaginative, but true to the biblical setting. "
            "Output as a numbered list."
        )
        response = ask_gpt_conversation(story_prompt)
        st.markdown("### 📚 Pixar-Style Bible Story Scenes")
        scenes = re.findall(r"\d+\.\s+(.*)", response) or [s.strip() for s in response.split("\n") if s.strip()]
        if not scenes:
            st.error("❌ Could not parse story scenes. Try different input.")
            return

        for idx, scene in enumerate(scenes[:5], 1):
            st.markdown(f"#### 🎬 Scene {idx}")
            st.markdown(f"*{scene}*")

# ================================================================
# PRACTICE CHAT (Quiz)
# ================================================================
def run_practice_chat():
    st.subheader("🤠 Practice Chat")
    if "practice_state" not in st.session_state:
        st.session_state.practice_state = {"questions": [], "current": 0, "score": 0, "awaiting_next": False}
    
    S = st.session_state.practice_state

    if not S["questions"]:
        topic = st.text_input("Enter Bible book for quiz:", "John")
        if st.button("Start Practice"):
            with st.spinner("Generating questions..."):
                q_prompt = f"Generate 5 unique multiple-choice questions from the book of {topic}. Format as a JSON list of objects, each with 'question', 'choices' (a list), and 'correct' (a string)."
                response = ask_gpt_conversation(q_prompt)
                questions_data = extract_json_from_response(response)
                if questions_data and isinstance(questions_data, list):
                    S["questions"] = questions_data
                    S["current"] = 0
                    S["score"] = 0
                    st.rerun()
                else:
                    st.error("Failed to generate practice questions.")
    
    elif S["current"] < len(S["questions"]):
        q = S["questions"][S["current"]]
        st.markdown(f"**Question {S['current'] + 1}:** {q['question']}")
        
        # Ensure choices is a list before processing
        choices = q.get('choices', [])
        if not isinstance(choices, list):
            choices = [] # Default to empty list if format is wrong
        
        # Shuffle choices for display
        random.shuffle(choices)
        ans = st.radio("Choose:", choices, key=f"q_{S['current']}")

        if not S["awaiting_next"]:
            if st.button("Submit Answer", key=f"submit_{S['current']}"):
                if ans == q['correct']:
                    st.success("✅ Correct!")
                    S["score"] += 1
                    S["current"] += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer was: {q['correct']}")
                    S["awaiting_next"] = True
                st.rerun()
        else:
            if st.button("Next Question", key=f"next_{S['current']}"):
                S["awaiting_next"] = False
                S["current"] += 1
                st.rerun()
    else:
        st.markdown(f"**Quiz complete! Your score: {S['score']}/{len(S['questions'])}**")
        if st.button("Start New Quiz"):
            S["questions"] = []
            st.rerun()

# ================================================================
# FAITH JOURNAL
# ================================================================
def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:")
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_{ts}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(entry)
        st.success(f"Saved as {filename}.")
        if st.checkbox("Get spiritual insight from this entry"):
            insight = ask_gpt_conversation(
                f"Analyze this faith journal and offer spiritual insight and encouragement: {entry}"
            )
            st.markdown("**💡 Insight:**")
            st.write(insight)

# ================================================================
# TAILORED LEARNING PATH
# ================================================================
def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path")
    user_type = st.selectbox("User type:", ["child", "adult"])
    goal = st.text_input("Learning goal:")
    level = st.selectbox("Bible knowledge level:", ["beginner", "intermediate", "advanced"])
    styles = st.multiselect(
        "Preferred learning styles:",
        ["storytelling", "questions", "memory games", "reflection", "devotional"],
    )
    if st.button("Generate Path") and goal and styles:
        style_str = ", ".join(styles)
        prompt = (
            f"Design a creative Bible learning path for a {user_type} with goal '{goal}', level '{level}', "
            f"using these learning styles: {style_str}."
        )
        result = ask_gpt_conversation(prompt)
        st.text_area("📘 Learning Path", result, height=500)

# ================================================================
# BIBLE BETA
# ================================================================
def run_bible_beta():
    st.subheader("📘 Bible Beta Mode")
    st.info("🧪 Experimental: Read and Listen to Bible page by page.")
    book = st.text_input("Book (e.g., John):")
    chapter = st.number_input("Chapter:", min_value=1, step=1)
    if st.button("Display Page") and book:
        verse = f"{book} {chapter}" # Read whole chapter
        try:
            text = fetch_bible_verse(verse)
            st.text_area("📖 Bible Text:", value=text, height=200)
            if st.checkbox("✨ Highlight and Summarize"):
                highlight = st.text_area("Paste the section to summarize:")
                if highlight:
                    summary = ask_gpt_conversation(
                        f"Summarize and reflect on this Bible passage: {highlight}"
                    )
                    st.markdown("**💬 Summary:**")
                    st.markdown(summary)
        except Exception as e:
            st.error(str(e))

# ================================================================
# SERMON TRANSCRIBER & SUMMARIZER (YouTube or file upload)
# ================================================================
def _convert_to_wav_if_needed(src_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wav_path = tmp_file.name
    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"ffmpeg failed with exit code {e.returncode}: {e.stderr}")
    return wav_path

def download_youtube_audio(url: str) -> tuple[str, str, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        output_path = temp_file.name
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best", "outtmpl": output_path,
        "ffmpeg_location": os.environ.get("FFMPEG_LOCATION", _FFMPEG_DIR),
        "quiet": True, "retries": 3, "noprogress": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        **({"cookiefile": "cookies.txt"} if os.path.exists("cookies.txt") else {}),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled Sermon")
            uploader = info.get("uploader", "Unknown")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Audio download failed, resulting in an empty file.")
        return output_path, uploader, title
    except Exception as e:
        raise Exception(f"❌ Failed during YouTube audio processing: {e}")

def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    yt_link = st.text_input("📺 YouTube Link (≤ 15 mins):")
    if st.button("⏺️ Transcribe & Summarize") and yt_link:
        with st.spinner("Transcribing... please wait."):
            try:
                with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                    info = ydl.extract_info(yt_link, download=False)
                    if int(info.get("duration", 0)) > 900:
                        raise Exception("❌ Sermon too long. Please limit to 15 minutes.")
                audio_path, preacher, title = download_youtube_audio(yt_link)
                model = whisper.load_model("base")
                transcription = model.transcribe(audio_path)
                transcript_text = transcription.get("text", "").strip()
                if not transcript_text:
                    raise Exception("Transcription produced empty text.")
                st.success("✅ Transcription complete.")
                st.text_area("📝 Transcript", transcript_text, height=300)
                prompt = f"You are a sermon summarizer. Summarize key takeaways, Bible verses, and reflection questions from this transcript:\nPreacher: {preacher}\nTitle: {title}\nTranscript:\n{transcript_text[:2000]}"
                summary = ask_gpt_conversation(prompt)
                st.markdown("### 🧠 Sermon Summary")
                st.markdown(summary)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ================================================================
# SIMPLE STUDY PLAN (Mode 5)
# ================================================================
def run_study_plan():
    st.subheader("📅 Personalized Bible Study Plan")
    goal = st.text_input("Study goal (e.g., 'Grow in faith', 'Understand forgiveness'):")
    duration = st.slider("How many days do you want your plan to last?", 7, 60, 14)
    if st.button("Generate Study Plan") and goal:
        with st.spinner("✍️ Creating your personalized study plan..."):
            prompt = f"Create a detailed, day-by-day {duration}-day Bible study plan for the goal: '{goal}'. Include themes, passages, summaries, and reflection questions."
            plan = ask_gpt_conversation(prompt)
            st.text_area("📘 Your Study Plan", plan, height=600)

# ================================================================
# VERSE OF THE DAY, PRAYER STARTER, FAST DEVOTIONAL, SMALL GROUP
# ================================================================
def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    if "verse_of_day" not in st.session_state:
        st.session_state.verse_of_day = None
    if st.button("Get Today's Verse"):
        prompt = "Provide one inspiring Bible verse (e.g., John 3:16 KJV) and a 2-3 sentence reflection on it."
        st.session_state.verse_of_day = ask_gpt_conversation(prompt)
    if st.session_state.verse_of_day:
        st.success(st.session_state.verse_of_day)

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    theme = st.text_input("Theme (e.g., gratitude, anxiety, guidance):")
    if st.button("Generate Prayer") and theme:
        prayer = ask_gpt_conversation(f"Write a short, theologically faithful prayer starter on {theme}.")
        st.text_area("Prayer", prayer, height=300)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    topic = st.text_input("Topic (e.g., hope, perseverance):")
    if st.button("Generate Devotional") and topic:
        devo = ask_gpt_conversation(f"Compose a 150-word devotional on {topic} with a primary verse and a challenge.")
        st.text_area("Devotional", devo, height=350)

def run_small_group_generator():
    st.subheader("👥 Small Group Generator")
    passage = st.text_input("Passage for discussion (e.g., James 1:2-8):")
    if st.button("Create Guide") and passage:
        guide = ask_gpt_conversation(f"Create a small-group discussion guide for {passage} with 5 thoughtful questions and a key truth.")
        st.text_area("Group Guide", guide, height=500)

# ================================================================
# NEW LEARNING MODULE (SYNTAX CORRECTED)
# ================================================================

def create_lesson_prompt(level_topic: str, lesson_number: int, user_learning_style: str, time_commitment: str) -> str:
    """Generates the prompt for GPT to create a single lesson with embedded knowledge checks."""
    return f"""
You are an expert AI, Python coder, pastor, and theologian teacher. Your task is to generate a single, biblically sound Christian lesson for a learning app, tailored to the user's preferences.

**Lesson Details:**
- **Level Topic:** "{level_topic}"
- **Lesson Number:** {lesson_number}
- **User Learning Style:** "{user_learning_style}"
- **Time Commitment:** "{time_commitment}"

**Instructions for Lesson Generation:**
1.  **Lesson Content:**
    * Craft a comprehensive, biblically sound lesson.
    * Integrate relevant Bible verses naturally into the text.
    * Maintain a warm, pastoral, and encouraging tone.
2.  **Knowledge Checks:**
    * Embed 2-3 "knowledge checks" throughout the lesson content of varied types (multiple choice, true/false, fill-in-the-blank).

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
