# ================================================================
# ✅ Bible GPT — v2.5 (File Handling Fixed)
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
        max_tokens=2000, # Increased max_tokens for lessons/quizzes
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
    try:
        # Use regex to find the JSON string that starts with { and ends with }
        # This is more robust if there's leading/trailing text outside the JSON
        json_match = re.search(r"```json\n(\{.*?\})\n```", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            return json.loads(json_text)
        else:
            # Fallback for when the markdown isn't used
            json_text = re.search(r"(\{.*?\})", response_text, re.DOTALL)
            if json_text:
                return json.loads(json_text.group(0))
            return None
    except Exception as e:
        st.error(f"Error extracting or parsing JSON: {e}\nResponse text: {response_text[:500]}...")
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

            # Summary
            summary = ask_gpt_conversation(
                f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). "
                "Include a daily life takeaway."
            )
            st.markdown("**💡 AI Summary:**")
            st.info(summary)

            # Cross-references
            cross = ask_gpt_conversation(
                f"List 2–3 cross-referenced Bible verses related to: '{verse_text}' and explain their connection."
            )
            st.markdown("**🔗 Cross References:**")
            st.markdown(cross)

            # Related sermons
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
        st.session_state.practice_state = {
            "questions": [],
            "current": 0,
            "score": 0,
            "book": "",
            "style": "",
            "level": "",
            "awaiting_next": False,
            "used_questions": set(),
            "used_phrases": set(),
            "restart_flag": False,
        }
    S = st.session_state.practice_state

    if S.get("restart_flag"):
        st.session_state.practice_state = {
            "questions": [],
            "current": 0,
            "score": 0,
            "book": "",
            "style": "",
            "level": "",
            "awaiting_next": False,
            "used_questions": set(),
            "used_phrases": set(),
            "restart_flag": False,
        }
        st.rerun()

    if not S["questions"]:
        random_practice = st.checkbox("📖 Random questions from the Bible")
        book = "" if random_practice else st.text_input("Enter Bible book:")
        style = st.selectbox("Choose question style:", ["multiple choice", "fill in the blank", "true or false", "mixed"])
        level = st.selectbox("Select your understanding level:", ["beginner", "intermediate", "advanced"])

        if st.button("Start Practice") and (random_practice or book):
            S["book"] = book
            S["style"] = style
            S["level"] = level

            num_questions = random.randint(7, 10)
            while len(S["questions"]) < num_questions:
                chosen_style = style if style != "mixed" else random.choice(
                    ["multiple choice", "fill in the blank", "true or false"]
                )
                topic = book if book else "the Bible"

                if chosen_style == "true or false":
                    q_prompt = (
                        f"Generate a true or false Bible question from {topic} suitable for a {level} learner. "
                        "Format as JSON with 'question', 'correct', and 'choices' as ['True', 'False']."
                    )
                else:
                    q_prompt = (
                        f"Generate a {chosen_style} Bible question from {topic} suitable for a {level} learner, "
                        "with 1 correct answer and 3 incorrect ones. Format as JSON with 'question','correct','choices'."
                    )

                data = extract_json_from_response(ask_gpt_conversation(q_prompt))
                if not data:
                    continue

                norm = data["question"].strip().lower()
                if norm in S["used_questions"]:
                    continue
                S["used_questions"].add(norm)
                phrase_key = " ".join(sorted(norm.split()))
                if phrase_key in S["used_phrases"]:
                    continue
                S["used_phrases"].add(phrase_key)

                if chosen_style == "true or false":
                    data["choices"] = ["True", "False"]
                else:
                    # unique + shuffle
                    uniq = list(dict.fromkeys(data["choices"]))
                    if data["correct"] not in uniq:
                        uniq.append(data["correct"])
                    random.shuffle(uniq)
                    data["choices"] = uniq

                S["questions"].append(data)
            st.rerun()

    elif S["current"] < len(S["questions"]):
        q = S["questions"][S["current"]]
        st.markdown(f"**Q{S['current'] + 1}: {q['question']}**")
        ans = st.radio("Choose:", q["choices"], key=f"q{S['current']}_choice")

        if not S.get("awaiting_next", False):
            if st.button("Submit Answer"):
                if ans.lower() == q["correct"].lower():
                    S["score"] += 1
                    st.success("✅ Correct!")
                    S["current"] += 1
                    st.rerun()
                else:
                    st.error(f"❌ Incorrect. Correct answer: {q['correct']}")
                    explain = ask_gpt_conversation(
                        f"You're a theological Bible teacher. Explain why '{q['correct']}' is correct for: '{q['question']}', "
                        "and briefly clarify why the other options are incorrect, using Scripture-based reasoning."
                    )
                    st.markdown("**📜 Teaching Moment:**")
                    st.write(explain)
                    S["awaiting_next"] = True

        if S.get("awaiting_next"):
            if st.button("Next Question", key=f"next_{S['current']}"):
                S["current"] += 1
                S["awaiting_next"] = False
                st.rerun()

    else:
        st.markdown(f"**🌞 Final Score: {S['score']}/{len(S['questions'])}**")
        if st.button("Restart Practice"):
            S["restart_flag"] = True
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
        verse = f"{book} {chapter}:1"
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
# ✅ FIX APPLIED HERE
def _convert_to_wav_if_needed(src_path: str) -> str:
    """If Whisper has trouble with container, convert to 16k mono WAV using ffmpeg (no ffprobe)."""
    # Use a 'with' statement to ensure the temporary file is closed before ffmpeg uses it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wav_path = tmp_file.name

    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    try:
        # Capture output for better debugging if something goes wrong.
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # Raise a more informative error message.
        raise Exception(f"ffmpeg failed with exit code {e.returncode}: {e.stderr}")
        
    return wav_path

# ✅ FIX APPLIED HERE
def download_youtube_audio(url: str) -> tuple[str, str, str]:
    """
    Download audio *without* postprocessing (so yt_dlp won't call ffprobe).
    Return (local_path, uploader, title).
    If repo has cookies.txt, yt_dlp will use it (helps with 403).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        output_path = temp_file.name

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": os.environ.get("FFMPEG_LOCATION", _FFMPEG_DIR),
        "quiet": True,
        "retries": 3,
        "noprogress": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com/",
        },
        **({"cookiefile": "cookies.txt"} if os.path.exists("cookies.txt") else {}),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled Sermon")
            uploader = info.get("uploader", "Unknown")

        # ✅ FIX APPLIED HERE: Check if the downloaded file is empty.
        # If the file size is 0, the download failed silently.
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Audio download failed, resulting in an empty file. The video might be private, age-restricted, or unavailable.")

        return output_path, uploader, title
    except Exception as e:
        # This will now catch our custom error above or other yt-dlp errors.
        raise Exception(f"❌ Failed during YouTube audio processing: {e}")


def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    st.info("Upload a sermon audio or paste a YouTube link. Max length: 15 minutes (for testing).")

    yt_link = st.text_input("📺 YouTube Link (≤ 15 mins):")
    audio_file = st.file_uploader("🎙️ Or upload sermon audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])

    if st.button("⏺️ Transcribe & Summarize") and (yt_link or audio_file):
        with st.spinner("Transcribing... please wait."):
            try:
                preacher = "Unknown"
                title = "Untitled Sermon"
                audio_path = None # Initialize audio_path

                if yt_link:
                    # Check metadata & length WITHOUT downloading first
                    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True}) as ydl:
                        info = ydl.extract_info(yt_link, download=False)
                        duration = int(info.get("duration", 0) or 0)
                        if duration > 900:
                            raise Exception("❌ Sermon too long. Please limit to 15 minutes.")
                        preacher = info.get("uploader", "Unknown") or "Unknown"
                        title = info.get("title", "Untitled Sermon") or "Untitled Sermon"

                    # Download audio *without* postprocessing (no ffprobe)
                    audio_path, _, _ = download_youtube_audio(yt_link)

                elif audio_file:
                    # ✅ FIX APPLIED HERE
                    # Save uploaded file correctly using a 'with' statement.
                    suffix = os.path.splitext(audio_file.name)[1].lower() or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                        temp_audio.write(audio_file.getvalue())
                    audio_path = temp_audio.name

                # Transcribe with Whisper
                model = whisper.load_model("base")
                try:
                    transcription = model.transcribe(audio_path)
                except Exception:
                    # If container/codec odd, convert to WAV and retry
                    wav = _convert_to_wav_if_needed(audio_path)
                    transcription = model.transcribe(wav)

                transcript_text = transcription.get("text", "").strip()
                if not transcript_text:
                    raise Exception("Transcription produced empty text.")

                st.success("✅ Transcription complete.")
                st.markdown("### 📝 Transcript")
                st.text_area("Transcript", transcript_text, height=300)

                # Trim for GPT (safety)
                short = transcript_text[:1800].replace("\n", " ").replace("\r", " ")
                prompt = f"""
You are a sermon summarizer. From the transcript below, summarize the following:

- **Sermon Title**
- **Preacher Name**
- **Bible Verses Referenced**
- **Main Takeaways**
- **Reflection Questions**
- **Call to Action (if any)**

Preacher: {preacher}
Title: {title}

Transcript:
{short}
"""
                summary = ask_gpt_conversation(prompt)
                st.markdown("### 🧠 Sermon Summary")
                st.markdown(summary)

                # Save locally (journal)
                os.makedirs("sermon_journal", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"sermon_journal/transcript_{ts}.txt", "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                with open(f"sermon_journal/summary_{ts}.txt", "w", encoding="utf-8") as f:
                    f.write(summary)

                st.success("Saved transcript and summary to `sermon_journal/` folder.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ================================================================
# SIMPLE STUDY PLAN (Mode 5)
# ================================================================
def run_study_plan():
    st.subheader("📅 Personalized Bible Study Plan")

    goal = st.text_input("Study goal (e.g., 'Grow in faith', 'Understand forgiveness'):")
    duration = st.slider("How many days do you want your plan to last?", 7, 60, 14)
    focus = st.text_input("Focus area (optional):")
    level = st.selectbox("Knowledge level:", ["Beginner", "Intermediate", "Advanced"])
    include_reflections = st.checkbox("Include daily reflection questions?", True)

    if st.button("Generate Study Plan") and goal:
        with st.spinner("✍️ Creating your personalized study plan..."):
            prompt = f"""
You are a mature Bible mentor creating a detailed, Scripture-based daily study plan.

**Parameters:**
- Goal: {goal}
- Duration: {duration} days
- Focus area: {focus or 'General spiritual growth'}
- Knowledge level: {level}

**Instructions:**
Design a day-by-day Bible study plan.
For each day:
- Give a short **title or theme**
- Suggest **1–2 Bible passages to read**
- Write a **summary** (3–5 sentences) explaining the meaning and relevance
- Include a **cross-reference verse**
- Provide a **practical life application**
{"- Add a reflection question for journaling." if include_reflections else ""}
End with a brief closing paragraph encouraging the reader to stay consistent.

The tone should be pastoral, warm, and theologically sound.
Make sure it feels like a devotional guide.
"""
            try:
                plan = ask_gpt_conversation(prompt)
                st.markdown("### 📘 Your Study Plan")
                st.text_area("", plan, height=600)

                # Save locally
                os.makedirs("study_plans", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"study_plans/study_plan_{timestamp}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(plan)
                st.success(f"✅ Study plan saved to `{file_path}`.")
            except Exception as e:
                st.error(f"❌ Error generating study plan: {e}")

# ================================================================
# VERSE OF THE DAY, PRAYER STARTER, FAST DEVOTIONAL, SMALL GROUP
# (Lightweight but working versions to keep parity with your menu)
# ================================================================
def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    books = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
        "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
        "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
        "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
        "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
        "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
        "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
    ]
    try:
        book = random.choice(books)
        chapter = random.randint(1, 4)
        verse = random.randint(1, 20)
        ref = f"{book} {chapter}:{verse}"
        text = fetch_bible_verse(ref, "web")
        st.success(f"{ref} — {text}")
        reflection = ask_gpt_conversation(
            f"Offer a warm, practical reflection on this verse with 1 actionable takeaway: {text} ({ref})"
        )
        st.markdown("**💬 Reflection:**")
        st.write(reflection)
    except Exception as e:
        st.error(str(e))

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    theme = st.text_input("Theme (e.g., gratitude, anxiety, guidance):")
    if st.button("Generate Prayer") and theme:
        prayer = ask_gpt_conversation(
            f"Write a short, theologically faithful prayer starter on {theme}. Address God reverently; avoid clichés."
        )
        st.text_area("Prayer", prayer, height=300)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    topic = st.text_input("Topic (e.g., hope, perseverance):")
    if st.button("Generate Devotional") and topic:
        devo = ask_gpt_conversation(
            f"Compose a 150–200 word devotional on {topic} with one primary verse, 2 cross-refs, and 1 challenge for today."
        )
        st.text_area("Devotional", devo, height=350)

def run_small_group_generator():
    st.subheader("👥 Small Group Generator")
    passage = st.text_input("Passage for discussion (e.g., James 1:2-8):")
    if st.button("Create Guide") and passage:
        try:
            text = fetch_bible_verse(passage, "web")
        except Exception:
            text = passage
        guide = ask_gpt_conversation(
            f"Create a small-group discussion guide for this passage:\n{text}\n"
            "- 5 thoughtful questions (obs/interpretation/application)\n"
            "- One short opening and closing prompt\n"
            "- A key truth to remember"
        )
        st.text_area("Group Guide", guide, height=500)

# ================================================================
# NEW LEARNING MODULE
# ================================================================

def create_lesson_prompt(level_topic: str, lesson_number: int, user_learning_style: str, time_commitment: str) -> str:
    """Generates the prompt for GPT to create a single lesson with embedded knowledge checks."""
    return f"""
You are an expert AI, Python coder, pastor, and theologian teacher. Your task is to generate a single, biblically sound Christian lesson for a learning app, tailored to the user's preferences.

**Lesson Details:**
- **Level Topic:** "{level_topic}"
- **Lesson Number:** {lesson_number}
- **User Learning Style:** "{user_learning_style}" (e.g., storytelling, analytical, practical application, meditative)
- **Time Commitment:** "{time_commitment}" (this dictates the length and depth of the lesson and checks)

**Instructions for Lesson Generation:**
1.  **Lesson Content:**
    * Craft a comprehensive, biblically sound lesson that a theologian/pastor would teach their congregation, but simplified and engaging for an app user.
    * The content should directly relate to the '{level_topic}' for this level and be suitable for lesson number {lesson_number}.
    * Integrate relevant Bible verses naturally into the text.
    * Maintain a warm, pastoral, and encouraging tone.
    * Ensure the content is concise enough to fit within a '{time_commitment}' daily session. Adjust detail and sub-topics accordingly.
2.  **Knowledge Checks:**
    * Embed 2-3 "knowledge checks" throughout the lesson content.
    * Each knowledge check should consist of **one question** of a specific type.
    * Vary the question types: multiple choice, true/false, matching, and fill-in-the-blank. Ensure at least two different types are used across the 2-3 checks.
    * **For Multiple Choice:** Provide 1 correct answer and 3 incorrect but plausible options.
    * **For True/False:** Provide a statement and indicate if it's true or false.
    * **For Matching:** Provide 3-4 pairs of terms/concepts to match.
    * **For Fill-in-the-blank:** Provide a sentence with one key word or phrase missing, and the correct answer.

**Output Format (Strict JSON):**
Your entire response MUST be a single JSON object, wrapped in triple backticks and 'json' specifier, with the following structure:

```json
{{
  "lesson_title": "A concise, engaging title for this lesson (e.g., 'The Nature of Saving Faith')",
  "lesson_content_sections": [
    {{
      "type": "text",
      "content": "Paragraph 1 of the lesson content, biblically sound and engaging."
    }},
    {{
      "type": "text",
      "content": "Paragraph 2 of the lesson content, continuing the teaching."
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
      "question": "According to the Bible, faith is primarily based on human reason and intellect.",
      "correct_answer": "False",
      "biblical_reference": "Romans 10:17"
    }},
    {{
      "type": "text",
      "content": "More lesson content, leading to the next check or conclusion."
    }}
    // ... add more text and knowledge checks (2-3 checks per lesson, 5-10 sections total)
  ],
  "summary_points": [
    "Key takeaway 1 from the lesson.",
    "Key takeaway 2 from the lesson.",
    "Key takeaway 3 from the lesson."
  ]
}}
