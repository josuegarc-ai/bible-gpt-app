# ================================================================
# ✅ Bible GPT — v2.4 (Fully Fixed + All Modes Restored)
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
        max_tokens=600,
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
        json_text = re.search(r"\{.*\}", response_text, re.DOTALL).group(0)
        return json.loads(json_text)
    except Exception:
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
def _convert_to_wav_if_needed(src_path: str) -> str:
    """If Whisper has trouble with container, convert to 16k mono WAV using ffmpeg (no ffprobe)."""
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return wav_path


def download_youtube_audio(url: str) -> tuple[str, str, str]:
    """
    Download audio *without* postprocessing (so yt_dlp won't call ffprobe).
    Return (local_path, uploader, title).
    If repo has cookies.txt, yt_dlp will use it (helps with 403).
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
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
        # If you add a cookies.txt at repo root, yt_dlp will use it:
        **({"cookiefile": "cookies.txt"} if os.path.exists("cookies.txt") else {}),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled Sermon")
            uploader = info.get("uploader", "Unknown")
        return output_path, uploader, title
    except Exception as e:
        raise Exception(f"❌ yt_dlp download error: {e}")


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

                else:
                    # Save uploaded file as-is
                    suffix = os.path.splitext(audio_file.name)[1].lower() or ".wav"
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_audio.write(audio_file.read())
                    temp_audio.flush()
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
# MAIN UI
# ================================================================
mode = st.sidebar.selectbox(
    "Choose a mode:",
    [
        "Bible Lookup",
        "Chat with GPT",
        "Practice Chat",
        "Verse of the Day",
        "Study Plan",
        "Faith Journal",
        "Prayer Starter",
        "Fast Devotional",
        "Small Group Generator",
        "Tailored Learning Path",
        "Bible Beta Mode",
        "Pixar Story Animation",
        "Sermon Transcriber & Summarizer"
    ],
)

st.sidebar.write(f"DEBUG MODE SELECTED → [{mode}]")  # ✅ Add this line

if mode == "Bible Lookup":
    run_bible_lookup()
elif mode == "Chat with GPT":
    run_chat_mode()
elif mode == "Practice Chat":
    run_practice_chat()
elif mode == "Verse of the Day":
    run_verse_of_the_day()
elif mode == "Study Plan":   # ✅ This line is fine
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
elif mode == "Sermon Transcriber & Summarizer":
    run_sermon_transcriber()
else:
    st.warning("This mode is under construction.")
