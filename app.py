# ✅ Bible GPT — Restored + Enhanced with New Features
# Amazing working option v2.3 — Now with conversational chat, AI insight fixes, enhanced practice, mixed learning path, and 'Bible Beta' (AI voices + highlights)

import os
import openai
import requests
import json
import re
import random
import urllib.parse
from datetime import datetime
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup  # ✅ Add this line
import streamlit as st
from pytube import YouTube
from urllib.error import HTTPError

##NEWLY ADDED
import tempfile
import subprocess
import urllib.request
from pytube import YouTube
import whisper

import imageio_ffmpeg
import os
import shutil

# ---- FFmpeg (no ffprobe) setup ----
import imageio_ffmpeg, os
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()           # path to ffmpeg binary bundled with imageio-ffmpeg
FFMPEG_DIR = os.path.dirname(FFMPEG_BIN)
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ✅ Get ffmpeg binary path from imageio
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")

# ✅ Ensure yt_dlp & other tools can find them
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["FFMPEG_LOCATION"] = ffmpeg_path
os.environ["FFPROBE_LOCATION"] = ffprobe_path

# ✅ Double-check they’re accessible
print("FFmpeg path:", ffmpeg_path)
print("FFprobe path:", ffprobe_path)
print("FFmpeg exists:", shutil.which("ffmpeg"))
print("FFprobe exists:", shutil.which("ffprobe"))

# ================= CONFIG =================
bible_api_base = "https://bible-api.com/"
valid_translations = ["web", "kjv", "asv", "bbe", "oeb-us"]

client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-4o"

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
        temperature=0.3,
        max_tokens=500,
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
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    base_url = "https://www.youtube.com/results?search_query="
    results = []

    for pastor in pastors:
        query = f"{pastor} sermon on {passage}"
        search_url = base_url + urllib.parse.quote(query)
        try:
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            scripts = soup.find_all("script")

            for script in scripts:
                if "var ytInitialData" in script.text:
                    start = script.text.find("var ytInitialData") + len("var ytInitialData = ")
                    end = script.text.find("};", start) + 1
                    json_text = script.text[start:end]
                    yt_data = json.loads(json_text)
                    videos = yt_data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents']
                    
                    # find the first video link
                    for item in videos:
                        if "videoRenderer" in item:
                            video_id = item['videoRenderer']['videoId']
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            results.append({"pastor": pastor, "url": video_url})
                            break
                    else:
                        results.append({"pastor": pastor, "url": "❌ No result"})
                    break
            else:
                results.append({"pastor": pastor, "url": "❌ Script not found"})
        except Exception as e:
            results.append({"pastor": pastor, "url": f"❌ Error: {str(e)}"})
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
    st.subheader("💬 Chat with GPT")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask a question or share a thought:")
    if st.button("Send") and user_input:
        if user_input.lower().strip() in ["exit", "quit", "end", "stop"]:
            full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
            guidance_prompt = (
                "You are a Christ-centered, pastoral guide. Based on the following conversation, write a short, encouraging reflection that gently sends the user off. "
                "Do not pray for them directly. Instead, guide them to seek God's presence, remind them of Jesus' love, and create a related prayer for for the user. "
                "Speak life, truth, and peace over them using Scripture and loving counsel. End with a hopeful, Spirit-led encouragement.\n\n"
                f"Conversation:\n{full_context}"
            )
            reflection = ask_gpt_conversation(guidance_prompt)
            st.markdown("**🙏 Final Encouragement:**")
            st.write(reflection)
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        history_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
        history_messages.insert(0, {
            "role": "system",
            "content": (
                "You are a pastoral, compassionate, honest, and expert biblical mentor with deep theological understanding. "
                "You speak with empathy and truth, offering thoughtful, wise, and scripturally grounded guidance to help people through all walks of life. "
                "You encourage people to seek God's presence first in prayer and reflection, and point them to Jesus as their ultimate source of peace, wisdom, and strength."
            )
        })
        response = client.chat.completions.create(
            model=model,
            messages=history_messages,
            temperature=0.4
        )
        reply = response.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    for msg in st.session_state.chat_history:
        speaker = "✝️ Bible GPT" if msg["role"] == "assistant" else "🧍 You"
        st.markdown(f"**{speaker}:** {msg['content']}")

def run_pixar_story_animation():
    st.subheader("🎥 Pixar-Studio Animated Bible Story")
    st.info("Generate a biblically accurate Pixar-style short film scene-by-scene based on Scripture.")

    book = st.text_input("📘 Bible book (e.g., Exodus):")
    chapter = st.text_input("🔢 Chapter (optional):")
    tone = st.selectbox("🎭 Pixar tone:", ["Adventurous", "Heartwarming", "Funny", "Epic", "All Ages"])
    theme = st.text_input("💡 Lesson or theme (e.g., faith, obedience):")

    if st.button("🎬 Generate Pixar Story") and book:
        reference = f"{book} {chapter}" if chapter else book
        story_prompt = (
            f"Turn the Bible story from {reference} into a Pixar-studio style film story for kids ages 4–10. "
            f"Tone: {tone}. Theme: {theme if theme else 'faith'}. "
            "Break it into exactly 5 cinematic scenes with 1–2 sentences each. "
            "Each scene should show a clear moment, visually imaginative, but true to the biblical setting. "
            "Output as numbered list (1. ..., 2. ..., etc.)"
        )
        response = ask_gpt_conversation(story_prompt)
        st.markdown("### 📚 Pixar-Style Bible Story Scenes")

        scenes = re.findall(r'\d+\.\s+(.*)', response)
        if not scenes:
            st.error("❌ Could not parse story scenes. Try different input.")
            return

        for idx, scene in enumerate(scenes):
            st.markdown(f"#### 🎬 Scene {idx + 1}")
            st.markdown(f"*{scene}*")

            # Enhanced prompt for DALLE with Pixar-studio film style and biblical setting
            prompt_enhancer = (
                f"You are a creative visual designer generating a concept prompt for a 3D animated Bible movie for children. "
                f"Turn the following scene into a rich, biblically accurate image prompt. "
                f"The style should be: 'In a 3D modern animation style, reminiscent of early 21st century design principles like Final Fantasy for children'. "
                f"Make it cinematic, colorful, soft-lit, and child-appropriate. Output only the final DALL·E-compatible prompt.\n\nScene: {scene}"
            )

            enhanced_prompt = ask_gpt_conversation(prompt_enhancer)

            try:
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    n=1
                )
                image_url = image_response.data[0].url
                st.image(image_url, caption=f"🎞️ Scene {idx + 1}: Pixar-style", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Error generating image: {e}")

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
            "proceed": False,
            "restart_flag": False,
            "last_answer_correct": False,
            "used_questions": set(),
            "used_phrases": set()
        }

    state = st.session_state.practice_state

    # Restart the quiz
    if state.get("restart_flag"):
        st.session_state.practice_state = {
            "questions": [],
            "current": 0,
            "score": 0,
            "book": "",
            "style": "",
            "level": "",
            "awaiting_next": False,
            "proceed": False,
            "restart_flag": False,
            "last_answer_correct": False,
            "used_questions": set(),
            "used_phrases": set()
        }
        st.rerun()

    if not state["questions"]:
        random_practice = st.checkbox("📖 Random questions from the Bible")
        book = ""
        if not random_practice:
            book = st.text_input("Enter Bible book:")
        style = st.selectbox("Choose question style:", ["multiple choice", "fill in the blank", "true or false", "mixed"])
        level = st.selectbox("Select your understanding level:", ["beginner", "intermediate", "advanced"])

        if st.button("Start Practice") and (random_practice or book):
            state["book"] = book
            state["style"] = style
            state["level"] = level
            num_questions = random.randint(7, 10)
            while len(state["questions"]) < num_questions:
                chosen_style = style if style != "mixed" else random.choice(["multiple choice", "fill in the blank", "true or false"])
                topic = book if book else "the Bible"

                if chosen_style == "true or false":
                    q_prompt = f"Generate a true or false Bible question from {topic} suitable for a {level} learner. Format as JSON with 'question', 'correct', and 'choices' as ['True', 'False']. The answer should be clearly either 'True' or 'False'. Avoid asking the same question in slightly different phrasing."
                else:
                    q_prompt = f"Generate a {chosen_style} Bible question from {topic} suitable for a {level} learner, with 1 correct answer and 3 incorrect ones. Format as JSON with 'question', 'correct', 'choices'. Avoid asking the same question in slightly different phrasing."

                response = ask_gpt_conversation(q_prompt)
                q_data = extract_json_from_response(response)

                if q_data:
                    normalized_question = q_data['question'].strip().lower()
                    if normalized_question not in state['used_questions']:
                        state['used_questions'].add(normalized_question)
                        phrase_key = " ".join(sorted(normalized_question.split()))
                        if phrase_key not in state['used_phrases']:
                            state['used_phrases'].add(phrase_key)
                            if chosen_style == "true or false":
                                q_data['choices'] = ["True", "False"]
                            else:
                                unique_choices = list(dict.fromkeys(q_data['choices']))
                                if q_data['correct'] not in unique_choices:
                                    unique_choices.append(q_data['correct'])
                                random.shuffle(unique_choices)
                                q_data['choices'] = unique_choices
                            state["questions"].append(q_data)

            st.rerun()

    elif state["current"] < len(state["questions"]):
        q_data = state["questions"][state["current"]]
        st.markdown(f"**Q{state['current'] + 1}: {q_data['question']}**")
        user_answer = st.radio("Choose:", q_data['choices'], key=f"q{state['current']}_choice")

        if not state.get("awaiting_next", False) and not state.get("proceed", False):
            if st.button("Submit Answer"):
                if user_answer.lower() == q_data['correct'].lower():
                    state["score"] += 1
                    st.success("✅ Correct!")
                    state["current"] += 1
                    st.rerun()
                else:
                    st.error(f"❌ Incorrect. Correct answer: {q_data['correct']}")
                    explain_prompt = f"You're a theological Bible teacher. Explain why '{q_data['correct']}' is correct for: '{q_data['question']}', and briefly clarify why the other options are incorrect, using Scripture-based reasoning."
                    explanation = ask_gpt_conversation(explain_prompt)
                    st.markdown("**📜 Teaching Moment:**")
                    st.write(explanation)
                    state["awaiting_next"] = True

        if state.get("awaiting_next"):
            if st.button("Next Question", key=f"next_{state['current']}"):
                state["current"] += 1
                state["awaiting_next"] = False
                st.rerun()

    else:
        st.markdown(f"**🌞 Final Score: {state['score']}/{len(state['questions'])}**")
        score_percent = (state['score'] / len(state['questions'])) * 100
        if score_percent >= 80:
            st.markdown('<script>confetti({ particleCount: 300, spread: 70, origin: { y: 0.6 }, scalar: 0.1 });</script>', unsafe_allow_html=True)
        if st.button("Restart Practice"):
            state["restart_flag"] = True
            st.rerun()

def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:")
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"journal_{ts}.txt"
        with open(filename, "w") as f:
            f.write(entry)
        st.success(f"Saved as {filename}.")
        if st.checkbox("Get spiritual insight from this entry"):
            insight = ask_gpt_conversation(f"Analyze this faith journal and offer spiritual insight and encouragement: {entry}")
            st.markdown("**💡 Insight:**")
            st.write(insight)

def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path")
    user_type = st.selectbox("User type:", ["child", "adult"])
    goal = st.text_input("Learning goal:")
    level = st.selectbox("Bible knowledge level:", ["beginner", "intermediate", "advanced"])
    styles = st.multiselect("Preferred learning styles:", ["storytelling", "questions", "memory games", "reflection", "devotional"])
    if st.button("Generate Path") and goal and styles:
        style_str = ", ".join(styles)
        prompt = f"Design a creative Bible learning path for a {user_type} with goal '{goal}', level '{level}', using these learning styles: {style_str}."
        result = ask_gpt_conversation(prompt)
        st.text_area("📘 Learning Path", result, height=500)

def run_bible_beta():
    st.subheader("📘 Bible Beta Mode")
    st.info("🧪 Experimental: Read and Listen to Bible page by page.")
    book = st.text_input("Book (e.g., John):")
    chapter = st.number_input("Chapter:", min_value=1, step=1)
    if st.button("Display Page"):
        verse = f"{book} {chapter}:1"
        try:
            text = fetch_bible_verse(verse)
            st.text_area("📖 Bible Text:", value=text, height=200)
            if st.button("🔊 Listen (AI Voice TBD)"):
                st.warning("Voice synthesis with celebrity tones coming soon.")
            if st.checkbox("✨ Highlight and Summarize"):
                highlight = st.text_area("Paste the section to summarize:")
                if highlight:
                    summary = ask_gpt_conversation(f"Summarize and reflect on this Bible passage: {highlight}")
                    st.markdown("**💬 Summary:**")
                    st.markdown(summary)
        except Exception as e:
            st.error(str(e))

##NEWLY ADDED

import yt_dlp
import tempfile
import os

def download_youtube_audio(url: str):
    """
    Download best available audio WITHOUT postprocessing.
    Uses a modern user agent and updated extractor args to bypass 403 errors.
    """
    tmp_dir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_DIR,
        "http_headers": {
            # ✅ Spoof a real desktop browser to avoid 403 Forbidden
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        # ✅ Use a stable extractor and retry logic
        "extractor_args": {"youtube": {"player_skip": ["configs"]}},
        "retries": 5,
        "fragment_retries": 10,
        
        "cookiefile": "cookies.txt"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)
            title = info.get("title", "Untitled Sermon")
            uploader = info.get("uploader", "Unknown")
        return file_path, uploader, title
    except Exception as e:
        raise Exception(f"❌ yt_dlp download error: {e}")

def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    st.info("Upload a sermon audio or paste a YouTube link. Max length: 15 minutes.")

    yt_link = st.text_input("📺 YouTube Link (15 mins max):")
    audio_file = st.file_uploader("🎙️ Or upload sermon audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])

    if st.button("⏺️ Transcribe & Summarize") and (yt_link or audio_file):
        with st.spinner("Transcribing... please wait."):
            audio_path = None
            preacher_name, sermon_title = "Unknown", "Untitled Sermon"

            try:
                # ✅ If YouTube link: check duration first WITHOUT downloading
                if yt_link:
                    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                        info = ydl.extract_info(yt_link, download=False)
                        duration = int(info.get("duration", 0) or 0)
                        if duration > 900:
                            raise Exception("❌ Sermon too long. Please limit to 15 minutes.")
                        preacher_name = info.get("uploader", "Unknown")
                        sermon_title = info.get("title", "Untitled Sermon")

                    # ✅ Now download bestaudio without postprocessing (no ffprobe)
                    audio_path, _, _ = download_youtube_audio(yt_link)

                elif audio_file:
                    # ✅ Save uploaded file as-is
                    suffix = os.path.splitext(audio_file.name)[1].lower() or ".wav"
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    temp_audio.write(audio_file.read())
                    temp_audio.flush()
                    audio_path = temp_audio.name

                # ✅ Transcribe using Whisper (reads webm/m4a via ffmpeg only — no ffprobe)
                whisper_model = whisper.load_model("base")  # or "small" if you need more accuracy
                try:
                    transcription = whisper_model.transcribe(audio_path)
                except Exception:
                    # 🔁 Fallback: convert to WAV with ffmpeg if container is odd
                    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    cmd = [FFMPEG_BIN, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    transcription = whisper_model.transcribe(wav_path)

                transcript_text = transcription["text"].strip()

                st.success("✅ Transcription complete.")
                st.markdown("### 📝 Transcript")
                st.text_area("Transcript", transcript_text, height=300)

                # ✅ Shorten for GPT if needed
                short_transcript = transcript_text[:1800].replace("\n", " ").replace("\r", " ")

                prompt = f"""
You are a sermon summarizer. From the transcript below, summarize the following:

- **Sermon Title**
- **Preacher Name**
- **Bible Verses Referenced**
- **Main Takeaways**
- **Reflection Questions**
- **Call to Action (if any)**

Preacher: {preacher_name}
Title: {sermon_title}

Transcript:
{short_transcript}
"""

                summary = ask_gpt_conversation(prompt)
                st.markdown("### 🧠 Sermon Summary")
                st.markdown(summary)

                # ✅ Save outputs
                os.makedirs("sermon_journal", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"sermon_journal/transcript_{timestamp}.txt", "w") as f:
                    f.write(transcript_text)
                with open(f"sermon_journal/summary_{timestamp}.txt", "w") as f:
                    f.write(summary)

                st.success("Saved transcript and summary to `sermon_journal/` folder.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# =============== MAIN UI ===============
mode = st.sidebar.selectbox("Choose a mode:", [
    "Bible Lookup", "Chat with GPT", "Practice Chat", "Verse of the Day",
    "Study Plan", "Faith Journal", "Prayer Starter", "Fast Devotional",
    "Small Group Generator", "Tailored Learning Path", "Bible Beta Mode",
    "Pixar Story Animation", "Sermon Transcriber & Summarizer"   # 👈 Make sure this is listed
])

if mode == "Bible Lookup":
    run_bible_lookup()
elif mode == "Chat with GPT":
    run_chat_mode()
elif mode == "Practice Chat":
    run_practice_chat()
elif mode == "Faith Journal":
    run_faith_journal()
elif mode == "Tailored Learning Path":
    run_learning_path_mode()
elif mode == "Bible Beta Mode":
    run_bible_beta()
elif mode == "Pixar Story Animation":
    run_pixar_story_animation() 
elif mode == "Sermon Transcriber & Summarizer":
    run_sermon_transcriber() # 👈 This line is required
else:
    st.warning("This mode is under construction.")
    


