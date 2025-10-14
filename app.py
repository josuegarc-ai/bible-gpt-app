# ================================================================
# ✅ Bible GPT — v3.2 (Complete & Fully Integrated)
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

os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ["FFMPEG_LOCATION"] = _FFMPEG_DIR

# ================================================================
# CONFIG
# ================================================================
BIBLE_API_BASE = "https://bible-api.com/"
VALID_TRANSLATIONS = ["web", "kjv", "asv", "bbe", "oeb-us"]
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# ================================================================
# UTILITIES & ORIGINAL APP MODES (Restored)
# ================================================================
def fetch_bible_verse(passage: str, translation: str = "web") -> str:
    """Fetches a Bible passage from bible-api.com."""
    if translation not in VALID_TRANSLATIONS:
        raise ValueError(f"Unsupported translation. Choose from: {VALID_TRANSLATIONS}")
    encoded_passage = urllib.parse.quote(passage.strip())
    url = f"{BIBLE_API_BASE}{encoded_passage}?translation={translation}"
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            raise Exception(f"Error {resp.status_code}: Unable to fetch passage. Check reference format.")
        data = resp.json()
        text = data.get("text", "").strip()
        if not text:
            raise Exception("Passage returned no text. Check book/chapter/verse.")
        return text
    except requests.RequestException as e:
        raise Exception(f"Network error: {e}")
    except Exception as e:
        raise Exception(f"API Error: {e}")

def ask_gpt_conversation(prompt: str) -> str:
    """Stable, conservative GPT call for summaries and guidance."""
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a biblical mentor and teacher. You explain Scripture clearly, compassionately, and apply it to modern life with spiritual insight."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content.strip()

def extract_json_from_response(response_text: str):
    """Legacy JSON extractor for simple objects."""
    try:
        json_text = re.search(r"\{.*\}", response_text, re.DOTALL).group(0)
        return json.loads(json_text)
    except Exception:
        return None

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
                        contents = yt_data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
                        for item in contents:
                            if "videoRenderer" in item:
                                video_id = item["videoRenderer"]["videoId"]
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                results.append({"pastor": pastor, "url": video_url})
                                found = True; break
                        break
            if not found:
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
                st.success(f"**{passage.strip()} ({translation.upper()})**\n\n{verse_text}")
                st.markdown("---")
                summary = ask_gpt_conversation(f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway.")
                st.markdown("**💡 AI Summary & Takeaway:**"); st.info(summary)
                cross = ask_gpt_conversation(f"List 2–3 cross-referenced Bible verses related to: '{verse_text}' and briefly explain their connection.")
                st.markdown("**🔗 Cross References:**"); st.markdown(cross)
                sermons = search_sermons_online(passage)
                st.markdown("**🎙️ Related Sermons:**")
                for item in sermons: st.markdown(f"- {item['pastor']}: {item['url']}")
            except Exception as e: st.error(str(e))

def run_chat_mode():
    st.subheader("💬 Chat with GPT")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    user_input = st.text_input("Ask a question or share a thought:")
    if st.button("Send") and user_input:
        if user_input.lower().strip() in ["exit", "quit", "end", "stop"]:
            full_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            reflection = ask_gpt_conversation(f"You are a Christ-centered, pastoral guide. Based on the following conversation, write a short, encouraging reflection and a related prayer for the user to use.\n\n{full_context}")
            st.markdown("**🙏 Final Encouragement:**"); st.write(reflection); return
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": "You are a loving, biblically grounded mentor."}] + st.session_state.chat_history
        r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.4)
        reply = r.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    for msg in st.session_state.chat_history:
        who = "✝️ Bible GPT" if msg["role"] == "assistant" else "🧍 You"
        st.markdown(f"**{who}:** {msg['content']}")

def run_pixar_story_animation():
    st.subheader("🎥 Pixar-Style Animated Bible Story")
    st.info("Generate a biblically accurate Pixar-style short film scene-by-scene based on Scripture.")
    book = st.text_input("📘 Bible book (e.g., Exodus):"); chapter = st.text_input("🔢 Chapter (optional):")
    tone = st.selectbox("🎭 Pixar tone:", ["Adventurous", "Heartwarming", "Funny", "Epic", "All Ages"])
    theme = st.text_input("💡 Lesson or theme (e.g., faith, obedience):")
    if st.button("🎬 Generate Pixar Story") and book:
        reference = f"{book} {chapter}".strip() if chapter else book
        story_prompt = f"Turn the Bible story from {reference} into a Pixar-studio style film story for kids ages 4–10. Tone: {tone}. Theme: {theme or 'faith'}. Break it into exactly 5 cinematic scenes with 1–2 sentences each. Output as a numbered list."
        response = ask_gpt_conversation(story_prompt)
        st.markdown("### 📚 Pixar-Style Bible Story Scenes")
        scenes = re.findall(r"\d+\.\s+(.*)", response) or [s.strip() for s in response.split("\n") if s.strip()]
        if not scenes: st.error("❌ Could not parse story scenes."); return
        for idx, scene in enumerate(scenes[:5], 1): st.markdown(f"#### 🎬 Scene {idx}\n*{scene}*")

def run_practice_chat():
    st.subheader("🤠 Practice Chat")
    if "practice_state" not in st.session_state:
        st.session_state.practice_state = {"questions": [], "current": 0, "score": 0, "awaiting_next": False, "restart_flag": False}
    S = st.session_state.practice_state
    if S.get("restart_flag"): st.session_state.practice_state = {}; st.rerun()
    if not S.get("questions"):
        st.info("Choose a topic to start a practice quiz.")
        book = st.text_input("Enter Bible book (e.g., Genesis):")
        style = st.selectbox("Choose question style:", ["multiple choice", "fill in the blank", "true or false", "mixed"])
        level = st.selectbox("Select your understanding level:", ["beginner", "intermediate", "advanced"])
        if st.button("Start Practice") and book:
            with st.spinner("Generating questions..."):
                S.update({"book": book, "style": style, "level": level, "questions": [], "used_questions": set()})
                num_questions = 7
                while len(S["questions"]) < num_questions:
                    chosen_style = style if style != "mixed" else random.choice(["multiple choice", "fill in the blank", "true or false"])
                    q_prompt = f"Generate a {chosen_style} Bible question from {book} for a {level} learner. Format as JSON with 'question','correct','choices'."
                    data = extract_json_from_response(ask_gpt_conversation(q_prompt))
                    if data and data.get("question") and data["question"].strip().lower() not in S["used_questions"]:
                        S["used_questions"].add(data["question"].strip().lower())
                        S["questions"].append(data)
            st.rerun()
    elif S["current"] < len(S.get("questions", [])):
        q = S["questions"][S["current"]]
        st.markdown(f"**Q{S['current'] + 1}: {q['question']}**")
        choices = q.get("choices", [])
        ans = st.radio("Choose:", choices, key=f"q_{S['current']}", index=None)
        if not S.get("awaiting_next"):
            if st.button("Submit Answer"):
                if ans and ans.lower() == str(q["correct"]).lower():
                    S["score"] += 1; S["current"] += 1; st.success("✅ Correct!"); st.rerun()
                else:
                    S["awaiting_next"] = True; st.error(f"❌ Incorrect. The correct answer was: {q['correct']}")
                    st.rerun()
        if S.get("awaiting_next"):
            if st.button("Next Question"):
                S["current"] += 1; S["awaiting_next"] = False; st.rerun()
    else:
        st.markdown(f"**🌞 Final Score: {S.get('score', 0)}/{len(S.get('questions',[]))}**")
        if st.button("Restart Practice"): S["restart_flag"] = True; st.rerun()

def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:")
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"journal_{ts}.txt"
        with open(filename, "w", encoding="utf-8") as f: f.write(entry)
        st.success(f"Saved as {filename}.")
        if st.checkbox("Get spiritual insight from this entry"):
            insight = ask_gpt_conversation(f"Analyze this faith journal and offer spiritual insight: {entry}")
            st.markdown("**💡 Insight:**"); st.write(insight)

def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path")
    st.warning("This mode has been upgraded! Please use the new 'Learn Module' for a more personalized experience.")

def run_bible_beta():
    st.subheader("📘 Bible Beta Mode")
    book = st.text_input("Book (e.g., John):"); chapter = st.number_input("Chapter:", min_value=1, step=1)
    if st.button("Display Page") and book:
        try:
            text = fetch_bible_verse(f"{book} {chapter}"); st.text_area("📖 Bible Text:", value=text, height=200)
            if st.checkbox("✨ Highlight and Summarize"):
                highlight = st.text_area("Paste the section to summarize:")
                if highlight:
                    summary = ask_gpt_conversation(f"Summarize and reflect on this passage: {highlight}")
                    st.markdown("**💬 Summary:**"); st.markdown(summary)
        except Exception as e: st.error(str(e))

def _convert_to_wav_if_needed(src_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file: wav_path = tmp_file.name
    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    try: subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e: raise Exception(f"ffmpeg failed: {e.stderr}")
    return wav_path

def download_youtube_audio(url: str) -> tuple[str, str, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file: output_path = temp_file.name
    ydl_opts = {"format": "bestaudio[ext=m4a]/bestaudio/best", "outtmpl": output_path, "ffmpeg_location": _FFMPEG_DIR, "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(url, download=True)
        title = info.get("title", "Untitled"); uploader = info.get("uploader", "Unknown")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0: raise Exception("Download failed.")
        return output_path, uploader, title
    except Exception as e: raise Exception(f"YouTube processing error: {e}")

def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    yt_link = st.text_input("📺 YouTube Link (≤ 15 mins):")
    audio_file = st.file_uploader("🎙️ Or upload audio (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    if st.button("⏺️ Transcribe & Summarize") and (yt_link or audio_file):
        with st.spinner("Processing..."):
            try:
                audio_path = None; preacher = "Unknown"; title = "Untitled"
                if yt_link: audio_path, preacher, title = download_youtube_audio(yt_link)
                elif audio_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_audio:
                        temp_audio.write(audio_file.getvalue()); audio_path = temp_audio.name
                model = whisper.load_model("base")
                try: transcription = model.transcribe(audio_path)
                except Exception: transcription = model.transcribe(_convert_to_wav_if_needed(audio_path))
                transcript_text = transcription.get("text", "").strip()
                if not transcript_text: raise Exception("Transcription was empty.")
                st.success("✅ Transcription Complete")
                st.text_area("📝 Transcript", transcript_text, height=250)
                summary_prompt = f"Summarize this sermon transcript by {preacher} titled '{title}':\n\n{transcript_text[:4000]}"
                summary = ask_gpt_conversation(summary_prompt)
                st.markdown("### 🧠 Sermon Summary"); st.info(summary)
            except Exception as e: st.error(f"An error occurred: {e}")
            finally:
                if audio_path and os.path.exists(audio_path): os.remove(audio_path)

def run_study_plan():
    st.subheader("📅 Personalized Bible Study Plan")
    goal = st.text_input("Study goal (e.g., 'Understand forgiveness'):")
    duration = st.slider("Plan duration (days):", 7, 60, 14)
    if st.button("Generate Study Plan") and goal:
        with st.spinner("✍️ Creating your plan..."):
            prompt = f"Create a {duration}-day Bible study plan on '{goal}'. For each day, provide a theme, 1-2 passages, a short summary, and a reflection question."
            plan = ask_gpt_conversation(prompt)
            st.text_area("📘 Your Study Plan", plan, height=600)

def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    if st.button("Get Today's Verse"):
        books = ["Psalms", "Proverbs", "John", "Romans", "Ephesians", "Philippians", "James"]
        try:
            book = random.choice(books)
            # A simple way to get a plausible chapter/verse
            ref_prompt = f"Give me a single, encouraging Bible verse reference from the book of {book} as a string, like 'John 3:16'."
            ref = ask_gpt_conversation(ref_prompt).strip()
            text = fetch_bible_verse(ref)
            st.success(f"**{ref}**\n\n{text}")
            reflection = ask_gpt_conversation(f"Offer a short, warm reflection on {ref}: '{text}'")
            st.markdown("**💬 Reflection:**"); st.write(reflection)
        except Exception as e: st.error(f"Could not fetch verse: {e}")

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    theme = st.text_input("Theme (e.g., gratitude, anxiety):")
    if st.button("Generate Prayer") and theme:
        prayer = ask_gpt_conversation(f"Write a short prayer starter on the theme of {theme}.")
        st.text_area("Your Prayer Starter", prayer, height=200)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    topic = st.text_input("Topic (e.g., hope, perseverance):")
    if st.button("Generate Devotional") and topic:
        devo = ask_gpt_conversation(f"Compose a 200-word devotional on {topic} with a key verse and a closing challenge.")
        st.text_area("Your Devotional", devo, height=350)

def run_small_group_generator():
    st.subheader("👥 Small Group Generator")
    passage = st.text_input("Passage for discussion (e.g., James 1:2-8):")
    if st.button("Create Guide") and passage:
        guide = ask_gpt_conversation(f"Create a small-group discussion guide for {passage} with an opener, 5 questions (observation, interpretation, application), and a closer.")
        st.text_area("Your Group Guide", guide, height=500)

# ================================================================
# LEARN MODULE (v3.2 - ARCHITECTURE FIX)
# ================================================================
def _learn_extract_json_any(response_text: str):
    if not response_text: return None
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
    json_str = match.group(1) if match else response_text
    try: return json.loads(json_str)
    except json.JSONDecodeError: return None

TOKENS_BY_TIME = {"15 minutes": 2000, "30 minutes": 3000, "45 minutes": 4000}

def ask_gpt_json(prompt: str, max_tokens: int = 4000):
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "system", "content": "You only return valid JSON."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.2, response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except Exception as e: st.error(f"Error calling OpenAI API: {e}"); return None

def _answers_match(user_answer, correct_answer, question_type="text") -> bool:
    if user_answer is None or correct_answer is None: return False
    user_ans_str = str(user_answer).strip(); correct_ans_str = str(correct_answer).strip()
    if question_type == 'multiple_choice': return user_ans_str.upper().startswith(correct_ans_str.upper())
    return user_ans_str.lower() == correct_ans_str.lower()

def summarize_lesson_content(lesson_data: dict) -> str:
    text_content = " ".join([sec['content'] for sec in lesson_data.get('lesson_content_sections', []) if sec.get('type') == 'text'])
    if not text_content: return "No textual content."
    prompt = f"Summarize this lesson into one concise sentence: {text_content[:2000]}"
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=150)
        return resp.choices[0].message.content.strip()
    except Exception: return lesson_data.get("lesson_title", "Summary unavailable.")

def create_full_learning_plan_prompt(form_data: dict) -> str:
    return f"""
    You are an expert theologian and curriculum designer. A user has provided this profile:
    - Topics: {form_data['topics']}
    - Knowledge: {form_data['knowledge_level']}
    - Goal: {", ".join(form_data['objectives'])}
    - Pacing: {form_data['pacing']}
    **TASK:** Design a complete, multi-level Bible study curriculum.
    1. Create a personalized title and a brief introduction.
    2. Determine the appropriate number of levels (e.g., "quick overview" = 2-3 levels; "deep dive" = 5-7).
    3. For each level, create a "level_name".
    4. For each level, break its topic into a sequence of 2 to 4 lessons, each with a "lesson_title" and "lesson_focus".
    **OUTPUT FORMAT:** You MUST return ONLY a single, valid JSON object.
    {{
      "plan_title": "...",
      "introduction": "...",
      "levels": [
        {{
          "level_name": "Level 1: ...",
          "lessons": [
            {{"lesson_title": "Lesson 1: ...", "lesson_focus": "..."}},
            {{"lesson_title": "Lesson 2: ...", "lesson_focus": "..."}}
          ]
        }}
      ]
    }}
    """

def create_lesson_prompt(lesson_focus: str, lesson_title: str, form_data: dict, previous_lesson_summary: str = None) -> str:
    length_instructions = {
        "15 minutes": "You MUST generate exactly 3 teaching sections (150-200 words each) and exactly 2 knowledge checks.",
        "30 minutes": "You MUST generate exactly 5 teaching sections (200-250 words each) and exactly 3 knowledge checks.",
        "45 minutes": "You MUST generate exactly 7 teaching sections (250-300 words each) and exactly 4 knowledge checks."
    }
    context_clause = f"This lesson must build upon the previous one, which covered: '{previous_lesson_summary}'." if previous_lesson_summary else ""
    return f"""
    You are an expert theologian creating a Bible lesson.
    - Lesson Title: "{lesson_title}"
    - Specific Focus: "{lesson_focus}"
    - User's Style: "{form_data['learning_style']}"
    **TASK:** Create lesson content following these rules precisely:
    - Requirements: {length_instructions.get(form_data['time_commitment'])}
    - Context: {context_clause}
    Return ONLY a valid JSON object.
    {{
      "lesson_title": "{lesson_title}", "lesson_content_sections": [], "summary_points": []
    }}
    """

def create_level_quiz_prompt(level_name: str, lesson_summaries: list) -> str:
    summaries_text = "\n".join(f"- {s}" for s in lesson_summaries)
    return f"Create a 10-question quiz for the level '{level_name}', based on these summaries:\n{summaries_text}\nReturn ONLY a JSON array of 10 question objects."

def display_knowledge_check_question(S):
    level_data = S["levels"][S["current_level"]]
    current_lesson = level_data["generated_lessons"][S["current_lesson_index"]]
    q = current_lesson["lesson_content_sections"][S["current_section_index"]]
    st.markdown("---"); st.markdown(f"#### ✅ Knowledge Check"); st.markdown(f"**{q.get('question', 'Missing question.')}**")
    user_answer = None; input_key = f"kc_{S['current_level']}_{S['current_lesson_index']}_{S['current_section_index']}"
    q_type = q.get('question_type')
    if q_type == 'multiple_choice': user_answer = st.radio("Select an answer:", q.get('options', []), key=input_key, index=None)
    elif q_type == 'true_false': user_answer = st.radio("True or False?", ['True', 'False'], key=input_key, index=None)
    elif q_type == 'fill_in_the_blank': user_answer = st.text_input("Fill in the blank:", key=input_key)
    if st.button("Submit Answer", key=f"submit_{input_key}"):
        if user_answer is None: st.warning("Please select an answer."); return
        if _answers_match(user_answer, q.get('correct_answer'), q_type):
            st.success("Correct!"); S["current_section_index"] += 1
            if "kc_answered_incorrectly" in S: del S["kc_answered_incorrectly"]
            st.rerun()
        else: S["kc_answered_incorrectly"] = True; st.rerun()
    if S.get("kc_answered_incorrectly"):
        st.error(f"Not quite. Correct answer: **{q.get('correct_answer')}**")
        if st.button("Continue", key=f"continue_{input_key}"): del S["kc_answered_incorrectly"]; S["current_section_index"] += 1; st.rerun()

def run_level_quiz(S):
    level_data = S["levels"][S["current_level"]]; quiz_questions = level_data.get("quiz_questions", [])
    q_index = S.get("current_question_index", 0)
    st.markdown("### 📝 Final Level Quiz")
    if not quiz_questions: st.warning("Quiz questions not available."); return
    st.progress((q_index) / len(quiz_questions)); st.markdown(f"**Score: {S.get('user_score', 0)}/{len(quiz_questions)}**")
    if q_index < len(quiz_questions):
        q = quiz_questions[q_index]; st.markdown(f"**Question {q_index + 1}:** {q.get('question', '')}")
        user_answer = None; q_key = f"quiz_{S['current_level']}_{q_index}"; q_type = q.get('question_type')
        if q_type == 'multiple_choice': user_answer = st.radio("Answer:", q.get('options', []), key=q_key, index=None)
        elif q_type == 'true_false': user_answer = st.radio("Answer:", ["True", "False"], key=q_key, index=None)
        elif q_type == 'fill_in_the_blank': user_answer = st.text_input("Answer:", key=q_key)
        if st.button("Submit Quiz Answer", key=f"submit_{q_key}"):
            if user_answer is None: st.warning("Please provide an answer."); return
            if _answers_match(user_answer, q.get('correct_answer'), q_type):
                st.success("Correct!"); S["user_score"] = S.get("user_score", 0) + 1
            else: st.error(f"Incorrect. Correct answer: {q.get('correct_answer')}")
            S["current_question_index"] = q_index + 1; st.rerun()
    else:
        st.success(f"### Quiz Completed! Final Score: {S.get('user_score', 0)}/{len(quiz_questions)}")
        if S.get('user_score', 0) >= len(quiz_questions) * 0.7:
            st.balloons(); st.markdown(f"Congratulations! You passed {level_data.get('level_name','this level')}!")
            if st.button("Go to Next Level ▶️"):
                S.update({"current_level": S["current_level"] + 1, "current_lesson_index": 0, "current_section_index": 0, "quiz_mode": False})
                st.rerun()
        else:
            st.error("Please review the lessons and try the quiz again.")
            if st.button("Retake Quiz"): S["current_question_index"] = 0; S["user_score"] = 0; st.rerun()

def run_learn_module_setup():
    st.info("Let's create a personalized learning plan based on your unique needs.")
    with st.form("user_profile_form"):
        form_data = {}
        form_data['topics'] = st.text_input("**What topics are on your heart to learn about?**", "Understanding grace, The life of David")
        form_data['knowledge_level'] = st.radio("**How would you describe your current Bible knowledge?**", ["Just starting out", "I know the main stories", "I'm comfortable with deeper concepts"], horizontal=True)
        form_data['objectives'] = st.multiselect("**What do you hope to achieve?**", ["Gain knowledge", "Find practical application", "Strengthen my faith"])
        form_data['pacing'] = st.select_slider("**How would you like to pace your learning?**", options=["A quick, high-level overview", "A steady, detailed study", "A deep, comprehensive dive"])
        form_data['learning_style'] = st.selectbox("**Preferred learning style:**", ["storytelling", "analytical", "practical"])
        form_data['time_commitment'] = st.selectbox("**How much time per lesson?**", ["15 minutes", "30 minutes", "45 minutes"])
        submitted = st.form_submit_button("🚀 Generate My Tailor-Made Plan")
    if submitted:
        if not form_data['topics'] or not form_data['objectives']: st.warning("Please fill out topics and objectives."); return
        with st.spinner("Our AI is designing your personalized curriculum..."):
            master_prompt = create_full_learning_plan_prompt(form_data)
            plan_resp = ask_gpt_json(master_prompt)
            plan_data = _learn_extract_json_any(plan_resp)
            if plan_data and "levels" in plan_data:
                st.session_state.learn_state.update({
                    "plan": plan_data, "levels": plan_data["levels"], "form_data": form_data, "current_level": 0,
                    "current_lesson_index": 0, "current_section_index": 0, "quiz_mode": False,
                    "current_question_index": 0, "user_score": 0,
                }); st.rerun()
            else: st.error("Failed to generate a valid learning plan. Please adjust your inputs.")

def run_learn_module():
    st.subheader("📚 Learn Module — Personalized Bible Learning")
    if "learn_state" not in st.session_state: st.session_state.learn_state = {}
    S = st.session_state.learn_state
    if "plan" not in S: run_learn_module_setup(); return

    st.title(S["plan"].get("plan_title", "Your Learning Journey")); st.write(S["plan"].get("introduction", ""))
    if S["current_level"] >= len(S["levels"]):
        st.success("🎉 You've completed your entire learning journey!"); st.balloons()
        if st.button("Start a New Journey"): del st.session_state.learn_state; st.rerun()
        return

    level_data = S["levels"][S["current_level"]]; st.markdown(f"--- \n## {level_data.get('level_name','Current Level')}")
    if S.get("quiz_mode"): run_level_quiz(S); return

    if "generated_lessons" not in level_data: level_data["generated_lessons"] = []
    num_lessons_in_level = len(level_data.get("lessons", []))
    if num_lessons_in_level == 0: st.error("This level has no lessons defined."); return

    if S["current_lesson_index"] >= len(level_data["generated_lessons"]):
        with st.spinner("Generating your next lesson..."):
            lesson_info = level_data["lessons"][S["current_lesson_index"]]
            prev_summary = level_data["generated_lessons"][S["current_lesson_index"] - 1].get("lesson_summary") if S["current_lesson_index"] > 0 else None
            lesson_prompt = create_lesson_prompt(lesson_title=lesson_info["lesson_title"], lesson_focus=lesson_info["lesson_focus"], form_data=S["form_data"], previous_lesson_summary=prev_summary)
            lesson_resp = ask_gpt_json(lesson_prompt, max_tokens=TOKENS_BY_TIME.get(S['form_data']['time_commitment']))
            lesson_content = _learn_extract_json_any(lesson_resp)
            if lesson_content:
                lesson_content["lesson_summary"] = summarize_lesson_content(lesson_content)
                level_data["generated_lessons"].append(lesson_content); S["current_section_index"] = 0; st.rerun()
            else: st.error("Failed to generate lesson content."); return

    current_lesson_content = level_data["generated_lessons"][S["current_lesson_index"]]
    st.markdown(f"### Lesson {S['current_lesson_index'] + 1} of {num_lessons_in_level}: {current_lesson_content.get('lesson_title', 'Untitled')}")
    lesson_sections = current_lesson_content.get("lesson_content_sections", [])
    if S["current_section_index"] < len(lesson_sections):
        section = lesson_sections[S["current_section_index"]]
        if section.get("type") == "text":
            st.markdown(section.get("content"))
            if st.button("Continue", key=f"cont_{S['current_level']}_{S['current_lesson_index']}_{S['current_section_index']}"): S["current_section_index"] += 1; st.rerun()
        elif section.get("type") == "knowledge_check": display_knowledge_check_question(S)
    else:
        st.success("Lesson Completed!"); st.markdown("**Key Takeaways:**")
        for point in current_lesson_content.get("summary_points", []): st.markdown(f"- {point}")
        if S["current_lesson_index"] < num_lessons_in_level - 1:
            if st.button("Go to Next Lesson"): S["current_lesson_index"] += 1; S["current_section_index"] = 0; st.rerun()
        else:
            st.info("You've completed all lessons for this level. Time for the final quiz!")
            if st.button("Start Level Quiz"):
                if "quiz_questions" not in level_data:
                    with st.spinner("Generating your level quiz..."):
                        all_summaries = [l.get("lesson_summary", "") for l in level_data["generated_lessons"]]
                        quiz_prompt = create_level_quiz_prompt(level_data.get("level_name"), all_summaries)
                        quiz_resp = ask_gpt_json(quiz_prompt, max_tokens=2500)
                        quiz_data = _learn_extract_json_any(quiz_resp)
                        if quiz_data: level_data["quiz_questions"] = quiz_data
                        else: st.error("Failed to generate quiz questions."); return
                S.update({"quiz_mode": True, "current_question_index": 0, "user_score": 0}); st.rerun()

# ================================================================
# MAIN UI
# ================================================================
st.set_page_config(page_title="Bible GPT", layout="wide")
st.title("✅ Bible GPT")

mode = st.sidebar.selectbox("Choose a mode:", [
    "Learn Module", "Bible Lookup", "Chat with GPT", "Sermon Transcriber & Summarizer",
    "Practice Chat", "Verse of the Day", "Study Plan", "Faith Journal", "Prayer Starter",
    "Fast Devotional", "Small Group Generator", "Tailored Learning Path", "Bible Beta Mode",
    "Pixar Story Animation",
])

mode_functions = {
    "Learn Module": run_learn_module, "Bible Lookup": run_bible_lookup, "Chat with GPT": run_chat_mode,
    "Sermon Transcriber & Summarizer": run_sermon_transcriber, "Practice Chat": run_practice_chat,
    "Verse of the Day": run_verse_of_the_day, "Study Plan": run_study_plan, "Faith Journal": run_faith_journal,
    "Prayer Starter": run_prayer_starter, "Fast Devotional": run_fast_devotional,
    "Small Group Generator": run_small_group_generator, "Tailored Learning Path": run_learning_path_mode,
    "Bible Beta Mode": run_bible_beta, "Pixar Story Animation": run_pixar_story_animation,
}

if mode in mode_functions:
    mode_functions[mode]()
else:
    st.warning("Selected mode not found.")
