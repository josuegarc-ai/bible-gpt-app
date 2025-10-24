# ================================================================
# ✅ Bible GPT — v2.9 (Personalized Plan Generator - Full & Integrated)
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
from thefuzz import fuzz # <-- REQUIRED FOR FUZZY STRING MATCHING

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
client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# Diagnostic Quiz Questions (with "I don't know")
DIAGNOSTIC_QUESTIONS = [
    {
        "question": "Who led the Israelites out of Egypt?",
        "options": ["Abraham", "Moses", "David", "Noah", "I don't know"],
        "correct": "Moses"
    },
    {
        "question": "Which disciple denied Jesus three times before the rooster crowed?",
        "options": ["Judas", "John", "Peter", "Thomas", "I don't know"],
        "correct": "Peter"
    },
    {
        "question": "What theological term refers to the study of 'last things' or end times?",
        "options": ["Soteriology", "Eschatology", "Christology", "Pneumatology", "I don't know"],
        "correct": "Eschatology"
    }
]

GENERAL_SYSTEM_PROMPT = """
You are Bible GPT, a mentor who speaks with the love and authority of **absolute biblical truth**.
Your primary goal is to guide the user toward righteousness and holiness by clearly stating what the Bible teaches.

- **Tone:** Be conversational, loving, and pastoral, but also **firm, direct, and unwavering in the truth**. 
- **Format:** Your answers must be **concise and conversational**. Do NOT use long, numbered lists. Give your answer in a direct paragraph.

- **Role:** You are not a neutral counselor. You are a Bible-based guide providing the truth as it is written.

- **Core Instruction - The Two Paths:**
You MUST first determine if the user's question is about a **Clear Biblical Command** or a **Disputable Matter**.

1.  **If it is a Clear Biblical Command:**
    - This includes any topic the Bible clearly *condemns* (e.g., idolatry, sexual immorality, occult/pagan practices) or *commands*.
    - You must state the biblical position as a **clear "yes" or "no" truth**. (e.g., "No, as followers of Christ, we are called to avoid that.")
    - Immediately back up your answer with 1-2 relevant scriptures.
    - **CRITICAL: You MUST NOT mention "personal conviction," "disputable matters," or "Romans 14" in this type of answer. These topics are not disputable, and mentioning them is confusing and weak.**

2.  **If it is a Disputable Matter (Romans 14):**
    - This *only* applies to topics where the Bible *does not* give a direct command (e.g., eating certain foods, observing certain Sabbath days).
    - **Only** for these topics may you explain that it is a matter of personal conviction before God, citing Romans 14.

- **Default:** When in doubt, default to the **Clear Biblical Command** path.
"""

THEOLOGICAL_SYSTEM_PROMPT = """
You are the "Theological Scholar" mode of Bible GPT.
Your goal is not just academic knowledge, but **deep spiritual conviction** built on doctrinal truth.
- **Tone & Format:** You are an expert theologian who speaks with authority. Be **concise, direct, and authoritative**. Your response should be a sharp, insightful paragraph, not a long academic list.
- **Depth:** Provide deep, analytical, and comprehensive theological answers. Cite theological concepts, historical context, and original languages where appropriate.
- **Tools:** You have a web search tool. Use it *any time* the user asks for connections between prophecy, scripture, and **current events**.

- **Core Instruction - The Two Paths (Scholarly Application):**
You MUST apply this core logic.

1.  **If it is a Clear Biblical Command:**
    - This includes any topic the Bible clearly *condemns* (e.g., idolatry, occult practices).
    - You must state the biblical position as **doctrinal truth**. Explain *why* it is theologically non-negotiable.
    - (Example: For Halloween, identify its pagan origins (Samhain) as a form of occultism, which the Bible clearly condemns. This makes it a matter of **truth** (2 Cor 6:14), not "adiaphora.")
    - **CRITICAL: You MUST NOT mention "disputable matters" (adiaphora) or "Romans 14" when discussing a topic of clear biblical condemnation. These concepts are mutually exclusive.**

2.  **If it is a Disputable Matter (Adiaphora):**
    - This *only* applies to topics where the Bible *does not* give a direct command.
    - You must identify these topics as "adiaphora" (things indifferent) and explain the principle of Christian liberty based on Romans 14.
"""

# ================================================================
# UTILITIES
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
        # Prioritize JSON within markdown code blocks
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
             # Fallback to finding the first curly brace object
            json_text_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_text_match: return None
            json_text = json_text_match.group(0)
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Error extracting JSON: {e}") # Added error logging
        return None

# ================================================================
# CHAT UTILITIES
# ================================================================

def load_chat_history() -> list:
    """Loads chat history from a local JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty if file is corrupt
    return []

def save_chat_history(history: list):
    """Saves chat history to a local JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")
        
def get_chat_messages(history: list, max_turns: int = 20) -> list:
    """Manages chat history to prevent token overflow by summarizing old messages."""
    if len(history) <= max_turns:
        return history # Return full history if it's short

    # History is too long, summarize the oldest part
    # Keep the most recent 10 messages (5 turns)
    messages_to_keep = history[-10:]
    messages_to_summarize = history[:-10]
    
    # Create a text blob of the old chat
    old_chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
    
    # Use your existing function to summarize
    try:
        summary_prompt = f"Concisely summarize the key points of this conversation in one paragraph: {old_chat_text}"
        summary = ask_gpt_conversation(summary_prompt) # This is your existing function
    except Exception as e:
        st.warning(f"Could not summarize history: {e}")
        summary = "Summary of prior conversation is unavailable."

    # Return a new history object
    return [
        {"role": "system", "content": f"[Prior Conversation Summary]: {summary}"},
        *messages_to_keep
    ]

def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo."""
    st.caption(f"🔎 Searching the web for: '{query}'") # Show the user it's searching
    try:
        with DDGS() as ddgs:
            # We will format the results as a clean string for the AI
            results = [r for r in ddgs.text(query, max_results=5)]
            if not results:
                return "No relevant web results found."
            
            # Format this for the AI to read
            formatted_results = "\n".join([
                f"- Snippet: {r['body']}\n  Source: {r['href']}" 
                for r in results
            ])
            return f"Here are the web search results:\n{formatted_results}"
            
    except Exception as e:
        return f"Search failed. Error: {str(e)}"
        
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
                        # Added checks for key existence
                        contents = yt_data.get("contents",{}).get("twoColumnSearchResultsRenderer",{}).get("primaryContents",{}).get("sectionListRenderer",{}).get("contents",[{}])[0].get("itemSectionRenderer",{}).get("contents",[])
                        for item in contents:
                            if "videoRenderer" in item:
                                video_id = item["videoRenderer"]["videoId"]
                                video_url = f"https://www.youtube.com/watch?v={video_id}"
                                results.append({"pastor": pastor, "url": video_url})
                                found = True
                                break # Found first result for this pastor
                        if found: break # Move to next pastor
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
        with st.spinner("Fetching and analyzing..."):
            try:
                verse_text = fetch_bible_verse(passage, translation)
                st.success(f"**{passage.strip()} ({translation.upper()})**\n\n{verse_text}")
                st.markdown("---")

                summary = ask_gpt_conversation(f"Summarize and explain this Bible verse clearly: '{verse_text}' ({passage}). Include a daily life takeaway.")
                st.markdown("**💡 AI Summary & Takeaway:**")
                st.info(summary)

                cross = ask_gpt_conversation(f"List 2–3 cross-referenced Bible verses related to: '{verse_text}' and briefly explain their connection.")
                st.markdown("**🔗 Cross References:**")
                st.markdown(cross)

                sermons = search_sermons_online(passage)
                st.markdown("**🎙️ Related Sermons:**")
                for item in sermons:
                    st.markdown(f"- {item['pastor']}: {item['url']}")
            except Exception as e:
                st.error(str(e))

# ================================================================
# CHAT MODE (FINAL VERSION - STATEFUL, CONVICTING, & WEB-ENABLED)
# ================================================================
def run_chat_mode():
    st.subheader("💬 Chat with GPT")
    
    is_theological_mode = st.toggle(
        "Enable Deep Theological Chat", 
        value=False,
        help="Toggle on for in-depth, scholarly answers with web search for current events."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    st.markdown("---")
    chat_container = st.container(height=400, border=False)
    with chat_container:
        if not st.session_state.chat_history:
             st.caption("Your conversation will appear here. Your chat history is saved automatically.")
        
        for msg in st.session_state.chat_history:
            # Don't show the tool call/result messages to the user, only human/ai
            if msg["role"] in ["user", "assistant"]:
                who = "✝️ Bible GPT" if msg["role"] == "assistant" else "🧍 You"
                st.markdown(f"**{who}:** {msg['content']}")
    st.markdown("---")
    
    user_input = st.text_input("Ask a question or share a thought:")
    
    if st.button("Send", type="primary") and user_input:
        
        if user_input.lower().strip() in ["exit", "quit", "end", "stop"]:
            st.info("Conversation ended. Your history is saved.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun() # Re-run to show the user's message immediately

    # --- This block handles processing the chat after the user message is added ---
    # Check if the last message was from the user, meaning AI needs to respond
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Thinking..."):
            
            # 1. Select prompt and tools
            system_prompt = THEOLOGICAL_SYSTEM_PROMPT if is_theological_mode else GENERAL_SYSTEM_PROMPT
            
            # Only give the web search tool to the "Theological" mode
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Searches the internet for current events, news, or topics. Use this to connect prophecy or biblical topics to the modern world.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string", "description": "The search query"}},
                            "required": ["query"],
                        },
                    },
                }
            ] if is_theological_mode else None

            # 2. Get managed message history
            messages_for_api = get_chat_messages(st.session_state.chat_history)
            final_messages = [{"role": "system", "content": system_prompt}] + messages_for_api
            
            try:
                # 3. Call the AI
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=final_messages,
                    temperature=0.3, # Lower temp for more direct, factual answers
                    tools=tools
                )
                response_message = response.choices[0].message

                # 4. Check if AI wants to use a tool (web search)
                if response_message.tool_calls:
                    st.session_state.chat_history.append(response_message) # Save the AI's tool request
                    
                    # --- This is the new Tool-Calling Loop ---
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        if function_name == "web_search":
                            # Get the query from the AI
                            function_args = json.loads(tool_call.function.arguments)
                            query = function_args.get("query")
                            
                            # Call our Python web_search function
                            function_response = web_search(query=query)
                            
                            # Send the search results back to the AI
                            st.session_state.chat_history.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response,
                                }
                            )
                    
                    # 5. Call AI *AGAIN* with the search results
                    # This lets the AI form a final answer
                    second_response = client.chat.completions.create(
                        model=MODEL,
                        messages=st.session_state.chat_history, # Send the *full* history including tool results
                    )
                    final_reply = second_response.choices[0].message.content.strip()
                    st.session_state.chat_history.append({"role": "assistant", "content": final_reply})

                else:
                    # 6. No tool was needed, just a direct answer
                    final_reply = response_message.content.strip()
                    st.session_state.chat_history.append({"role": "assistant", "content": final_reply})

                # 7. Save and refresh
                save_chat_history(st.session_state.chat_history)
                st.rerun()

            except Exception as e:
                st.error(f"Error communicating with AI: {e}")
                st.session_state.chat_history.pop() # Remove the user's message if it failed
                save_chat_history(st.session_state.chat_history)
# ================================================================
# PIXAR STORY ANIMATION
# ================================================================
def run_pixar_story_animation():
    st.subheader("🎥 Pixar-Style Animated Bible Story")
    st.info("Generate a biblically accurate Pixar-style short film scene-by-scene based on Scripture.")
    book = st.text_input("📘 Bible book (e.g., Exodus):")
    chapter = st.text_input("🔢 Chapter (optional):")
    tone = st.selectbox("🎭 Pixar tone:", ["Adventurous", "Heartwarming", "Funny", "Epic", "All Ages"])
    theme = st.text_input("💡 Lesson or theme (e.g., faith, obedience):")
    if st.button("🎬 Generate Pixar Story") and book:
        reference = f"{book} {chapter}".strip() if chapter else book
        story_prompt = f"Turn the Bible story from {reference} into a Pixar-studio style film story for kids ages 4–10. Tone: {tone}. Theme: {theme or 'faith'}. Break it into exactly 5 cinematic scenes with 1–2 sentences each. Each scene should be visually imaginative but true to the biblical setting. Output as a numbered list."
        response = ask_gpt_conversation(story_prompt)
        st.markdown("### 📚 Pixar-Style Bible Story Scenes")
        # Improved parsing for numbered lists or simple newlines
        scenes = re.findall(r"^\d+\.\s*(.*)", response, re.MULTILINE) or [s.strip() for s in response.split("\n") if s.strip() and not s.strip().isdigit()]
        if not scenes:
            st.error("❌ Could not parse story scenes. Try different input.")
            return
        for idx, scene in enumerate(scenes[:5], 1): # Limit to 5 scenes
            st.markdown(f"#### 🎬 Scene {idx}\n*{scene}*")

# ================================================================
# PRACTICE CHAT (Quiz) - UPDATED FOR FUZZY MATCHING AND EXPLANATION
# ================================================================
def run_practice_chat():
    st.subheader("🤠 Practice Chat")
    if "practice_state" not in st.session_state:
        st.session_state.practice_state = {
            "questions": [], "current": 0, "score": 0, "book": "", "style": "", "level": "",
            "awaiting_next": False, "used_questions": set(), "used_phrases": set(), "restart_flag": False,
        }
    S = st.session_state.practice_state

    if S.get("restart_flag"):
        st.session_state.practice_state = {
            "questions": [], "current": 0, "score": 0, "book": "", "style": "", "level": "",
            "awaiting_next": False, "used_questions": set(), "used_phrases": set(), "restart_flag": False,
        }
        st.rerun()

    if not S["questions"]:
        random_practice = st.checkbox("📖 Random questions from the Bible")
        book = "" if random_practice else st.text_input("Enter Bible book:")
        style = st.selectbox("Choose question style:", ["multiple choice", "fill in the blank", "true or false", "mixed"])
        level = st.selectbox("Select your understanding level:", ["beginner", "intermediate", "advanced"])

        if st.button("Start Practice") and (random_practice or book):
            S["book"] = book; S["style"] = style; S["level"] = level
            num_questions = random.randint(7, 10)
            with st.spinner("Generating practice questions..."): # Added spinner
                while len(S["questions"]) < num_questions:
                    chosen_style = style if style != "mixed" else random.choice(["multiple choice", "fill in the blank", "true or false"])
                    topic = book if book else "the Bible"
                    q_prompt = (
                        f"Generate a {chosen_style} Bible question from {topic} suitable for a {level} learner. "
                        f"Format as JSON with 'question', 'correct', 'choices' (list of strings), and 'question_type' ('multiple_choice', 'fill_in_the_blank', or 'true_false'). " # Added question_type
                        f"{'Choices for true/false should be [\"True\", \"False\"].' if chosen_style == 'true or false' else 'For multiple choice, include 1 correct and 3 incorrect options.'}"
                    )
                    data = extract_json_from_response(ask_gpt_conversation(q_prompt)) # Using more robust extractor
                    if not data or 'question' not in data or 'correct' not in data: continue # Basic validation
                    
                    # Add question_type if missing (shouldn't happen with updated prompt)
                    if 'question_type' not in data: data['question_type'] = chosen_style

                    norm = data["question"].strip().lower()
                    if norm in S["used_questions"]: continue
                    S["used_questions"].add(norm)
                    
                    # Deduplicate options and ensure correct is present for MC/FITB
                    if data['question_type'] != 'true_false':
                         if 'choices' not in data: data['choices'] = []
                         # Ensure choices is a list
                         if not isinstance(data['choices'], list): data['choices'] = []
                         # Deduplicate while preserving order (important for potential distractors)
                         seen = set()
                         uniq = [x for x in data['choices'] if not (x in seen or seen.add(x))]
                         # Add correct answer if not already in choices
                         if data['correct'] not in seen:
                             uniq.append(data['correct'])
                         random.shuffle(uniq)
                         data['choices'] = uniq
                    else:
                        data['choices'] = ["True", "False"] # Standardize TF choices

                    S["questions"].append(data)
            if not S["questions"]: # Handle case where generation failed
                 st.error("Failed to generate questions. Please try again.")
            else:
                 st.rerun()

    elif S["current"] < len(S["questions"]):
        q = S["questions"][S["current"]]
        st.markdown(f"**Q{S['current'] + 1}: {q['question']}**")
        
        q_type = q.get("question_type", "multiple_choice") # Default if missing
        ans = None
        if q_type == 'multiple_choice':
             ans = st.radio("Choose:", q.get("choices", []), key=f"q{S['current']}_choice", index=None)
        elif q_type == 'true_false':
             ans = st.radio("Choose:", ["True", "False"], key=f"q{S['current']}_choice", index=None)
        elif q_type == 'fill_in_the_blank':
             ans = st.text_input("Fill in the blank:", key=f"q{S['current']}_choice")

        if not S.get("awaiting_next", False):
            if st.button("Submit Answer"):
                 if ans is None and (q_type == 'multiple_choice' or q_type == 'true_false'):
                      st.warning("Please select an answer.")
                 elif ans is not None:
                    if _answers_match(ans, q["correct"], q_type): # Using fuzzy match function
                        S["score"] += 1; st.success("✅ Correct!"); S["current"] += 1; st.rerun()
                    else:
                        st.error(f"❌ Incorrect. Correct answer: **{q['correct']}**"); # Made correct answer bold
                        # Updated explanation prompt to include the incorrect answer
                        explain_prompt = (
                            f"You're a theological Bible teacher. A student answered '{ans}' to the question: '{q['question']}'. "
                            f"The correct answer is '{q['correct']}'. Explain briefly why their answer '{ans}' was incorrect and why '{q['correct']}' is the right one, using Scripture-based reasoning if possible."
                        )
                        explanation = ask_gpt_conversation(explain_prompt)
                        st.markdown("**📜 Teaching Moment:**"); st.write(explanation); S["awaiting_next"] = True
                        st.rerun() # Rerun to show teaching moment and Next button
        
        # Display Next button only after an incorrect answer's explanation is shown
        if S.get("awaiting_next"):
            if st.button("Next Question", key=f"next_{S['current']}"):
                S["awaiting_next"] = False # Reset flag
                S["current"] += 1
                st.rerun()
    else:
        st.markdown(f"**🌞 Final Score: {S['score']}/{len(S['questions'])}**")
        if st.button("Restart Practice"): S["restart_flag"] = True; st.rerun()

# ================================================================
# FAITH JOURNAL
# ================================================================
def run_faith_journal():
    st.subheader("📝 Faith Journal")
    entry = st.text_area("Write your thoughts, prayers, or reflections:")
    if st.button("Save Entry") and entry:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure directory exists
        os.makedirs("faith_journal", exist_ok=True)
        filename = os.path.join("faith_journal", f"journal_{ts}.txt")
        try:
            with open(filename, "w", encoding="utf-8") as f: f.write(entry)
            st.success(f"Saved as `{filename}`.")
            # Added option for AI insight after saving
            if st.checkbox("Get spiritual insight from this entry?"):
                 with st.spinner("Analyzing your entry..."):
                      insight = ask_gpt_conversation(f"Analyze this faith journal entry and offer spiritual insight and encouragement based on biblical principles: {entry}")
                      st.markdown("**💡 Insight:**"); st.write(insight)
        except Exception as e:
            st.error(f"Failed to save entry: {e}")

# ================================================================
# TAILORED LEARNING PATH (Kept for compatibility, but Learn Module is preferred)
# ================================================================
def run_learning_path_mode():
    st.subheader("📚 Tailored Learning Path (Legacy)")
    st.warning("Consider using the 'Learn Module' for a more interactive experience.")
    user_type = st.selectbox("User type:", ["child", "adult"])
    goal = st.text_input("Learning goal:")
    level = st.selectbox("Bible knowledge level:", ["beginner", "intermediate", "advanced"])
    styles = st.multiselect(
        "Preferred learning styles:",
        ["storytelling", "questions", "memory games", "reflection", "devotional"],
    )
    if st.button("Generate Legacy Path") and goal and styles:
        style_str = ", ".join(styles)
        prompt = (f"Design a creative Bible learning path outline for a {user_type} with goal '{goal}', level '{level}', "
                  f"using these learning styles: {style_str}. Provide a list of suggested topics or activities.")
        result = ask_gpt_conversation(prompt)
        st.text_area("📘 Learning Path Outline", result, height=500)

# ================================================================
# BIBLE BETA
# ================================================================
def run_bible_beta():
    st.subheader("📘 Bible Beta Mode")
    st.info("🧪 Experimental: Read Bible chapters.")
    book = st.text_input("Book (e.g., John):")
    # Changed to text input for flexibility, e.g., "John 3"
    passage_ref = st.text_input("Chapter or Passage (e.g., 3 or 3:1-16):", "1") 
    translation_beta = st.selectbox("Translation:", VALID_TRANSLATIONS, key="beta_trans")

    if st.button("Display Passage") and book and passage_ref:
        full_ref = f"{book} {passage_ref}"
        try:
            with st.spinner(f"Fetching {full_ref}..."):
                 text = fetch_bible_verse(full_ref, translation_beta)
                 st.text_area(f"📖 {full_ref} ({translation_beta.upper()})", value=text, height=400)
            
            # Optional summarization integrated below text area
            if st.checkbox("✨ Summarize this passage?"):
                with st.spinner("Generating summary..."):
                     summary = ask_gpt_conversation(f"Summarize and explain the key points of this Bible passage: {text} ({full_ref})")
                     st.markdown("**💬 Summary & Key Points:**"); st.markdown(summary)

        except Exception as e:
            st.error(f"Error fetching passage: {e}")

# ================================================================
# SERMON TRANSCRIBER & SUMMARIZER (YouTube or file upload)
# ================================================================
def _convert_to_wav_if_needed(src_path: str) -> str:
    """If Whisper has trouble with container, convert to 16k mono WAV using ffmpeg."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wav_path = tmp_file.name
    cmd = [_FFMPEG_BIN, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
    try:
        # Added timeout and capture stderr
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120) 
    except subprocess.CalledProcessError as e:
        raise Exception(f"ffmpeg conversion failed with exit code {e.returncode}: {e.stderr}")
    except subprocess.TimeoutExpired:
         raise Exception("ffmpeg conversion timed out after 2 minutes.")
    return wav_path

def download_youtube_audio(url: str) -> tuple[str, str, str]:
    """Download audio *without* postprocessing (so yt_dlp won't call ffprobe)."""
    # Use a specific file extension preferred by yt-dlp format selection
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file: 
        output_path = temp_file.name
    
    ydl_opts = {
        # Prefer m4a, fallback to best audio
        "format": "bestaudio[ext=m4a]/bestaudio/best", 
        "outtmpl": output_path, # Use the temp file path directly
        "ffmpeg_location": os.environ.get("FFMPEG_LOCATION", _FFMPEG_DIR),
        "quiet": True, 
        "retries": 3, 
        "noprogress": True,
        "http_headers": { # Standard browser headers
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9", 
            "Referer": "https://www.youtube.com/",
        },
        # Use cookies if available
        **({"cookiefile": "cookies.txt"} if os.path.exists("cookies.txt") else {}), 
        # Set socket timeout
        "socket_timeout": 30,
    }
    try:
        # Use context manager for yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled Sermon")
            uploader = info.get("uploader", "Unknown")
            
        # Check if file exists and is not empty AFTER download attempt
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Audio download failed or resulted in an empty file.")
            
        return output_path, uploader, title
    except yt_dlp.utils.DownloadError as e: # Catch specific download errors
         raise Exception(f"❌ YouTube download error: {e}")
    except Exception as e:
        raise Exception(f"❌ Failed during YouTube audio processing: {e}")

def run_sermon_transcriber():
    st.subheader("🎧 Sermon Transcriber & Summarizer")
    st.info("Upload sermon audio or paste a YouTube link (max ~15-20 mins recommended due to processing limits).")
    yt_link = st.text_input("📺 YouTube Link:")
    audio_file = st.file_uploader("🎙️ Or upload audio (MP3/WAV/M4A/MP4):", type=["mp3", "wav", "m4a", "mp4"])

    if st.button("⏺️ Transcribe & Summarize") and (yt_link or audio_file):
        audio_path = None
        preacher = "Unknown"; title = "Untitled Sermon"
        cleanup_path = None # Path to delete later

        try:
            with st.spinner("Processing audio..."):
                if yt_link:
                    # Validate URL format roughly
                    if not yt_link.startswith(("http://", "https://")):
                         raise ValueError("Invalid YouTube URL provided.")
                    # Get duration without full download first
                    with yt_dlp.YoutubeDL({"quiet": True, "noprogress": True}) as ydl:
                        info = ydl.extract_info(yt_link, download=False)
                        duration = info.get("duration")
                        # Add a reasonable limit, e.g., 20 mins (1200 seconds)
                        if duration and duration > 1200: 
                            st.warning(f"⚠️ Warning: Video is longer than 20 minutes ({duration}s). Transcription might be slow or fail.")
                        preacher = info.get("uploader", "Unknown") or "Unknown"
                        title = info.get("title", "Untitled Sermon") or "Untitled Sermon"
                    # Download the audio
                    audio_path, _, _ = download_youtube_audio(yt_link)
                    cleanup_path = audio_path # Mark for deletion

                elif audio_file:
                    # Save uploaded file to a temporary path
                    suffix = os.path.splitext(audio_file.name)[1].lower() or ".tmp"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                        temp_audio.write(audio_file.getvalue())
                        audio_path = temp_audio.name
                        cleanup_path = audio_path # Mark for deletion
                    title = os.path.splitext(audio_file.name)[0] # Use filename as title

                if not audio_path:
                    st.warning("Please provide a YouTube link or upload an audio file.")
                    return

            with st.spinner("Transcribing audio... (This may take a few minutes)"):
                # Load the base model (faster, less accurate)
                # Consider 'small' or 'medium' for better accuracy if performance allows
                model = whisper.load_model("base") 
                transcription = None
                try: 
                    transcription = model.transcribe(audio_path, fp16=False) # fp16=False for CPU
                except Exception as e_transcribe:
                    st.warning(f"Initial transcription failed ({e_transcribe}), trying conversion to WAV...")
                    try:
                        wav_path = _convert_to_wav_if_needed(audio_path)
                        cleanup_path = wav_path # Now delete the wav file instead
                        transcription = model.transcribe(wav_path, fp16=False)
                    except Exception as e_convert:
                         raise Exception(f"Transcription failed even after conversion: {e_convert}")

                if not transcription or not transcription.get("text"):
                     raise Exception("Transcription failed or produced empty text.")

                transcript_text = transcription["text"].strip()
                st.success("✅ Transcription complete.")
                st.markdown("### 📝 Transcript")
                st.text_area("Full Transcript", transcript_text, height=300)

            with st.spinner("Generating summary..."):
                # Limit summary input to avoid excessive token usage
                summary_input_limit = 4000 
                short_transcript = transcript_text[:summary_input_limit] 
                
                summary_prompt = f"""You are a sermon summarizer. From the transcript below, provide a concise summary including:
- **Main Topic/Theme:** (Identify the core subject)
- **Key Bible Verses Referenced:** (List primary scriptures mentioned)
- **Main Takeaways:** (Bullet points of key messages or lessons)
- **Potential Reflection Questions:** (2-3 questions for the listener to consider)

Preacher: {preacher}
Title: {title}
Transcript Snippet (first ~{summary_input_limit} characters):
{short_transcript}"""

                summary = ask_gpt_conversation(summary_prompt)
                st.markdown("### 🧠 Sermon Summary")
                st.markdown(summary)

                # Save results
                os.makedirs("sermon_journal", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = re.sub(r'[\\/*?:"<>|]', "", title)[:50] # Sanitize title for filename
                
                transcript_filename = os.path.join("sermon_journal", f"transcript_{base_filename}_{ts}.txt")
                summary_filename = os.path.join("sermon_journal", f"summary_{base_filename}_{ts}.txt")
                
                with open(transcript_filename, "w", encoding="utf-8") as f: f.write(transcript_text)
                with open(summary_filename, "w", encoding="utf-8") as f: f.write(summary)
                st.success(f"Saved transcript and summary to `{transcript_filename}` and `{summary_filename}`.")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
        finally:
             # Clean up temporary audio file
            if cleanup_path and os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                except Exception as e_clean:
                     st.warning(f"Could not delete temporary file {cleanup_path}: {e_clean}")

# ================================================================
# SIMPLE STUDY PLAN
# ================================================================
def run_study_plan():
    st.subheader("📅 Personalized Bible Study Plan")
    goal = st.text_input("Study goal (e.g., 'Grow in faith', 'Understand forgiveness'):")
    duration = st.slider("How many days do you want your plan to last?", 7, 60, 14)
    focus = st.text_input("Focus area (e.g., 'Parables', 'Life of Paul', leave blank for general):")
    level_plan = st.selectbox("Your Bible knowledge level:", ["Beginner", "Intermediate", "Advanced"], key="plan_level")
    include_reflections = st.checkbox("Include daily reflection questions?", True)

    if st.button("Generate Study Plan") and goal:
        with st.spinner("✍️ Creating your personalized study plan..."):
            prompt = f"""You are a mature Bible mentor creating a detailed, Scripture-based daily study plan.
**Parameters:**
- Goal: {goal}
- Duration: {duration} days
- Focus area: {focus if focus else 'Based on Goal'}
- Knowledge level: {level_plan}

**Instructions:**
Design a day-by-day Bible study plan. For each day:
- **Day #:**
- **Theme/Title:** A concise theme for the day.
- **Reading:** Suggest 1–2 specific Bible passages (e.g., John 3:1-16).
- **Summary:** Explain the passage's meaning and relevance in 3-5 sentences, tailored to the '{level_plan}' level.
- **Connection:** Include 1 cross-reference verse and briefly explain its connection.
- **Application:** Provide a practical life application point or takeaway.
{'- **Reflection:** Add 1 thoughtful reflection question for journaling.' if include_reflections else ''}

Format clearly for each day. End with a brief closing paragraph encouraging consistency. Tone should be pastoral, warm, and theologically sound. Ensure passages logically progress towards the goal."""
            try:
                plan = ask_gpt_conversation(prompt)
                st.markdown("### 📘 Your Study Plan")
                st.text_area("Generated Plan", plan, height=600) # Added label
                
                # Save the plan
                os.makedirs("study_plans", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create a safe filename from the goal
                safe_goal = re.sub(r'[\\/*?:"<>|]', "", goal)[:30]
                file_path = os.path.join("study_plans", f"study_plan_{safe_goal}_{timestamp}.txt")
                
                with open(file_path, "w", encoding="utf-8") as f: f.write(plan)
                st.success(f"✅ Study plan saved to `{file_path}`.")
            except Exception as e: 
                st.error(f"❌ Error generating study plan: {e}")

# ================================================================
# VERSE OF THE DAY, PRAYER STARTER, FAST DEVOTIONAL, SMALL GROUP
# ================================================================
def run_verse_of_the_day():
    st.subheader("🌅 Verse of the Day")
    # Expanded list of commonly quoted books
    books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", 
             "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", 
             "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", 
             "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", 
             "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", 
             "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", 
             "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", 
             "Haggai", "Zechariah", "Malachi", 
             "Matthew", "Mark", "Luke", "John", "Acts", "Romans", 
             "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", 
             "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", 
             "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", 
             "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", 
             "Jude", "Revelation"]
    try:
         # Generate a reference - consider API/library for valid chapter/verse counts later
        book = random.choice(books)
        # Simplified chapter/verse generation for now
        chapter = random.randint(1, 5) # Assume at least 5 chapters
        verse = random.randint(1, 10) # Assume at least 10 verses
        ref = f"{book} {chapter}:{verse}"

        with st.spinner("Fetching Verse of the Day..."):
             text = fetch_bible_verse(ref, "web") # Default to WEB translation
             st.success(f"**{ref} (WEB)**\n\n> {text}") # Use blockquote

             reflection_prompt = f"Offer a brief (2-3 sentences), warm, and practical reflection on this Bible verse, focusing on one simple takeaway: '{text}' ({ref})"
             reflection = ask_gpt_conversation(reflection_prompt)
             st.markdown("**💬 Reflection & Takeaway:**")
             st.write(reflection)

    except Exception as e: 
        # Handle cases where the random verse might be invalid
        st.error(f"Could not fetch Verse of the Day ({ref}): {e}") 

def run_prayer_starter():
    st.subheader("🙏 Prayer Starter")
    theme = st.text_input("What's on your heart? (e.g., gratitude, anxiety, guidance, forgiveness):")
    if st.button("Generate Prayer Starter") and theme:
        with st.spinner("Crafting a prayer..."):
             prayer = ask_gpt_conversation(f"Write a short (3-5 sentences), theologically faithful prayer starter focused on '{theme}'. Address God reverently (e.g., 'Heavenly Father', 'Lord Jesus') and base it on biblical truths. Avoid clichés.")
             st.text_area("Your Prayer Starter:", prayer, height=200)

def run_fast_devotional():
    st.subheader("⚡ Fast Devotional")
    topic = st.text_input("Devotional Topic (e.g., hope, perseverance, love, faith):")
    if st.button("Generate Fast Devotional") and topic:
         with st.spinner("Writing devotional..."):
              devo = ask_gpt_conversation(f"Compose a short devotional (approx. 150-200 words) on the topic of '{topic}'. Include one primary Bible verse, 1-2 related cross-references, a brief explanation connecting them to the topic, and one practical challenge or encouragement for today.")
              st.text_area(f"Devotional on {topic}:", devo, height=350)

def run_small_group_generator():
    st.subheader("👥 Small Group Guide Generator")
    passage = st.text_input("Bible Passage for Discussion (e.g., James 1:2-8):")
    group_size = st.slider("Approximate Group Size:", 2, 15, 5) # Optional context
    
    if st.button("Create Discussion Guide") and passage:
        with st.spinner("Generating guide..."):
            try: 
                # Fetch text to provide context to the AI
                text = fetch_bible_verse(passage, "web") 
                context_text = f"The passage is {passage}:\n\n> {text}\n\n"
            except Exception: 
                st.warning(f"Could not fetch text for {passage}, generating questions based on reference only.")
                context_text = f"The passage is {passage}.\n\n"

            guide_prompt = f"""Create a concise small group discussion guide for a group of about {group_size} people based on the following passage:
{context_text}
**Include:**
- **Opener:** One brief icebreaker question related to the theme.
- **Discussion Questions:** 3-4 thoughtful questions exploring Observation (What does it say?), Interpretation (What does it mean?), and Application (How does it apply to our lives?).
- **Key Truth:** One central takeaway message from the passage.
- **Closing:** A short closing prayer prompt or challenge."""
            guide = ask_gpt_conversation(guide_prompt)
            st.text_area(f"Discussion Guide for {passage}:", guide, height=500)

# ================================================================
# LEARN MODULE (NEW, PERSONALIZED WORKFLOW)
# ================================================================
def _learn_extract_json_any(response_text: str):
    """Robustly extracts JSON object or array from a string."""
    if not response_text: return None # Handle empty input
    # Prioritize fenced code blocks
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: Find first '{' or '[' and try to parse from there
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
            # Try to find matching closing bracket/brace - basic nesting support
            open_count = 0
            end_index = -1
            for i, char in enumerate(response_text[start_index:]):
                 if char == response_text[start_index]:
                      open_count += 1
                 elif char == end_char:
                      open_count -= 1
                 if open_count == 0:
                      end_index = start_index + i + 1
                      break
            if end_index != -1:
                 json_str = response_text[start_index:end_index]
            else: # Fallback if matching bracket not found
                 json_str = response_text[start_index:] 
        else:
             st.error("No JSON object or array found in AI response.")
             return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from AI response: {e}\nRaw content was: {json_str[:500]}...") # Show snippet
        return None

# ============================
# LEARN MODULE SUPPORT HELPERS
# ============================
TOKENS_BY_TIME = {"15 minutes": 1800, "30 minutes": 3000, "45 minutes": 4000} # Rough estimates

def ask_gpt_json(prompt: str, max_tokens: int = 4000):
    """Makes a call to the OpenAI API expecting a JSON response."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful curriculum designer. Respond ONLY with valid JSON that adheres to the requested structure."}, # Emphasize JSON only
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2, # Lower temperature for more deterministic JSON output
            response_format={"type": "json_object"} # Use JSON mode if available
        )
        return resp.choices[0].message.content
    except Exception as e: # Catch potential API errors (like invalid request due to JSON mode)
        try: # Fallback without JSON mode
            st.warning(f"JSON mode failed ({e}), attempting standard request...")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful curriculum designer. Respond ONLY with valid JSON wrapped in ```json ``` tags."}, # Instruct wrapping
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return resp.choices[0].message.content
        except Exception as e2:
             st.error(f"GPT JSON call failed completely: {e2}")
             return None


def _answers_match(user_answer, correct_answer, question_type="text") -> bool:
    """Flexible answer matching for quizzes, including fuzzy matching for text."""
    if user_answer is None or correct_answer is None: return False
    
    user_ans_str = str(user_answer).strip()
    correct_ans_str = str(correct_answer).strip()
    
    # Exact match needed for multiple choice and true/false
    if question_type == 'multiple_choice' or question_type == 'true_false':
        return user_ans_str.lower() == correct_ans_str.lower()
    
    # Use fuzzy matching for fill-in-the-blank (tolerant of typos)
    # fuzz.ratio calculates similarity from 0 to 100
    similarity_ratio = fuzz.ratio(user_ans_str.lower(), correct_ans_str.lower())
    # Adjust threshold as needed - 85 allows for minor errors
    return similarity_ratio >= 85 

def summarize_lesson_content(lesson_data: dict) -> str:
    """Summarizes lesson content for context memory."""
    text_content = " ".join([sec.get('content', '') for sec in lesson_data.get('lesson_content_sections', []) if sec.get('type') == 'text'])
    if not text_content: return "No textual content available for summary."
    # Limit length passed to summarizer
    prompt = f"Summarize the key topic of the following Bible lesson text in one concise sentence (less than 20 words): {text_content[:2000]}"
    try:
        # Use a faster/cheaper model potentially? For now, stick with primary.
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=60, temperature=0.1) 
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Could not generate lesson summary: {e}")
        return lesson_data.get("lesson_title", "Summary unavailable.") # Fallback

# -------------------------
# PROMPTS
# -------------------------
def create_full_learning_plan_prompt(form_data: dict) -> str:
    """Creates the master prompt to generate the entire curriculum."""
    pacing_to_lessons_per_level = {
        "A quick, high-level overview": 1,
        "A steady, detailed study": 2,
        "A deep, comprehensive dive": 3
    }
    num_lessons_per_level = pacing_to_lessons_per_level.get(form_data['pacing'], 2) # Default to steady

    return f"""
You are an expert theologian and personalized curriculum designer creating a Bible study plan.
User Profile:
- Topics of Interest: {form_data['topics']}
- Current Knowledge: {form_data['knowledge_level']} (Derived from diagnostic)
- Learning Goal: {", ".join(form_data['objectives'])}
- Common Struggles: {", ".join(form_data['struggles'])}
- Preferred Learning Style: {form_data['learning_style']}
- Desired Pacing: {form_data['pacing']}
- Time Commitment per Lesson: {form_data['time_commitment']}

Task: Design a complete Bible study curriculum plan based on this profile.
1. Create a personalized `plan_title`.
2. Write a brief, encouraging `introduction`.
3. Determine the appropriate number of levels based on the pacing (`quick`: 2-3 levels, `steady`: 3-5 levels, `deep`: 5-7 levels).
4. For each level, create a concise `name` and `topic` that flows logically towards the user's goals.
5. For each level, set `num_lessons` based on the user's 'Desired Pacing' (1 for 'quick', 2 for 'steady', 3 for 'deep').

Output ONLY a single, valid JSON object with keys "plan_title", "introduction", and "levels" (a list of objects, each with "name", "topic", and "num_lessons").
Example level object: {{"name": "Level 1: Title", "topic": "Brief topic description", "num_lessons": {num_lessons_per_level}}}
"""

def create_lesson_prompt(level_topic: str, lesson_number: int, total_lessons_in_level: int, form_data: dict, previous_lesson_summary: str = None) -> str:
    """Generates prompt for creating a single lesson, tailored to user level."""
    length_instructions = {
        "15 minutes": "Generate exactly 3 short teaching sections (approx. 250 words each) and 2-3 knowledge checks.",
        "30 minutes": "Generate exactly 5 medium teaching sections (approx. 300 to 375 words each) and 3-4 knowledge checks.",
        "45 minutes": "Generate exactly 7 detailed teaching sections (approx. 400 to 450 words each) and 5-7 knowledge checks."
    }
    context_clause = f" This lesson should logically follow the previous one, which covered: '{previous_lesson_summary}'." if previous_lesson_summary else ""

    return f"""
You are an expert theologian creating Lesson {lesson_number}/{total_lessons_in_level} on "{level_topic}".
User Profile:
- Knowledge Level: {form_data['knowledge_level']}
- Learning Style: {form_data['learning_style']}
- Time Commitment: {form_data['time_commitment']}

**INSTRUCTIONS:**
1.  **Tailor Depth & Vocabulary:** Adjust content complexity for the '{form_data['knowledge_level']}' user. Use simpler language and core concepts for 'Just starting out'; introduce richer theology for 'comfortable with deeper concepts'.
2.  **Cite Scripture:** Embed specific Bible references (e.g., Genesis 1:1) within the `content` of text sections.
3.  **Content & Structure:** Fulfill the requirements: {length_instructions.get(form_data['time_commitment'], 'Generate content fitting the time.')} Build sections logically.{context_clause}
4.  **Knowledge Checks:** Ensure checks directly relate to the preceding text section and include `question`, `question_type`, `options`/`correct_answer`, and `biblical_reference`.
5.  **Tone:** Maintain a pastoral, encouraging, and biblically sound tone.

Output ONLY a valid JSON object with keys "lesson_title", "lesson_content_sections" (list of objects with "type" ['text' or 'knowledge_check'] and relevant fields), and "summary_points" (list of 3 key takeaways).
Example text section: {{"type": "text", "content": "Start here (Reference 1:1)... then consider this (Reference 1:2)."}}
Example check: {{"type": "knowledge_check", "question_type": "multiple_choice", "question": "Q?", "options": ["A","B","C","D"], "correct_answer": "A", "biblical_reference": "Ref 1:1"}}
"""

def create_level_quiz_prompt(level_topic: str, lesson_summaries: list) -> str:
    """Generates prompt for creating the end-of-level quiz."""
    summaries_text = "\n".join(f"- {s}" for i, s in enumerate(lesson_summaries) if s) # Filter empty summaries
    return f"""
You are a Bible teacher creating a 10-question cumulative quiz for the level titled "{level_topic}".
The lessons covered these key points:
{summaries_text}

**Instructions:** Create a quiz based *only* on the topics mentioned in the lesson summaries above.
- Generate exactly 10 questions.
- Mix question types: 'multiple_choice', 'true_false', 'fill_in_the_blank'.
- Each question object MUST include: 'question' (string), 'question_type' (string), 'correct_answer' (string).
- For 'multiple_choice', also include 'options' (list of 4 strings).
- For 'true_false', `correct_answer` should be "True" or "False", and you can optionally include `options` as ["True", "False"].
- Include a relevant 'biblical_reference' (string) for each question if applicable.

Output ONLY a valid JSON array containing the 10 question objects.
"""

# -------------------------
# KNOWLEDGE CHECK & QUIZ UI
# -------------------------
def display_knowledge_check_question(S):
    """Displays knowledge check, handles submission, and shows explanation on incorrect."""
    level_data = S["levels"][S["current_level"]]
    current_lesson = level_data["lessons"][S["current_lesson_index"]]
    q = current_lesson["lesson_content_sections"][S["current_section_index"]]

    st.markdown("---")
    st.markdown(f"#### ✅ Knowledge Check")
    st.markdown(f"**{q.get('question', 'Missing question text.')}**")

    user_answer = None
    input_key = f"kc_{S['current_level']}_{S['current_lesson_index']}_{S['current_section_index']}"
    q_type = q.get('question_type')

    # Display input widget based on question type
    if q_type == 'multiple_choice':
        user_answer = st.radio("Select your answer:", q.get('options', []), key=input_key, index=None)
    elif q_type == 'true_false':
        user_answer = st.radio("True or False?", ['True', 'False'], key=input_key, index=None)
    elif q_type == 'fill_in_the_blank':
        user_answer = st.text_input("Fill in the blank:", key=input_key)
    else:
        st.error(f"Unknown question type: {q_type}")
        return # Avoid proceeding if type is wrong

    # Handle submission
    if st.button("Submit Answer", key=f"submit_{input_key}"):
        # Basic validation for radio buttons
        if user_answer is None and q_type in ['multiple_choice', 'true_false']:
             st.warning("Please select an answer.")
             # Keep kc_answered_incorrectly state if it was already set
             if S.get("kc_answered_incorrectly"): st.rerun() 
             return # Don't proceed without an answer

        is_correct = _answers_match(user_answer, q.get('correct_answer'), q_type)
        if is_correct:
            st.success("Correct! Moving on.")
            S["current_section_index"] += 1
            # Clear flags if they were set from a previous attempt
            if "kc_answered_incorrectly" in S: del S["kc_answered_incorrectly"]
            if "last_incorrect_answer" in S: del S["last_incorrect_answer"]
            st.rerun()
        else:
            S["kc_answered_incorrectly"] = True
            S["last_incorrect_answer"] = user_answer # Store the incorrect answer
            st.rerun() # Rerun to display the explanation section

    # Display explanation if answered incorrectly on the previous run
    if S.get("kc_answered_incorrectly"):
        st.error(f"Not quite. The correct answer is: **{q.get('correct_answer')}**")
        
        reference = q.get('biblical_reference', '')
        if reference:
            try:
                # Use a spinner while fetching verse and explanation
                with st.spinner("Loading context and explanation..."):
                    verse_text = fetch_bible_verse(reference)
                    incorrect_ans = S.get("last_incorrect_answer", "their answer") # Get the stored wrong answer
                    
                    explanation_prompt = (
                        f"A student was asked: '{q.get('question')}' "
                        f"They incorrectly answered: '{incorrect_ans}'. The correct answer is: '{q.get('correct_answer')}'. "
                        f"Based on the Bible verse '{reference}' which says: '{verse_text}', "
                        f"please provide a brief, one-paragraph explanation focusing on why their answer '{incorrect_ans}' was incorrect and why '{q.get('correct_answer')}' is the right one according to the verse."
                    )
                    
                    explanation = ask_gpt_conversation(explanation_prompt)

                # Display verse and explanation in an expander
                with st.expander(f"📖 See {reference} for context and explanation"):
                    st.markdown(f"**Verse Text:**\n\n> *{verse_text}*") # Use blockquote for verse
                    st.markdown("---")
                    st.markdown(f"**Explanation:**\n\n{explanation}")

            except Exception as e:
                # Fallback message if fetching fails
                st.info(f"See {reference} for context. (Could not load additional details: {e})")
        else:
             st.info("No specific Bible reference was provided for this question.") # Handle missing reference

        # Continue button appears only after showing the error/explanation
        if st.button("Continue Lesson", key=f"continue_{input_key}"):
            # Clear flags before moving on
            if "kc_answered_incorrectly" in S: del S["kc_answered_incorrectly"]
            if "last_incorrect_answer" in S: del S["last_incorrect_answer"]
            S["current_section_index"] += 1
            st.rerun()

# --- run_level_quiz remains unchanged from your last correct version ---
def run_level_quiz(S):
    level_data = S["levels"][S["current_level"]]
    quiz_questions = level_data.get("quiz_questions", [])
    q_index = S.get("current_question_index", 0)

    st.markdown("### 📝 Final Level Quiz")
    if not quiz_questions: 
        st.warning("Quiz questions not generated yet or generation failed.")
        # Optionally add a button to retry generation?
        return

    # Ensure quiz_questions is a list
    if not isinstance(quiz_questions, list):
         st.error("Quiz data is not in the expected format (list). Please restart the level.")
         return

    total_questions = len(quiz_questions)
    if total_questions == 0:
         st.warning("No quiz questions found for this level.")
         return
         
    st.progress(q_index / total_questions) # Use q_index for progress
    st.markdown(f"**Score: {S.get('user_score', 0)}/{total_questions}**")

    if q_index < total_questions:
        q = quiz_questions[q_index]
        # Basic check for question format
        if not isinstance(q, dict) or 'question' not in q or 'correct_answer' not in q:
             st.error(f"Error: Invalid question format at index {q_index}. Skipping question.")
             S["current_question_index"] = q_index + 1
             st.rerun()
             return

        st.markdown(f"**Question {q_index + 1}:** {q.get('question', '')}")
        user_answer = None
        q_key = f"quiz_{S['current_level']}_{q_index}"
        q_type = q.get('question_type')

        # Display input widget
        if q_type == 'multiple_choice':
            options = q.get('options', [])
            if not options: st.error("Error: Multiple choice question has no options."); return
            user_answer = st.radio("Answer:", options, key=q_key, index=None)
        elif q_type == 'true_false':
            user_answer = st.radio("Answer:", ["True", "False"], key=q_key, index=None)
        elif q_type == 'fill_in_the_blank':
            user_answer = st.text_input("Answer:", key=q_key)
        else:
             st.error(f"Unknown quiz question type: {q_type}"); return

        # Handle submission
        if st.button("Submit Quiz Answer", key=f"submit_{q_key}"):
            if user_answer is None and q_type in ['multiple_choice', 'true_false']:
                 st.warning("Please select an answer.")
                 return
                 
            if _answers_match(user_answer, q.get('correct_answer'), q_type):
                st.success("Correct!")
                S["user_score"] = S.get("user_score", 0) + 1
            else:
                st.error(f"Incorrect. The correct answer was: **{q.get('correct_answer')}**")
                # Optional: Add explanation fetching here too, similar to knowledge check?
                # For now, just moves on.
            S["current_question_index"] = q_index + 1
            st.rerun()
    else:
        # Quiz completed
        score = S.get('user_score', 0)
        passing_score = total_questions * 0.7 
        st.success(f"### Quiz Completed! Final Score: {score}/{total_questions}")
        
        if score >= passing_score:
            st.balloons()
            st.markdown(f"Congratulations! You passed {level_data.get('name','this level')}!")
            # Check if there's a next level before showing the button
            if S["current_level"] + 1 < len(S["levels"]):
                if st.button("Go to Next Level ▶️"):
                    S["current_level"] += 1
                    S["current_lesson_index"] = 0
                    S["current_section_index"] = 0
                    S["quiz_mode"] = False
                    S["current_question_index"] = 0 
                    S["user_score"] = 0 
                    # Optionally clear quiz questions for the completed level to save state?
                    # if "quiz_questions" in S["levels"][S["current_level"]-1]:
                    #     del S["levels"][S["current_level"]-1]["quiz_questions"]
                    st.rerun()
            else:
                 # This was the last level
                 st.info("You've completed all levels in this plan!")
                 if st.button("Start a New Journey"):
                      st.session_state.learn_state = {} # Reset everything
                      st.rerun()

        else:
            st.error("You didn't reach the passing score. Please review the lessons and try the quiz again.")
            if st.button("Review Lessons"):
                 # Reset state to go back to the first lesson of the current level
                 S["quiz_mode"] = False
                 S["current_lesson_index"] = 0
                 S["current_section_index"] = 0
                 S["current_question_index"] = 0 
                 S["user_score"] = 0
                 # Keep generated quiz questions so user doesn't wait again
                 st.rerun()
            if st.button("Retake Quiz"):
                S["current_question_index"] = 0
                S["user_score"] = 0
                st.rerun()

# ================================================================
# DIAGNOSTIC QUIZ FUNCTION
# ================================================================
def run_diagnostic_quiz():
    st.subheader("Quick Bible Knowledge Check")
    st.info("Let's figure out the best starting point for you with a few quick questions.")

    S_learn = st.session_state.learn_state 

    if 'diag_q_index' not in S_learn:
        S_learn['diag_q_index'] = 0
        S_learn['diag_score'] = 0

    q_index = S_learn['diag_q_index']

    if q_index < len(DIAGNOSTIC_QUESTIONS):
        q_data = DIAGNOSTIC_QUESTIONS[q_index]
        st.markdown(f"**Question {q_index + 1} of {len(DIAGNOSTIC_QUESTIONS)}:**")
        st.markdown(f"*{q_data['question']}*")
        
        # Add "I don't know" if not present (defensive coding)
        options = q_data['options']
        if "I don't know" not in options: options.append("I don't know")
        
        user_answer = st.radio(
            "Select your answer:",
            options, # Use the potentially modified options list
            key=f"diag_q_{q_index}",
            index=None 
        )

        if st.button("Submit Answer", key=f"diag_submit_{q_index}"):
            if user_answer:
                if user_answer == q_data['correct']:
                    S_learn['diag_score'] += 1
                    # Don't show "Correct!" during diagnostic to speed it up? Optional.
                    # st.success("Correct!") 
                # No specific feedback needed for incorrect answers during diagnostic
                # else:
                #     st.error(f"Not quite. The correct answer was: {q_data['correct']}")
                
                S_learn['diag_q_index'] += 1
                st.rerun() 
            else:
                st.warning("Please select an answer.")
    else:
        score = S_learn['diag_score']
        total = len(DIAGNOSTIC_QUESTIONS)
        knowledge_level = ""
        # Define levels based on score ranges
        if score / total <= 0.4: # Covers 0 or 1 out of 3
            knowledge_level = "Just starting out"
        elif score / total <= 0.7: # Covers 2 out of 3
            knowledge_level = "I know the main stories"
        else: # Covers 3 out of 3
            knowledge_level = "I'm comfortable with deeper concepts"

        st.success(f"Knowledge check complete! Score: {score}/{total}")
        st.info(f"Based on your answers, we'll tailor the plan using the **'{knowledge_level}'** level as a starting point.")
        
        S_learn['derived_knowledge_level'] = knowledge_level
        S_learn['diagnostic_complete'] = True
        
        # Automatically proceed by rerunning, removing the need for the button
        st.rerun() 
        # if st.button("Continue to Plan Setup"):
        #      st.rerun() # Proceed to the main form

# ================================================================
# LEARNING PLAN SETUP (QUESTIONNAIRE)
# ================================================================
def run_learn_module_setup():
    st.info("Now, let's create a personalized learning plan based on your unique needs.")
    derived_knowledge_level = st.session_state.learn_state.get('derived_knowledge_level', "Not determined")
    st.markdown(f"**Assessed Knowledge Level:** {derived_knowledge_level}") 

    with st.form("user_profile_form"):
        topics_input = st.text_input("**What topics are on your heart to learn about?** (Separate with commas)", "Understanding grace, The life of David")
        objectives_input = st.multiselect("**What do you hope to achieve with this study?**", ["Gain knowledge and understanding", "Find practical life application", "Strengthen my faith", "Prepare to teach others"], default=["Gain knowledge and understanding"]) # Added a default
        struggles_input = st.multiselect("**What are some of your common challenges?**", ["Understanding historical context", "Connecting it to my daily life", "Staying consistent", "Dealing with difficult passages"])
        learning_style_input = st.selectbox("**Preferred learning style:**", ["Analytical", "Storytelling", "Practical", "Reflective"]) # Added Reflective
        pacing_input = st.select_slider("**How would you like to pace your learning?**", options=["A quick, high-level overview", "A steady, detailed study", "A deep, comprehensive dive"], value="A steady, detailed study") # Added default
        time_commitment_input = st.selectbox("**How much time can you realistically commit to each lesson?**", ["15 minutes", "30 minutes", "45 minutes"], index=1) # Default to 30 mins
        
        submitted = st.form_submit_button("🚀 Generate My Tailor-Made Plan")
    
    if submitted:
        form_data = {
            'topics': topics_input,
            'knowledge_level': derived_knowledge_level, 
            'objectives': objectives_input,
            'struggles': struggles_input,
            'learning_style': learning_style_input.lower(), # Ensure lowercase for consistency
            'pacing': pacing_input,
            'time_commitment': time_commitment_input
        }
        
        if not form_data['topics'] or not form_data['objectives']:
            st.warning("Please fill out the topics and objectives to generate a plan.")
            return
        if form_data['knowledge_level'] == "Not determined":
             st.error("Knowledge level could not be determined. Please restart.")
             return
            
        with st.spinner("Our AI is designing your personalized curriculum..."):
            master_prompt = create_full_learning_plan_prompt(form_data)
            plan_resp = ask_gpt_json(master_prompt, max_tokens=2500)
            plan_data = _learn_extract_json_any(plan_resp) if plan_resp else None
            
            if plan_data and isinstance(plan_data, dict) and "levels" in plan_data and isinstance(plan_data["levels"], list):
                S = st.session_state.learn_state
                S.update({
                    "plan": plan_data, "levels": plan_data["levels"], "form_data": form_data,
                    "current_level": 0, "current_lesson_index": 0, "current_section_index": 0,
                    "quiz_mode": False, "current_question_index": 0, "user_score": 0,
                })
                # Keep diagnostic_complete and derived_knowledge_level
                st.rerun()
            else:
                st.error("Failed to generate a valid learning plan from AI response. Please try adjusting your inputs or try again later.")
                if plan_resp: st.text_area("Raw AI Response (for debugging):", plan_resp, height=200)

# ================================================================
# MAIN LEARN MODULE FLOW
# ================================================================
def run_learn_module():
    st.subheader("📚 Learn Module — Personalized Bible Learning")
    if "learn_state" not in st.session_state: st.session_state.learn_state = {}
    S = st.session_state.learn_state

    # --- Run Diagnostic Quiz if not completed ---
    if not S.get("diagnostic_complete", False):
        run_diagnostic_quiz() 
        return 
    
    # --- Run Plan Setup if plan not generated ---
    if "plan" not in S:
        run_learn_module_setup()
        return

    # --- Execute Learning Plan ---
    st.title(S["plan"].get("plan_title", "Your Learning Journey"))
    st.write(S["plan"].get("introduction", ""))

    # Check for plan completion
    if S["current_level"] >= len(S["levels"]):
        st.success("🎉 You've completed your entire learning journey!")
        st.balloons()
        if st.button("Start a New Journey"):
            st.session_state.learn_state = {} # Reset state completely
            st.rerun()
        return

    level_data = S["levels"][S["current_level"]]
    st.markdown(f"--- \n## {level_data.get('name','Current Level')}")
    st.markdown(f"**Topic:** {level_data.get('topic', 'N/A')}")
    
    # Display level progress (e.g., Lesson 1 of 2)
    num_lessons = level_data.get("num_lessons", 1)
    st.caption(f"Level {S['current_level'] + 1} of {len(S['levels'])} | Lesson {S['current_lesson_index'] + 1} of {num_lessons}")


    # --- Run Quiz Mode ---
    if S.get("quiz_mode"):
        run_level_quiz(S)
        return

    # --- Lesson Generation and Display ---
    if "lessons" not in level_data: level_data["lessons"] = []

    # Check if more lessons are needed for the level
    if S["current_lesson_index"] < num_lessons:
        # Generate lesson if it doesn't exist yet
        if S["current_lesson_index"] >= len(level_data["lessons"]):
            with st.spinner(f"Generating Lesson {S['current_lesson_index'] + 1}..."):
                prev_summary = None
                # Get summary from previous lesson in this level
                if S["current_lesson_index"] > 0:
                     prev_summary = level_data["lessons"][S["current_lesson_index"] - 1].get("lesson_summary")
                # Or get summary from last lesson of previous level if first lesson
                elif S["current_level"] > 0:
                    prev_level_lessons = S["levels"][S["current_level"]-1].get("lessons", [])
                    if prev_level_lessons:
                        prev_summary = prev_level_lessons[-1].get("lesson_summary")

                lesson_max_tokens = TOKENS_BY_TIME.get(S["form_data"]['time_commitment'], 4000)
                lesson_prompt = create_lesson_prompt(
                    level_topic=level_data.get("topic"),
                    lesson_number=S["current_lesson_index"] + 1,
                    total_lessons_in_level=num_lessons,
                    form_data=S["form_data"],
                    previous_lesson_summary=prev_summary
                )
                lesson_resp = ask_gpt_json(lesson_prompt, max_tokens=lesson_max_tokens)
                lesson_data = _learn_extract_json_any(lesson_resp) if lesson_resp else None
                
                # Validate lesson data structure
                if lesson_data and isinstance(lesson_data, dict) and "lesson_content_sections" in lesson_data and isinstance(lesson_data["lesson_content_sections"], list):
                    lesson_data["lesson_summary"] = summarize_lesson_content(lesson_data)
                    level_data["lessons"].append(lesson_data)
                    S["current_section_index"] = 0 # Start at the beginning of the new lesson
                    st.rerun() # Rerun to display the newly generated lesson
                else:
                    st.error("Failed to generate valid lesson content. The AI response might be malformed. Please try again.")
                    if lesson_resp: st.text_area("Raw AI Lesson Response (for debugging):", lesson_resp, height=200)
                    # Add a button to retry generation?
                    if st.button("Retry Lesson Generation"):
                         st.rerun() # Simple retry
                    return # Stop execution if lesson failed
        
        # --- Display Current Lesson Section ---
        current_lesson = level_data["lessons"][S["current_lesson_index"]]
        st.markdown(f"### Lesson {S['current_lesson_index'] + 1}: {current_lesson.get('lesson_title', 'Untitled Lesson')}")
        
        lesson_sections = current_lesson.get("lesson_content_sections", [])
        if not lesson_sections: # Handle empty lesson case
             st.warning("This lesson appears to be empty.")
             # Automatically move to next lesson or quiz? For now, show button.
             if st.button("Proceed Anyway"):
                  S["current_section_index"] = 999 # Force completion check below
                  st.rerun()
             return

        # Check if current section index is valid
        if S["current_section_index"] < len(lesson_sections):
            section = lesson_sections[S["current_section_index"]]
            section_type = section.get("type")

            if section_type == "text":
                st.markdown(section.get("content", "*No content for this section.*"))
                if st.button("Continue Reading", key=f"cont_{S['current_level']}_{S['current_lesson_index']}_{S['current_section_index']}"):
                    S["current_section_index"] += 1
                    st.rerun()
            elif section_type == "knowledge_check":
                display_knowledge_check_question(S) # This function handles its own state and rerun
            else:
                 st.warning(f"Unknown section type '{section_type}'. Skipping.")
                 S["current_section_index"] += 1
                 st.rerun()
        
        # --- End of Lesson Reached ---
        else: 
            st.success(f"Lesson {S['current_lesson_index'] + 1} Completed!")
            st.markdown("**Key Takeaways from this Lesson:**")
            summary_points = current_lesson.get("summary_points", [])
            if summary_points:
                for point in summary_points: st.markdown(f"- {point}")
            else:
                 st.write("*No summary points provided.*")

            # Check if there are more lessons in this level
            if S["current_lesson_index"] + 1 < num_lessons:
                if st.button("Go to Next Lesson ▶️"):
                    S["current_lesson_index"] += 1
                    S["current_section_index"] = 0 # Reset for new lesson
                    st.rerun()
            # Last lesson of the level completed, move to quiz
            else:
                st.info("You've completed all lessons for this level. Time for the final quiz!")
                if st.button("Start Level Quiz"):
                    # Generate quiz questions if they don't exist
                    if "quiz_questions" not in level_data:
                        with st.spinner("Generating your level quiz..."):
                            all_summaries = [l.get("lesson_summary", "") for l in level_data["lessons"]]
                            quiz_prompt = create_level_quiz_prompt(level_data.get("topic"), all_summaries)
                            quiz_resp = ask_gpt_json(quiz_prompt, max_tokens=2500) 
                            quiz_data = _learn_extract_json_any(quiz_resp)
                            # Validate quiz data structure
                            if quiz_data and isinstance(quiz_data, list):
                                level_data["quiz_questions"] = quiz_data
                            else:
                                st.error("Failed to generate valid quiz questions. Please try starting the quiz again.")
                                if quiz_resp: st.text_area("Raw AI Quiz Response (for debugging):", quiz_resp, height=200)
                                return # Stop if generation failed
                    
                    # Check again if questions were generated successfully before switching modes
                    if "quiz_questions" in level_data and level_data["quiz_questions"]:
                        S["quiz_mode"] = True
                        S["current_question_index"] = 0
                        S["user_score"] = 0
                        st.rerun()
                    else:
                         st.error("Quiz questions are still missing. Cannot start quiz.")

    # Fallback/Error case: Lesson index is beyond expected number but not in quiz mode
    # This might happen if num_lessons logic changes or state gets corrupted
    elif not S.get("quiz_mode"):
        st.warning("It looks like you've finished the lessons for this level.")
        if st.button("Proceed to Level Quiz"):
             # Attempt to generate quiz if needed
             if "quiz_questions" not in level_data:
                  # ... (Quiz generation logic as above) ...
                  pass # Placeholder for quiz gen logic if needed here
             # Switch to quiz mode
             if "quiz_questions" in level_data and level_data["quiz_questions"]:
                  S["quiz_mode"] = True
                  S["current_question_index"] = 0
                  S["user_score"] = 0
                  st.rerun()
             else:
                  st.error("Could not prepare quiz questions.")


# ================================================================
# MAIN UI
# ================================================================
st.set_page_config(page_title="Bible GPT", layout="wide")
st.title("✅ Bible GPT")

# Sidebar Navigation
mode = st.sidebar.selectbox("Choose a mode:", [
    "Learn Module", "Bible Lookup", "Chat with GPT", "Sermon Transcriber & Summarizer",
    "Practice Chat", "Verse of the Day", "Study Plan", "Faith Journal", "Prayer Starter",
    "Fast Devotional", "Small Group Generator", 
    # "Tailored Learning Path", # Consider removing legacy option?
    "Bible Beta Mode",
    "Pixar Story Animation",
])

# Mode Routing Dictionary
mode_functions = {
    "Learn Module": run_learn_module,
    "Bible Lookup": run_bible_lookup,
    "Chat with GPT": run_chat_mode,
    "Sermon Transcriber & Summarizer": run_sermon_transcriber,
    "Practice Chat": run_practice_chat,
    "Verse of the Day": run_verse_of_the_day,
    "Study Plan": run_study_plan,
    "Faith Journal": run_faith_journal,
    "Prayer Starter": run_prayer_starter,
    "Fast Devotional": run_fast_devotional,
    "Small Group Generator": run_small_group_generator,
    # "Tailored Learning Path": run_learning_path_mode, # Keep if needed
    "Bible Beta Mode": run_bible_beta,
    "Pixar Story Animation": run_pixar_story_animation,
}

# Execute Selected Mode
if mode in mode_functions:
    try:
        mode_functions[mode]()
    except Exception as e:
         st.error(f"An unexpected error occurred in {mode}: {e}")
         # Optionally add more detailed error logging here
         # import traceback
         # st.code(traceback.format_exc()) 
else:
    st.warning("Selected mode not found or mapped.")
