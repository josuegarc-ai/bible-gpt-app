# ✅ Bible GPT — Restored + Enhanced with New Features
# Amazing working option v2.3 — Now with conversational chat, AI insight fixes, enhanced practice, mixed learning path, and 'Bible Beta' (AI voices + highlights)

import os
import openai
import requests
import json
import re
import random
from datetime import datetime
from duckduckgo_search import DDGS
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

# =============== SERMON SEARCH ===============
def search_sermons_online(passage):
    pastors = ["Philip Anthony Mitchell", "TD Jakes", "Tony Evans", "Mike Todd"]
    results = []
    ddgs = DDGS()
    for pastor in pastors:
        query = f"{pastor} sermon on {passage} site:youtube.com"
        try:
            search_results = ddgs.text(query, max_results=1)
            if search_results:
                url = search_results[0]['href']
                results.append({"pastor": pastor, "url": url})
            else:
                results.append({"pastor": pastor, "url": "❌ No result"})
        except Exception as e:
            results.append({"pastor": pastor, "url": f"❌ Error searching"})
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

def run_practice_chat():
    st.subheader("🧠 Practice Chat")

    if "practice_state" not in st.session_state:
        st.session_state.practice_state = {
            "questions": [],
            "current": 0,
            "score": 0,
            "book": "",
            "style": "",
            "level": "",
            "awaiting_next": False,
            "answered": False,
            "last_correct": False
        }

    state = st.session_state.practice_state

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
            for _ in range(num_questions):
                chosen_style = style if style != "mixed" else random.choice(["multiple choice", "fill in the blank", "true or false"])
                topic = book if book else "the Bible"
                q_prompt = f"Generate a {chosen_style} Bible question from {topic} suitable for a {level} learner, with 1 correct answer and 3 incorrect ones. Format as JSON with 'question', 'correct', 'choices'."
                response = ask_gpt_conversation(q_prompt)
                q_data = extract_json_from_response(response)
                if q_data:
                    all_choices = list(dict.fromkeys(q_data['choices']))  # deduplicate
                    if q_data['correct'] not in all_choices:
                        all_choices.append(q_data['correct'])
                    random.shuffle(all_choices)
                    q_data['choices'] = all_choices
                    state["questions"].append(q_data)
            st.experimental_rerun()

    elif state["current"] < len(state["questions"]):
        q_data = state["questions"][state["current"]]
        st.markdown(f"**Q{state['current']+1}: {q_data['question']}**")
        user_answer = st.radio("Choose:", q_data['choices'], key=f"q{state['current']}_choice")

        if not state["answered"]:
            if st.button("Submit Answer"):
                state["answered"] = True
                if user_answer.lower() == q_data['correct'].lower():
                    state["score"] += 1
                    state["last_correct"] = True
                else:
                    state["last_correct"] = False
        else:
            if state["last_correct"]:
                st.success("✅ Correct!")
                state["current"] += 1
                state["answered"] = False
                state["last_correct"] = False
                st.experimental_rerun()
            else:
                st.error(f"❌ Incorrect. Correct answer: {q_data['correct']}")
                explain_prompt = f"You're a theological Bible teacher. Explain why '{q_data['correct']}' is correct for: '{q_data['question']}', and briefly clarify why the other options are incorrect, using Scripture-based reasoning."
                explanation = ask_gpt_conversation(explain_prompt)
                st.markdown("**📜 Teaching Moment:**")
                st.write(explanation)
                if st.button("Next Question"):
                    state["current"] += 1
                    state["answered"] = False
                    state["last_correct"] = False
                    st.experimental_rerun()

    else:
        st.markdown(f"**🌞 Final Score: {state['score']}/{len(state['questions'])}**")
        if st.button("Restart Practice"):
            st.session_state.practice_state = {}
            st.experimental_rerun()
            
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

# =============== MAIN UI ===============
mode = st.sidebar.selectbox("Choose a mode:", [
    "Bible Lookup", "Chat with GPT", "Practice Chat", "Verse of the Day",
    "Study Plan", "Faith Journal", "Prayer Starter", "Fast Devotional",
    "Small Group Generator", "Tailored Learning Path", "Bible Beta Mode"
])

if mode == "Bible Lookup": run_bible_lookup()
elif mode == "Chat with GPT": run_chat_mode()
elif mode == "Practice Chat": run_practice_chat()
elif mode == "Faith Journal": run_faith_journal()
elif mode == "Tailored Learning Path": run_learning_path_mode()
elif mode == "Bible Beta Mode": run_bible_beta()
else: st.warning("This mode is under construction.")
