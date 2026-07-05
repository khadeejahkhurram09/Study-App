"""
O-Level Study Hub 🎓📺 — Nixor AI + Cloud Course
================================================

A simple teach-and-study app for O-Level:

  • Teacher mode — upload a video lecture with notes for a subject.
  • Student mode — browse subjects, WATCH the lecture, read the notes, and study with
    an AI tutor: ask questions about the lecture, get a summary, or take an auto-quiz.

Designed to be easy to deploy on Streamlit and easy for students to use.

Dependencies: streamlit, openai, python-dotenv  (already in your sandbox).
Run with:  streamlit run olevel_study_hub.py --server.port 8501

NOTE on storage: uploaded videos are saved next to the app in ./lecture_data/. On the
class deploy (a Docker container) this resets when the app is redeployed — fine for a
demo. For permanent storage you'd use cloud blob storage (a good stretch task!).
NOTE on size: Streamlit caps uploads at 200 MB by default. To allow bigger videos, add a
file `.streamlit/config.toml` containing:  [server]\\n maxUploadSize = 1024
"""

import os
import re
import json
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --------------------------------------------------------------------------- #
# Storage + config
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).with_name("lecture_data")
VIDEO_DIR = DATA_DIR / "videos"
INDEX_FILE = DATA_DIR / "index.json"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
            "English", "Economics", "Accounting", "Islamiyat", "Pakistan Studies", "Urdu"]

VIDEO_TYPES = ["mp4", "mov", "webm", "m4v"]

# --------------------------------------------------------------------------- #
# Model registry — same wiring as the other course apps
# --------------------------------------------------------------------------- #
OPENAI_EP = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
FOUNDRY_EP = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
FOUNDRY_KEY = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

MODELS = {
    "GPT-5.5": (os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5"), OPENAI_EP, OPENAI_KEY),
    "DeepSeek-V4-Pro": (os.environ.get("MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT", "ds-v4pro"), FOUNDRY_EP, FOUNDRY_KEY),
    "Grok-4.3": (os.environ.get("MODEL_GROK43_DEPLOYMENT", "xai-grok43"), FOUNDRY_EP, FOUNDRY_KEY),
    "Mistral-Medium-3.5": (os.environ.get("MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT", "mstr-med35"), FOUNDRY_EP, FOUNDRY_KEY),
}


def ai_ready(model):
    return bool(MODELS[model][1] and MODELS[model][2])


# --------------------------------------------------------------------------- #
# Data helpers (no Streamlit)
# --------------------------------------------------------------------------- #
def load_index() -> list:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_index(items: list) -> None:
    INDEX_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:60]


def add_lecture(title, subject, description, notes, uploaded_file) -> None:
    lid = uuid.uuid4().hex[:10]
    ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
    fname = f"{lid}_{safe_name(Path(uploaded_file.name).stem)}{ext}"
    (VIDEO_DIR / fname).write_bytes(uploaded_file.getbuffer())
    items = load_index()
    items.append({
        "id": lid, "title": title.strip(), "subject": subject,
        "description": description.strip(), "notes": notes.strip(),
        "video": fname, "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_index(items)


def delete_lecture(lid: str) -> None:
    items = load_index()
    for it in items:
        if it["id"] == lid:
            try:
                (VIDEO_DIR / it["video"]).unlink(missing_ok=True)
            except OSError:
                pass
    save_index([it for it in items if it["id"] != lid])


def parse_json(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------------------- #
# AI helpers
# --------------------------------------------------------------------------- #
def _call(model, prompt, system, max_tokens=700):
    deployment, endpoint, key = MODELS[model]
    if not endpoint or not key:
        return {"ok": False, "text": f"⚠️ {model}: AI is not configured in this environment."}
    try:
        client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=API_VERSION)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=max_tokens,
        )
        return {"ok": True, "text": resp.choices[0].message.content or ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"⚠️ Could not reach {model}: {exc}"}


def _lecture_context(lec) -> str:
    return (f"Lecture: {lec['title']} ({lec['subject']})\n"
            f"Description: {lec.get('description','')}\n"
            f"Notes:\n{lec.get('notes','') or '(no notes provided)'}")


def ask_tutor(model, lec, question):
    system = ("You are a friendly O-Level tutor. Answer the student's question about this "
              "lecture using its notes. If the notes don't cover it, use your general "
              "O-Level knowledge but say so. Keep it clear and simple.")
    return _call(model, f"{_lecture_context(lec)}\n\nSTUDENT QUESTION: {question}",
                 system, max_tokens=600)["text"]


def summarize(model, lec):
    system = "You summarise lessons into clear revision notes for O-Level students."
    prompt = (f"{_lecture_context(lec)}\n\nWrite a revision summary: 5-7 key bullet points "
              "plus one 'exam tip'.")
    return _call(model, prompt, system, max_tokens=600)["text"]


def make_quiz(model, lec, n=5):
    system = "You write clear O-Level multiple-choice questions with one correct answer."
    prompt = (f"{_lecture_context(lec)}\n\nWrite {n} multiple-choice questions testing this "
              "lecture. Return ONLY JSON: "
              '{"questions":[{"q":"...","options":["a","b","c","d"],"answer_index":0,'
              '"explanation":"why"}]}')
    res = _call(model, prompt, system, max_tokens=1100)
    data = parse_json(res["text"]) if res["ok"] else None
    return data.get("questions") if isinstance(data, dict) else None


@st.cache_data(show_spinner=False)
def video_bytes(path_str, size):
    return Path(path_str).read_bytes()


# --------------------------------------------------------------------------- #
# UI — Teacher
# --------------------------------------------------------------------------- #
def teacher_view():
    st.subheader("👩‍🏫 Teacher — add a lecture")
    with st.form("upload", clear_on_submit=True):
        title = st.text_input("Lecture title", placeholder="e.g. Photosynthesis — Part 1")
        subject = st.selectbox("Subject", SUBJECTS)
        description = st.text_input("One-line description", placeholder="What is this lesson about?")
        notes = st.text_area("Lecture notes (the AI tutor & quiz use these)", height=180,
                             placeholder="Paste or write the key notes for this lecture…")
        video = st.file_uploader("Video lecture", type=VIDEO_TYPES)
        submitted = st.form_submit_button("⬆️ Upload lecture", type="primary")
        if submitted:
            if not title.strip() or video is None:
                st.warning("Please give a title and choose a video file.")
            else:
                with st.spinner("Saving…"):
                    add_lecture(title, subject, description, notes, video)
                st.success(f"Uploaded “{title.strip()}” to {subject}.")

    items = load_index()
    if items:
        st.markdown("#### Your lectures")
        for it in reversed(items):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{it['title']}** · {it['subject']}  \n"
                        f"<span style='opacity:.7'>{it.get('description','')} · "
                        f"uploaded {it.get('uploaded_at','')}</span>", unsafe_allow_html=True)
            if c2.button("🗑️ Delete", key=f"del_{it['id']}"):
                delete_lecture(it["id"])
                st.rerun()


# --------------------------------------------------------------------------- #
# UI — Student
# --------------------------------------------------------------------------- #
def _play_lecture(lec, model):
    st.markdown(f"### {lec['title']}")
    if lec.get("description"):
        st.caption(lec["description"])
    path = VIDEO_DIR / lec["video"]
    if path.exists():
        st.video(video_bytes(str(path), path.stat().st_size))
    else:
        st.error("Video file is missing (it may have been reset on redeploy).")

    tab_notes, tab_ask, tab_quiz = st.tabs(["📄 Notes & summary", "💬 Ask the tutor", "🧠 Quiz me"])

    with tab_notes:
        st.markdown(lec.get("notes") or "_No notes were added for this lecture._")
        if ai_ready(model) and lec.get("notes"):
            if st.button("✨ Summarise for revision", key=f"sum_{lec['id']}"):
                with st.spinner("Summarising…"):
                    st.session_state[f"summary_{lec['id']}"] = summarize(model, lec)
            if st.session_state.get(f"summary_{lec['id']}"):
                st.info(st.session_state[f"summary_{lec['id']}"])

    with tab_ask:
        if not ai_ready(model):
            st.info("The AI tutor isn't configured in this environment.")
        else:
            q = st.text_input("Ask anything about this lecture",
                              key=f"q_{lec['id']}", placeholder="e.g. Why is chlorophyll important?")
            if st.button("Ask", key=f"ask_{lec['id']}", type="primary") and q.strip():
                with st.spinner("Thinking…"):
                    st.session_state[f"ans_{lec['id']}"] = ask_tutor(model, lec, q)
            if st.session_state.get(f"ans_{lec['id']}"):
                st.markdown(st.session_state[f"ans_{lec['id']}"])

    with tab_quiz:
        if not ai_ready(model):
            st.info("Quizzes need the AI, which isn't configured here.")
        else:
            _quiz_ui(lec, model)


def _quiz_ui(lec, model):
    qkey = f"quiz_{lec['id']}"
    if st.button("🎯 Make me a quiz", key=f"mkquiz_{lec['id']}"):
        with st.spinner("Writing your quiz…"):
            st.session_state[qkey] = make_quiz(model, lec)
            st.session_state[f"{qkey}_submitted"] = False
    quiz = st.session_state.get(qkey)
    if not quiz:
        return
    answers = {}
    for i, item in enumerate(quiz):
        st.markdown(f"**Q{i+1}. {item['q']}**")
        answers[i] = st.radio("Pick one", item["options"], index=None,
                              key=f"{qkey}_{i}", label_visibility="collapsed")
    if st.button("Submit answers", key=f"{qkey}_submit", type="primary"):
        st.session_state[f"{qkey}_submitted"] = True
    if st.session_state.get(f"{qkey}_submitted"):
        correct = 0
        for i, item in enumerate(quiz):
            chosen = answers.get(i)
            right = item["options"][item["answer_index"]]
            ok = chosen == right
            correct += int(ok)
            st.markdown(("✅" if ok else "❌") + f" **Q{i+1}** — correct: *{right}*")
            st.caption("💡 " + item.get("explanation", ""))
        st.markdown(f"### Score: {correct}/{len(quiz)}")
        if correct == len(quiz):
            st.balloons()


def student_view(model):
    items = load_index()
    if not items:
        st.info("📭 No lectures yet. Ask your teacher to switch to **Teacher** mode and "
                "upload one!")
        return

    subjects = sorted({it["subject"] for it in items})
    subject = st.selectbox("📚 Choose a subject", subjects)
    in_subject = [it for it in items if it["subject"] == subject]

    titles = [it["title"] for it in in_subject]
    picked = st.selectbox("🎬 Choose a lecture", range(len(in_subject)),
                          format_func=lambda i: titles[i])
    st.divider()
    _play_lecture(in_subject[picked], model)


# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="O-Level Study Hub", page_icon="🎓", layout="wide")
    st.sidebar.title("🎓 O-Level Study Hub")
    role = st.sidebar.radio("I am a…", ["Student", "Teacher"], index=0)
    model = st.sidebar.selectbox("AI tutor model", list(MODELS.keys()))
    if not ai_ready(model):
        st.sidebar.warning("AI features are off (no key set) — video + notes still work.")
    st.sidebar.caption(f"{len(load_index())} lecture(s) available.")

    if role == "Teacher":
        st.title("👩‍🏫 Teacher dashboard")
        teacher_view()
    else:
        st.title("🎬 Study time!")
        st.caption("Pick a subject, watch the lecture, and study with your AI tutor.")
        student_view(model)


if __name__ == "__main__":
    main()
