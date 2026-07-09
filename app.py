
import os
import re
import json
import uuid
import sqlite3
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
 
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from typing import Any

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = Any


DAILY_AI_QUOTA = 10
PAYMENT_LINK = os.environ.get("SCHOLARWAVE_PAYMENT_LINK", "https://buy.stripe.com/mock_link")
PAYMENT_CONTACT_EMAIL = os.environ.get("SCHOLARWAVE_PAYMENT_CONTACT_EMAIL", "billing@scholarwavehub.com")
PAYMENT_CONTACT_PHONE = os.environ.get("SCHOLARWAVE_PAYMENT_CONTACT_PHONE", "+880-000-000000")
PREMIUM_DAYS_DEFAULT = int(os.environ.get("SCHOLARWAVE_PREMIUM_DAYS", "30"))


def _current_quota_date():
    return datetime.now().date().isoformat()


def _resolve_ai_usage_identity(role=None):
    active_role = role or st.session_state.get("role") or "Student"
    if active_role == "Student" and st.session_state.get("student_id"):
        return active_role, f"student:{st.session_state.get('student_id')}"
    if active_role == "Teacher" and st.session_state.get("teacher_id"):
        return active_role, f"teacher:{st.session_state.get('teacher_id')}"

    if "anonymous_ai_session_id" not in st.session_state:
        st.session_state["anonymous_ai_session_id"] = uuid.uuid4().hex[:12]
    return active_role, f"session:{active_role.lower()}:{st.session_state['anonymous_ai_session_id']}"


def _today_date():
    return datetime.now().date()


def _parse_iso_date(value):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def is_user_premium(role=None):
    active_role = role or st.session_state.get("role") or "Student"
    row = None
    if active_role == "Student":
        sid = st.session_state.get("student_id")
        if not sid:
            return False
        if supabase:
            try:
                response = supabase.table("students").select("premium_until").eq("student_id", sid).limit(1).execute()
                records = response.data or []
                row = records[0] if records else None
            except Exception:
                row = None
        if row is None:
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT premium_until FROM students WHERE student_id = ?;", (sid,))
                sqlite_row = cursor.fetchone()
                row = {"premium_until": sqlite_row[0]} if sqlite_row else None
    else:
        tid = st.session_state.get("teacher_id")
        if not tid:
            return False
        if supabase:
            try:
                response = supabase.table("teachers").select("premium_until").eq("teacher_id", tid).limit(1).execute()
                records = response.data or []
                row = records[0] if records else None
            except Exception:
                row = None
        if row is None:
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT premium_until FROM teachers WHERE teacher_id = ?;", (tid,))
                sqlite_row = cursor.fetchone()
                row = {"premium_until": sqlite_row[0]} if sqlite_row else None

    premium_until = _parse_iso_date(row.get("premium_until") if row else None)
    return bool(premium_until and premium_until >= _today_date())


def get_user_premium_until(role=None):
    active_role = role or st.session_state.get("role") or "Student"
    row = None
    if active_role == "Student":
        sid = st.session_state.get("student_id")
        if not sid:
            return None
        if supabase:
            try:
                response = supabase.table("students").select("premium_until").eq("student_id", sid).limit(1).execute()
                records = response.data or []
                row = records[0] if records else None
            except Exception:
                row = None
        if row is None:
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT premium_until FROM students WHERE student_id = ?;", (sid,))
                sqlite_row = cursor.fetchone()
                row = {"premium_until": sqlite_row[0]} if sqlite_row else None
    else:
        tid = st.session_state.get("teacher_id")
        if not tid:
            return None
        if supabase:
            try:
                response = supabase.table("teachers").select("premium_until").eq("teacher_id", tid).limit(1).execute()
                records = response.data or []
                row = records[0] if records else None
            except Exception:
                row = None
        if row is None:
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT premium_until FROM teachers WHERE teacher_id = ?;", (tid,))
                sqlite_row = cursor.fetchone()
                row = {"premium_until": sqlite_row[0]} if sqlite_row else None

    return _parse_iso_date(row.get("premium_until") if row else None)


def get_ai_usage_count(role=None):
    active_role, user_identifier = _resolve_ai_usage_identity(role=role)
    usage_date = _current_quota_date()
    count = 0
    use_sqlite_fallback = not supabase

    if supabase:
        try:
            response = (
                supabase.table("user_ai_usage")
                .select("questions_asked")
                .eq("user_role", active_role)
                .eq("user_identifier", user_identifier)
                .eq("usage_date", usage_date)
                .limit(1)
                .execute()
            )
            records = response.data or []
            if records:
                count = int(records[0].get("questions_asked") or 0)
        except Exception:
            use_sqlite_fallback = True

    if use_sqlite_fallback:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT questions_asked FROM user_ai_usage WHERE user_role = ? AND user_identifier = ? AND usage_date = ?;",
                (active_role, user_identifier, usage_date),
            )
            row = cursor.fetchone()
            count = int(row[0]) if row else 0

    st.session_state["ai_questions_asked"] = count
    return count


def check_ai_quota(role=None, show_paywall=True):
    active_role = role or st.session_state.get("role") or "Student"
    if is_user_premium(active_role):
        return True
    used = get_ai_usage_count(role=active_role)
    if used >= DAILY_AI_QUOTA:
        if show_paywall:
            st.error("⚠️ You have reached your limit of 10 free AI queries for today!")
            col1, col2 = st.columns([2, 1.5])
            with col1:
                st.subheader("💳 Upgrade to ScholarWave Premium")
                st.write("Get unlimited AI questions, essay grading, and advanced cheat sheets for just $2/month.")
                st.link_button("🚀 Unlock Unlimited Access", PAYMENT_LINK)
            with col2:
                st.info("💡 Premium benefits unlock instant server response speeds and personalized O-Level study feedback loops.")
            st.caption(f"Role: {active_role} · Usage today: {used}/{DAILY_AI_QUOTA}")
            st.caption(f"Need help paying? Contact us at {PAYMENT_CONTACT_EMAIL} or {PAYMENT_CONTACT_PHONE}")
        return False
    return True


def increment_ai_usage(role=None):
    active_role, user_identifier = _resolve_ai_usage_identity(role=role)
    usage_date = _current_quota_date()
    count = 0
    use_sqlite_fallback = not supabase

    if supabase:
        try:
            current = (
                supabase.table("user_ai_usage")
                .select("usage_id, questions_asked")
                .eq("user_role", active_role)
                .eq("user_identifier", user_identifier)
                .eq("usage_date", usage_date)
                .limit(1)
                .execute()
            )
            records = current.data or []
            if records:
                current_row = records[0]
                count = int(current_row.get("questions_asked") or 0) + 1
                (
                    supabase.table("user_ai_usage")
                    .update({"questions_asked": count, "updated_at": datetime.now().isoformat(timespec="seconds")})
                    .eq("usage_id", current_row.get("usage_id"))
                    .execute()
                )
            else:
                insert_resp = (
                    supabase.table("user_ai_usage")
                    .insert({
                        "user_role": active_role,
                        "user_identifier": user_identifier,
                        "usage_date": usage_date,
                        "questions_asked": 1,
                    })
                    .execute()
                )
                inserted = insert_resp.data or []
                count = int(inserted[0].get("questions_asked") or 1) if inserted else 1
        except Exception:
            use_sqlite_fallback = True

    if use_sqlite_fallback:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_ai_usage (user_role, user_identifier, usage_date, questions_asked)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_role, user_identifier, usage_date)
                DO UPDATE SET questions_asked = questions_asked + 1, updated_at = CURRENT_TIMESTAMP;
                """,
                (active_role, user_identifier, usage_date),
            )
            cursor.execute(
                "SELECT questions_asked FROM user_ai_usage WHERE user_role = ? AND user_identifier = ? AND usage_date = ?;",
                (active_role, user_identifier, usage_date),
            )
            row = cursor.fetchone()
            conn.commit()
            count = int(row[0]) if row else 0

    st.session_state["ai_questions_asked"] = count
    return count


def increment_ai_quota(role=None):
    return increment_ai_usage(role=role)
 
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
# Only use supabase if keys are present; fallback gracefully when unavailable.
supabase: Client = create_client(supabase_url, supabase_key) if create_client and supabase_url and supabase_key else None
 
# --------------------------------------------------------------------------- #
# Storage + config
# --------------------------------------------------------------------------- #
# Set SCHOLARWAVE_DATA_DIR to a persistent folder in production.
DATA_DIR = Path(os.environ.get("SCHOLARWAVE_DATA_DIR", str(Path(__file__).with_name("lecture_data")))).expanduser()
VIDEO_DIR = DATA_DIR / "videos"
INDEX_FILE = DATA_DIR / "index.json"
DB_PATH = DATA_DIR / "school.db"          # <-- preference-system database lives alongside lecture data
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_RETENTION = int(os.environ.get("SCHOLARWAVE_BACKUP_RETENTION", "30"))
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _backup_data_files():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_targets = []
    if DB_PATH.exists():
        backup_targets.append((DB_PATH, BACKUP_DIR / f"school_{stamp}.db"))
    if INDEX_FILE.exists():
        backup_targets.append((INDEX_FILE, BACKUP_DIR / f"index_{stamp}.json"))

    for src, dst in backup_targets:
        try:
            shutil.copy2(src, dst)
        except OSError:
            # Never block startup if backup fails.
            continue

    # Keep only the newest N backup files.
    if BACKUP_RETENTION > 0:
        backups = sorted(BACKUP_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[BACKUP_RETENTION:]:
            try:
                old.unlink()
            except OSError:
                continue


def load_index():
    if not INDEX_FILE.exists():
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def safe_name(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def parse_json(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    # Direct parse first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Handle fenced blocks like ```json ... ```.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        fenced_text = fence.group(1).strip()
        try:
            return json.loads(fenced_text)
        except json.JSONDecodeError:
            pass

    # Fallback: try from first object/array bracket to the last one.
    start_candidates = [i for i in (raw.find("{"), raw.find("[")) if i != -1]
    end_candidates = [raw.rfind("}"), raw.rfind("]")]
    if start_candidates and max(end_candidates) != -1:
        start = min(start_candidates)
        end = max(end_candidates)
        if end > start:
            snippet = raw[start:end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def add_lecture(title, subject, description, notes, video_file, teacher_id=None, grade_level=None):
    if not title or video_file is None:
        return None

    video_name = save_uploaded_file(video_file, "videos", "lecture")
    if not video_name:
        return None
    video_path = Path(video_name)

    items = load_index()
    next_id = max([int(item.get("id", 0)) for item in items] + [0]) + 1
    lecture = {
        "id": next_id,
        "title": str(title).strip(),
        "subject": str(subject).strip(),
        "description": str(description or "").strip(),
        "notes": str(notes or "").strip(),
        "video": video_path.name,
        "teacher_id": teacher_id,
        "grade_level": str(grade_level or "").strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    items.append(lecture)

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    return lecture
 
SUBJECTS = [
    "Accounting", "Additional Mathematics", "Agriculture", "Arabic", "Art & Design",
    "Bangladesh Studies", "Bengali", "Biology", "Business Studies", "Chemistry",
    "Commerce", "Commercial Studies", "Combined Science", "Computer Science",
    "Design & Technology", "Economics", "English Language", "Environmental Management",
    "Fashion & Textiles", "Food & Nutrition", "French", "Geography", "German",
    "Global Perspectives", "Hinduism", "History", "Islamic Religion and Culture",
    "Islamiyat", "Literature in English", "Marine Science", "Mathematics (Syllabus D)",
    "Pakistan Studies", "Physics", "Second Language Urdu", "Sinhala", "Sociology",
    "Spanish", "Swahili", "Tamil", "First Language Urdu", "Classical Studies", "Law",
    "Psychology", "Thinking Skills", "Divinity", "Islamic Studies", "Further Mathematics",
    "Media Studies", "Digital Media & Design", "Global Perspectives & Research", "General Paper"
]

GRADE_LEVEL_OPTIONS = [
    "Kindergarten",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "OL/IGCSE",
    "AS",
    "A2",
]
DEFAULT_PERFORMANCE_THRESHOLD = 60
 
VIDEO_TYPES = ["mp4", "mov", "webm", "m4v"]
 
# --------------------------------------------------------------------------- #
# Model registry — Groq client wiring
# --------------------------------------------------------------------------- #
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODEL = "Groq"


def get_ai_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def ai_ready(model=None):
    return get_ai_client() is not None


def parse_score_value(value, total_marks):
    """Accept values like 12, '12', '12/20' and convert to a safe float score."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0

    parsed = float(m.group(1))
    if total_marks is None:
        return max(parsed, 0.0)
    return max(0.0, min(parsed, float(total_marks)))
 
 
# --------------------------------------------------------------------------- #
# AI helpers
# --------------------------------------------------------------------------- #
def _call(model, prompt, system, max_tokens=700):
    client = get_ai_client()
    if client is None:
        return {"ok": False, "text": "⚠️ AI is not configured. Set GROQ_API_KEY in your environment."}
    if not check_ai_quota():
        return {
            "ok": False,
            "text": (
                "⚠️ You have reached your limit of 10 free AI queries for today. "
                "Please upgrade to continue using unlimited AI features."
            ),
        }
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        increment_ai_quota()
        return {"ok": True, "text": resp.choices[0].message.content or ""}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "text": f"⚠️ Could not reach Groq: {exc}"}
 
 
def _lecture_context(lec) -> str:
    return (f"Lecture: {lec['title']} ({lec['subject']})\n"
            f"Description: {lec.get('description','')}\n"
            f"Notes:\n{lec.get('notes','') or '(no notes provided)'}")
 
 
def ask_tutor(model, lec, question):
    socratic_mode = st.session_state.get("socratic_mode", False)
    mode_instruction = (
        "Guide the student with a few thoughtful questions and hints rather than giving "
        "the answer immediately. Encourage them to reason it out, and only reveal the "
        "answer once they have tried."
        if socratic_mode else
        "Answer the student's question clearly and directly using the lecture notes."
    )
    system = (
        "You are a friendly O-Level tutor. " + mode_instruction +
        " If the notes don't cover it, use your general O-Level knowledge but say so. "
        "Keep it clear, supportive, and easy to follow."
    )
    return _call(model, f"{_lecture_context(lec)}\n\nSTUDENT QUESTION: {question}",
                 system, max_tokens=600)["text"]
 
 
def summarize(model, lec):
    system = "You summarise lessons into clear revision notes for O-Level students."
    prompt = (f"{_lecture_context(lec)}\n\nWrite a revision summary: 5-7 key bullet points "
              "plus one 'exam tip'.")
    return _call(model, prompt, system, max_tokens=600)["text"]
 
 
def make_quiz(model, lec, n=20):
    system = "You write clear, challenging O-Level multiple-choice questions with one correct answer."
    prompt = (f"{_lecture_context(lec)}\n\nWrite {n} multiple-choice questions testing this "
              "lecture. Make them exam-style and challenging. Return ONLY JSON: "
              '{"questions":[{"q":"...","options":["a","b","c","d"],"answer_index":0,'
              '"explanation":"why"}]}')
    res = _call(model, prompt, system, max_tokens=1100)
    data = parse_json(res["text"]) if res["ok"] else None
    return data.get("questions") if isinstance(data, dict) else None


def _quiz_bank_fallback(subject_name, count=10):
    subject_lower = (subject_name or "").lower()
    bank = []
    for q in QUIZ_BANK:
        q_subject = str(q.get("subject", "")).lower()
        if subject_lower in q_subject or q_subject in subject_lower:
            bank.append(q)
    if not bank:
        bank = QUIZ_BANK[:]

    questions = []
    for item in bank[:count]:
        options = item.get("options", [])
        if not options:
            continue
        questions.append({
            "q": str(item.get("q", "")).strip(),
            "options": [str(opt) for opt in options],
            "answer_index": int(item.get("answer_index", 0)),
            "explanation": str(item.get("explanation", "Review this concept from your notes.")),
        })
    return questions
 
 
def generate_welcome_note(model, name, subjects):
    """Personalized AI-concierge welcome note shown after sign-up — the app's signature touch."""
    system = ("You are an enthusiastic school orientation concierge who writes short, warm, "
              "personalized welcome notes for new O-Level students. No corporate tone, no "
              "generic filler — make it feel handwritten and specific.")
    prompt = (f"Student name: {name}\n"
              f"Subjects enrolled: {', '.join(subjects) if subjects else 'none yet'}\n\n"
              "Write a short welcome note (3-4 sentences): greet them by name, say something "
              "specific and encouraging about the *combination* of subjects they picked, and "
              "end with one light, motivating line about their first week.")
    res = _call(model, prompt, system, max_tokens=220)
    return res["text"] if res["ok"] else None


def generate_boss_battle_challenge(model, subject_name, grade_level, topic):
    # Get syllabus chapters for context
    chapters = get_syllabus_chapters(subject_name)
    syllabus_context = f"Syllabus chapters: {', '.join(chapters[:5])}" if chapters else ""

    system = "You create motivating mini study challenges for students using a playful but supportive tone. Return ONLY JSON with format: {\"questions\":[{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"correct_index\":0,\"hint\":\"...\",\"explanation\":\"...\"}],\"final_mission\":\"...\"}"
    prompt = (f"Create a short 'Boss Battle' challenge for a student in {grade_level or 'the chosen grade'} studying {subject_name}. "
              f"The topic is: {topic}. {syllabus_context} "
              f"Include 10 multiple-choice questions with options, hints, and explanations. Base questions on the syllabus content and topic, and keep them challenging. Also include a final mission. "
              "Keep it concise and motivating.")
    res = _call(model, prompt, system, max_tokens=1800)
    data = parse_json(res["text"]) if res.get("ok") else None

    # Normalize/validate AI response before using it.
    if isinstance(data, dict):
        normalized_questions = []
        for item in data.get("questions", []):
            if not isinstance(item, dict):
                continue
            options = item.get("options") or []
            if not isinstance(options, list) or len(options) < 2:
                continue
            try:
                correct_index = int(item.get("correct_index", 0))
            except (TypeError, ValueError):
                correct_index = 0
            if correct_index < 0 or correct_index >= len(options):
                correct_index = 0
            normalized_questions.append({
                "question": str(item.get("question") or "").strip(),
                "options": [str(opt) for opt in options],
                "correct_index": correct_index,
                "hint": str(item.get("hint") or "Think about the core concept and eliminate unlikely options."),
                "explanation": str(item.get("explanation") or "Review the related topic notes and key definitions."),
            })

        if normalized_questions:
            return {
                "questions": normalized_questions,
                "final_mission": str(data.get("final_mission") or "Great effort. Revisit weak spots and try again for a perfect score!"),
            }

    # Fallback: keep Boss Battle usable even if AI JSON is malformed or quota is exhausted.
    bank = []
    subject_lower = (subject_name or "").lower()
    for q in QUIZ_BANK:
        q_subject = str(q.get("subject", "")).lower()
        if subject_lower in q_subject or q_subject in subject_lower:
            bank.append(q)
    if not bank:
        bank = QUIZ_BANK[:]

    fallback_questions = []
    for q in bank[:10]:
        options = q.get("options", [])
        if not options:
            continue
        fallback_questions.append({
            "question": q.get("q", ""),
            "options": options,
            "correct_index": int(q.get("answer_index", 0)),
            "hint": "Use elimination and recall the key definition/formula.",
            "explanation": q.get("explanation", "Review this concept from your notes."),
        })

    fallback_message = (
        "AI challenge generation is temporarily unavailable, so we loaded a practice Boss Battle from your local question bank."
    )
    if res and not res.get("ok") and res.get("text"):
        fallback_message = res["text"]

    return {
        "questions": fallback_questions,
        "final_mission": fallback_message,
    }


def simplify_for_eli5(model, subject_name, topic, explanation):
    system = "You rewrite school topics in simple, friendly language that a curious beginner would understand."
    prompt = (f"Subject: {subject_name}\nTopic: {topic}\nOriginal explanation: {explanation}\n\n"
              "Rewrite this in a clear ELI5 style using short sentences and a warm, encouraging tone.")
    res = _call(model, prompt, system, max_tokens=700)
    return res["text"] if res.get("ok") else "ELI5 simplification is not available right now."


def connect_concepts(model, subject_name, topic_a, topic_b):
    system = "You connect academic ideas into a clear concept map. Return ONLY JSON with format: {\"central_topic\":\"...\",\"connections\":[{\"from\":\"...\",\"to\":\"...\",\"relationship\":\"...\"}],\"key_points\":[\"...\"]}"
    prompt = (f"Subject: {subject_name}\nTopic A: {topic_a}\nTopic B: {topic_b}\n\n"
              "Create a visual concept map showing how these topics connect. Include 3-4 key connections with relationship descriptions and 3 key points.")
    res = _call(model, prompt, system, max_tokens=900)
    data = parse_json(res["text"]) if res.get("ok") else None
    return data if data else {"central_topic": topic_a, "connections": [], "key_points": ["Concept mapping is not available right now."]}


def generate_cheat_sheet(model, subject_name, topic):
    system = "You make compact, high-value revision cheat sheets for students."
    chapters = get_syllabus_chapters(subject_name)
    syllabus_lines = "\n".join([f"- {ch}" for ch in chapters]) if chapters else "- No chapter list provided"
    prompt = (
        f"Subject: {subject_name}\n"
        f"Focus topic: {topic}\n"
        f"Syllabus chapters:\n{syllabus_lines}\n\n"
        "Create a comprehensive cheat sheet that covers the full syllabus criteria in concise form, "
        "making sure no chapter is missed. Include chapter-wise key points, likely exam traps, "
        "and a fast final-revision checklist."
    )
    res = _call(model, prompt, system, max_tokens=1200)
    return res["text"] if res.get("ok") else "Cheat-sheet generation is not available right now."


def build_intervention_plan(average_percentage, threshold):
    if average_percentage >= threshold:
        return []
    return [{
        "type": "past_papers",
        "message": (
            f"Your average is {average_percentage}% — below your target of {threshold}%. "
            "Try 5 past papers and a guided revision sprint this week."
        ),
    }]

# --------------------------------------------------------------------------- #
# Preference-system database
# --------------------------------------------------------------------------- #
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
 
 
def create_tables():
    with get_conn() as conn:
        cursor = conn.cursor()
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            grade_level TEXT,
            email TEXT,
            password_hash TEXT,
            premium_until TEXT,
            performance_threshold INTEGER DEFAULT 60,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_ai_usage (
            usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_role TEXT NOT NULL,
            user_identifier TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            questions_asked INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (user_role, user_identifier, usage_date)
        );
        """)
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT UNIQUE NOT NULL,
            subject_code TEXT UNIQUE
        );
        """)
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT,
            grade_level TEXT,
            max_students INTEGER DEFAULT 30,
            username TEXT UNIQUE,
            password_hash TEXT,
            premium_until TEXT
        );
        """)
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            teacher_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            grade_level TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (teacher_id, subject_id, grade_level),
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
        );
        """)
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            preferred_teacher_id INTEGER,
            priority INTEGER DEFAULT 1,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
            FOREIGN KEY (preferred_teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL,
            UNIQUE (student_id, subject_id, priority)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS suggestions (
            suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_role TEXT,
            user_name TEXT,
            user_email TEXT,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS payment_requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_role TEXT,
            user_identifier TEXT,
            user_name TEXT,
            user_email TEXT,
            reference TEXT,
            note TEXT,
            status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            reviewed_note TEXT,
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
    ensure_profile_columns()


def ensure_profile_columns():
    with get_conn() as conn:
        cursor = conn.cursor()
        for statement in [
            "ALTER TABLE students ADD COLUMN password_hash TEXT",
            "ALTER TABLE students ADD COLUMN premium_until TEXT",
            "ALTER TABLE students ADD COLUMN performance_threshold INTEGER DEFAULT 60",
            "ALTER TABLE teachers ADD COLUMN grade_level TEXT",
            "ALTER TABLE teachers ADD COLUMN username TEXT",
            "ALTER TABLE teachers ADD COLUMN password_hash TEXT",
            "ALTER TABLE teachers ADD COLUMN premium_until TEXT",
            "ALTER TABLE teacher_subjects ADD COLUMN grade_level TEXT DEFAULT ''",
            "ALTER TABLE flashcards ADD COLUMN grade_level TEXT",
            "ALTER TABLE syllabus_chapters ADD COLUMN grade_level TEXT DEFAULT ''",
            "ALTER TABLE syllabus_documents ADD COLUMN grade_level TEXT",
            "ALTER TABLE assessments ADD COLUMN exam_duration_minutes INTEGER DEFAULT 60",
            "ALTER TABLE past_papers ADD COLUMN grade_level TEXT",
            "ALTER TABLE past_papers ADD COLUMN duration_minutes INTEGER DEFAULT 60",
            "ALTER TABLE assessment_submissions ADD COLUMN completed_in_time INTEGER DEFAULT 1",
            "ALTER TABLE assessment_submissions ADD COLUMN answer_file_path TEXT",
            "ALTER TABLE assessment_submissions ADD COLUMN self_check_enabled INTEGER DEFAULT 0",
            "ALTER TABLE assessment_submissions ADD COLUMN teacher_check_enabled INTEGER DEFAULT 0",
            "ALTER TABLE assessment_submissions ADD COLUMN ai_check_enabled INTEGER DEFAULT 0",
            "ALTER TABLE payment_requests ADD COLUMN user_identifier TEXT",
            "ALTER TABLE payment_requests ADD COLUMN reviewed_by TEXT",
            "ALTER TABLE payment_requests ADD COLUMN reviewed_note TEXT",
            "ALTER TABLE payment_requests ADD COLUMN reviewed_at TIMESTAMP",
        ]:
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.commit()


def seed_subjects():
    """Keep the `subjects` table in sync with the app's SUBJECTS list."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO subjects (subject_name) VALUES (?);",
            [(s,) for s in SUBJECTS]
        )
        conn.commit()


def create_syllabus_table():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllabus_chapters (
            chapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            grade_level TEXT DEFAULT '',
            chapter_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (subject_name, grade_level, chapter_name)
        );
        """)
        conn.commit()


def seed_syllabus_chapters():
    default_chapters = {
        "Mathematics (Syllabus D)": ["Algebra and equations", "Geometry and trigonometry", "Mensuration", "Probability and statistics"],
        "Physics": ["Forces and motion", "Electricity", "Waves", "Thermal physics"],
        "Chemistry": ["Atomic structure", "Chemical bonding", "Acids and bases", "Organic chemistry"],
        "Biology": ["Cell structure and transport", "Nutrition", "Reproduction", "Genetics"],
        "Computer Science": ["Algorithms", "Programming fundamentals", "Computer systems", "Data representation"],
        "English Language": ["Comprehension skills", "Grammar", "Writing techniques", "Summary writing"],
        "Economics": ["Basic economic problems", "Demand and supply", "Market structures", "International trade"],
        "Accounting": ["Accounting principles", "Double entry", "Trial balance", "Financial statements"],
        "Business Studies": ["Business activity", "Marketing", "Finance", "Human resources"],
        "Additional Mathematics": ["Functions", "Calculus", "Vectors", "Complex numbers"],
        "Islamiyat": ["Quranic verses", "Hadith", "Beliefs", "Islamic history"],
        "Pakistan Studies": ["Geography", "History", "Government", "Culture"],
        "First Language Urdu": ["Grammar", "Comprehension", "Composition", "Literature"],
        "Second Language Urdu": ["Basic grammar", "Reading comprehension", "Writing skills", "Vocabulary"],
        "Geography": ["Population", "Settlement", "Natural environment", "Economic development"],
        "History": ["Modern world history", "Regional studies", "Source analysis", "Historical interpretation"],
    }
    with get_conn() as conn:
        cursor = conn.cursor()
        for subject_name, chapters in default_chapters.items():
            for chapter_name in chapters:
                cursor.execute(
                    "INSERT OR IGNORE INTO syllabus_chapters (subject_name, chapter_name) VALUES (?, ?);",
                    (subject_name, chapter_name),
                )
        conn.commit()


def get_syllabus_chapters(subject_name, grade_level=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                "SELECT chapter_name FROM syllabus_chapters WHERE subject_name = ? AND (grade_level = ? OR grade_level = '') ORDER BY chapter_name;",
                (subject_name, grade_level),
            )
        else:
            cursor.execute(
                "SELECT chapter_name FROM syllabus_chapters WHERE subject_name = ? ORDER BY chapter_name;",
                (subject_name,),
            )
        rows = cursor.fetchall()
    return [row[0] for row in rows]


def add_syllabus_chapter(subject_name, chapter_name, grade_level=None):
    if not subject_name or not chapter_name:
        return False
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO syllabus_chapters (subject_name, grade_level, chapter_name) VALUES (?, ?, ?);",
            (subject_name.strip(), (grade_level or "").strip(), chapter_name.strip()),
        )
        conn.commit()
    return True


def delete_syllabus_chapter(subject_name, chapter_name, grade_level=None):
    if not subject_name or not chapter_name:
        return False
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                "DELETE FROM syllabus_chapters WHERE subject_name = ? AND chapter_name = ? AND (grade_level = ? OR grade_level = '');",
                (subject_name.strip(), chapter_name.strip(), grade_level.strip()),
            )
        else:
            cursor.execute(
                "DELETE FROM syllabus_chapters WHERE subject_name = ? AND chapter_name = ?;",
                (subject_name.strip(), chapter_name.strip()),
            )
        conn.commit()
    return True


def create_teacher_questions_table():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            student_name TEXT,
            subject_name TEXT NOT NULL,
            teacher_id INTEGER,
            question_text TEXT NOT NULL,
            answer_text TEXT,
            answered_by TEXT,
            answered_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("PRAGMA table_info(teacher_questions);")
        columns = {row[1] for row in cursor.fetchall()}
        if "answer_text" not in columns:
            cursor.execute("ALTER TABLE teacher_questions ADD COLUMN answer_text TEXT;")
        if "answered_by" not in columns:
            cursor.execute("ALTER TABLE teacher_questions ADD COLUMN answered_by TEXT;")
        if "answered_at" not in columns:
            cursor.execute("ALTER TABLE teacher_questions ADD COLUMN answered_at TIMESTAMP;")
        conn.commit()


def save_teacher_question(student_id, subject_name, teacher_id, question_text, student_name=None):
    if not subject_name or not question_text:
        return False
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO teacher_questions (student_id, student_name, subject_name, teacher_id, question_text) VALUES (?, ?, ?, ?, ?);",
            (student_id, student_name or "Student", subject_name.strip(), teacher_id, question_text.strip()),
        )
        conn.commit()
    return True


def save_teacher_question_answer(question_id, answer_text, answered_by=None):
    if not question_id or not answer_text:
        return False
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE teacher_questions SET answer_text = ?, answered_by = ?, answered_at = CURRENT_TIMESTAMP WHERE question_id = ?;",
            (answer_text.strip(), answered_by, question_id),
        )
        conn.commit()
    return True


def get_teacher_questions(teacher_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT question_id, student_name, subject_name, question_text, answer_text, answered_by, answered_at, created_at
        FROM teacher_questions
        WHERE teacher_id IS NULL OR teacher_id = ?
        ORDER BY created_at DESC;
    """, (teacher_id,))
        rows = cursor.fetchall()
    return rows


def get_student_teacher_questions(student_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT question_id, subject_name, question_text, answer_text, answered_by, answered_at, created_at
        FROM teacher_questions
        WHERE student_id = ?
        ORDER BY created_at DESC;
        """, (student_id,))
        rows = cursor.fetchall()
    return rows


def save_suggestion(user_role, user_name, user_email, category, message):
    if not category or not message.strip():
        return False
    payload = {
        "user_role": user_role,
        "user_name": user_name,
        "user_email": user_email,
        "category": category,
        "message": message.strip(),
    }

    if supabase:
        try:
            supabase.table("suggestions").insert(payload).execute()
            return True
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO suggestions (user_role, user_name, user_email, category, message) VALUES (?, ?, ?, ?, ?);",
            (user_role, user_name, user_email, category, message.strip()),
        )
        conn.commit()
    return True


def get_recent_suggestions(limit=50):
    if supabase:
        try:
            response = (
                supabase.table("suggestions")
                .select("suggestion_id, user_role, user_name, user_email, category, message, created_at")
                .order("created_at", desc=True)
                .limit(int(limit))
                .execute()
            )
            return response.data or []
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT suggestion_id, user_role, user_name, user_email, category, message, created_at FROM suggestions ORDER BY created_at DESC LIMIT ?;",
            (int(limit),),
        )
        rows = cursor.fetchall()
    normalized = []
    for row in rows:
        normalized.append({
            "suggestion_id": row[0],
            "user_role": row[1],
            "user_name": row[2],
            "user_email": row[3],
            "category": row[4],
            "message": row[5],
            "created_at": row[6],
        })
    return normalized


def save_payment_request(user_role, user_identifier, user_name, user_email, reference, note):
    payload = {
        "user_role": user_role,
        "user_identifier": user_identifier,
        "user_name": user_name,
        "user_email": user_email,
        "reference": (reference or "").strip(),
        "note": (note or "").strip(),
        "status": "pending",
    }

    if supabase:
        try:
            supabase.table("payment_requests").insert(payload).execute()
            return True
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO payment_requests (user_role, user_identifier, user_name, user_email, reference, note) VALUES (?, ?, ?, ?, ?, ?);",
            (user_role, user_identifier, user_name, user_email, (reference or "").strip(), (note or "").strip()),
        )
        conn.commit()
    return True


def get_payment_requests(limit=50):
    if supabase:
        try:
            response = (
                supabase.table("payment_requests")
                .select("request_id, user_role, user_identifier, user_name, user_email, reference, note, status, reviewed_by, reviewed_note, reviewed_at, created_at")
                .order("created_at", desc=True)
                .limit(int(limit))
                .execute()
            )
            return response.data or []
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT request_id, user_role, user_identifier, user_name, user_email, reference, note, status, reviewed_by, reviewed_note, reviewed_at, created_at FROM payment_requests ORDER BY created_at DESC LIMIT ?;",
            (int(limit),),
        )
        rows = cursor.fetchall()
    normalized = []
    for row in rows:
        normalized.append({
            "request_id": row[0],
            "user_role": row[1],
            "user_identifier": row[2],
            "user_name": row[3],
            "user_email": row[4],
            "reference": row[5],
            "note": row[6],
            "status": row[7],
            "reviewed_by": row[8],
            "reviewed_note": row[9],
            "reviewed_at": row[10],
            "created_at": row[11],
        })
    return normalized


def update_payment_request_status(request_id, status, reviewed_by=None, reviewed_note=None):
    if supabase:
        try:
            (
                supabase.table("payment_requests")
                .update({
                    "status": status,
                    "reviewed_by": reviewed_by,
                    "reviewed_note": reviewed_note,
                    "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                })
                .eq("request_id", request_id)
                .execute()
            )
            return True
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE payment_requests
            SET status = ?, reviewed_by = ?, reviewed_note = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE request_id = ?;
            """,
            (status, reviewed_by, reviewed_note, request_id),
        )
        conn.commit()
    return True


def grant_premium_access(user_role, user_identifier=None, user_email=None, days=PREMIUM_DAYS_DEFAULT):
    duration = max(int(days or PREMIUM_DAYS_DEFAULT), 1)
    today = _today_date()
    if supabase:
        try:
            if user_role == "Student":
                student_id = None
                if user_identifier and str(user_identifier).startswith("student:"):
                    try:
                        student_id = int(str(user_identifier).split(":", 1)[1])
                    except ValueError:
                        student_id = None

                row = None
                if student_id is not None:
                    response = supabase.table("students").select("student_id, premium_until").eq("student_id", student_id).limit(1).execute()
                    records = response.data or []
                    row = records[0] if records else None
                elif user_email:
                    response = supabase.table("students").select("student_id, premium_until").eq("email", user_email).order("student_id", desc=True).limit(1).execute()
                    records = response.data or []
                    row = records[0] if records else None

                if not row:
                    return False, "Student account not found."

                current_until = _parse_iso_date(row.get("premium_until"))
                base_date = current_until if current_until and current_until > today else today
                new_until = base_date + timedelta(days=duration)
                supabase.table("students").update({"premium_until": new_until.isoformat()}).eq("student_id", row.get("student_id")).execute()
            else:
                teacher_id = None
                if user_identifier and str(user_identifier).startswith("teacher:"):
                    try:
                        teacher_id = int(str(user_identifier).split(":", 1)[1])
                    except ValueError:
                        teacher_id = None

                row = None
                if teacher_id is not None:
                    response = supabase.table("teachers").select("teacher_id, premium_until").eq("teacher_id", teacher_id).limit(1).execute()
                    records = response.data or []
                    row = records[0] if records else None
                elif user_email:
                    response = supabase.table("teachers").select("teacher_id, premium_until").eq("email", user_email).order("teacher_id", desc=True).limit(1).execute()
                    records = response.data or []
                    row = records[0] if records else None

                if not row:
                    return False, "Teacher account not found."

                current_until = _parse_iso_date(row.get("premium_until"))
                base_date = current_until if current_until and current_until > today else today
                new_until = base_date + timedelta(days=duration)
                supabase.table("teachers").update({"premium_until": new_until.isoformat()}).eq("teacher_id", row.get("teacher_id")).execute()

            return True, f"Premium active until {new_until.isoformat()}"
        except Exception:
            pass

    with get_conn() as conn:
        cursor = conn.cursor()
        if user_role == "Student":
            student_id = None
            if user_identifier and str(user_identifier).startswith("student:"):
                try:
                    student_id = int(str(user_identifier).split(":", 1)[1])
                except ValueError:
                    student_id = None
            if student_id is None and user_email:
                cursor.execute("SELECT student_id FROM students WHERE email = ? ORDER BY student_id DESC LIMIT 1;", (user_email,))
                row = cursor.fetchone()
                student_id = row[0] if row else None
            if student_id is None:
                return False, "Student account not found."

            cursor.execute("SELECT premium_until FROM students WHERE student_id = ?;", (student_id,))
            row = cursor.fetchone()
            current_until = _parse_iso_date(row[0]) if row else None
            base_date = current_until if current_until and current_until > today else today
            new_until = base_date + timedelta(days=duration)
            cursor.execute("UPDATE students SET premium_until = ? WHERE student_id = ?;", (new_until.isoformat(), student_id))
        else:
            teacher_id = None
            if user_identifier and str(user_identifier).startswith("teacher:"):
                try:
                    teacher_id = int(str(user_identifier).split(":", 1)[1])
                except ValueError:
                    teacher_id = None
            if teacher_id is None and user_email:
                cursor.execute("SELECT teacher_id FROM teachers WHERE email = ? ORDER BY teacher_id DESC LIMIT 1;", (user_email,))
                row = cursor.fetchone()
                teacher_id = row[0] if row else None
            if teacher_id is None:
                return False, "Teacher account not found."

            cursor.execute("SELECT premium_until FROM teachers WHERE teacher_id = ?;", (teacher_id,))
            row = cursor.fetchone()
            current_until = _parse_iso_date(row[0]) if row else None
            base_date = current_until if current_until and current_until > today else today
            new_until = base_date + timedelta(days=duration)
            cursor.execute("UPDATE teachers SET premium_until = ? WHERE teacher_id = ?;", (new_until.isoformat(), teacher_id))

        conn.commit()
    return True, f"Premium active until {new_until.isoformat()}"


def _current_user_profile():
    role = st.session_state.get("role") or "Student"
    if role == "Teacher":
        teacher_id = st.session_state.get("teacher_id")
        if teacher_id:
            profile = get_teacher_profile(teacher_id)
            if profile:
                return role, f"teacher:{teacher_id}", profile[1], profile[2] or ""
        return role, None, st.session_state.get("teacher_name", "Teacher"), ""

    student_id = st.session_state.get("student_id")
    if student_id:
        row = get_student_by_id(student_id)
        if row:
            return role, f"student:{student_id}", row[1], row[4] or ""
    return role, None, st.session_state.get("student_name", "Student"), ""


def _render_corner_suggestion_box():
    st.markdown(
        """
        <style>
        .sw-corner-feedback {
            position: fixed;
            right: 18px;
            bottom: 18px;
            z-index: 9999;
            width: 320px;
            max-width: calc(100vw - 30px);
            border-radius: 14px;
            border: 1px solid #d8e7f2;
            background: #ffffff;
            box-shadow: 0 14px 28px rgba(0,0,0,.16);
            padding: .75rem .85rem;
        }
        .sw-corner-feedback h4 {
            margin: 0 0 .35rem;
            color: #0f4f73;
            font-size: .96rem;
        }
        .sw-corner-feedback p {
            margin: 0;
            color: #2e3d4f;
            font-size: .82rem;
            line-height: 1.25;
        }
        </style>
        <div class="sw-corner-feedback">
            <h4>💬 Suggestion Corner</h4>
            <p>Have an idea, complaint, or improvement? Open the <b>Feedback & Payments</b> panel in the sidebar and send it in 30 seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feedback_and_payment_panel():
    role, user_identifier, name, email = _current_user_profile()
    with st.sidebar.expander("💬 Feedback & Payments", expanded=False):
        st.caption("Share suggestions, complaints, and payment details.")

        with st.form("feedback_form", clear_on_submit=True):
            category = st.selectbox("Type", ["Suggestion", "Improvement", "Complaint", "Bug report"])
            message = st.text_area("Your feedback", height=110, placeholder="Tell us what to improve…")
            submitted_feedback = st.form_submit_button("Send feedback", type="primary")
            if submitted_feedback:
                if not message.strip():
                    st.warning("Please write your feedback first.")
                else:
                    save_suggestion(role, name, email, category, message)
                    st.success("Thanks. Your feedback was saved.")

        st.divider()
        st.markdown("**💳 Premium payment**")
        st.write("Use the secure payment link below. After payment, submit your reference so we can activate your premium access.")
        st.link_button("Pay for Premium", PAYMENT_LINK)
        st.caption(f"Support contact: {PAYMENT_CONTACT_EMAIL} · {PAYMENT_CONTACT_PHONE}")

        with st.form("payment_request_form", clear_on_submit=True):
            reference = st.text_input("Payment reference / transaction ID")
            note = st.text_area("Optional note", height=80, placeholder="e.g. Paid for Student account roll 2026005")
            submitted_payment = st.form_submit_button("I have paid", type="primary")
            if submitted_payment:
                if not reference.strip():
                    st.warning("Please add a transaction reference so we can verify your payment.")
                else:
                    save_payment_request(role, user_identifier, name, email, reference, note)
                    st.success("Payment request received. We will verify and activate soon.")
 
 
def get_student_by_roll(roll_number):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, full_name, grade_level, email FROM students WHERE roll_number = ?;",
                       (roll_number,))
        row = cursor.fetchone()
    return row  # (student_id, full_name, grade_level, email) or None


def authenticate_student(roll_number, password):
    """Verify student credentials (roll number + password)."""
    if not roll_number or not password:
        return None
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, full_name, grade_level, email FROM students WHERE roll_number = ? AND password_hash = ?;",
            (roll_number, password_hash),
        )
        row = cursor.fetchone()
    return row if row else None
 
 
def get_student_by_id(student_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, full_name, roll_number, grade_level, email FROM students WHERE student_id = ?;",
                       (student_id,))
        row = cursor.fetchone()
    return row  # (student_id, full_name, roll_number, grade_level, email) or None
 
 
def register_student(full_name, roll_number, grade_level, email, password):
    """Add a new student with an auto-generated roll number.

    Roll format is YYYYNNN, where NNN is the registration sequence for that year.
    Returns (student_id, roll_number) on success, or (None, None) on failure.
    """
    if not password:
        return None, None

    current_year = datetime.now().year
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    with get_conn() as conn:
        cursor = conn.cursor()

        # Retry a few times in case of concurrent registrations creating the same sequence.
        for _ in range(5):
            cursor.execute(
                "SELECT COUNT(*) FROM students WHERE roll_number LIKE ?;",
                (f"{current_year}%",),
            )
            current_count = int(cursor.fetchone()[0] or 0)
            next_roll = f"{current_year}{current_count + 1:03d}"

            try:
                cursor.execute(
                    "INSERT INTO students (full_name, roll_number, grade_level, email, password_hash) VALUES (?, ?, ?, ?, ?);",
                    (full_name, next_roll, grade_level, email, password_hash)
                )
                conn.commit()
                return cursor.lastrowid, next_roll
            except sqlite3.IntegrityError:
                # If roll_number collided, loop and try the next available sequence.
                continue

        conn.rollback()
    return None, None
 
 
def register_teacher_for_subjects(full_name, email, subject_names, grade_level=None, username=None, password=None, teacher_id=None):
    """Create the teacher if needed, then link them to each chosen subject."""
    grade_levels = []
    if isinstance(grade_level, list):
        grade_levels = [g for g in grade_level if g]
    elif isinstance(grade_level, str) and grade_level.strip():
        grade_levels = [grade_level.strip()]
    if not grade_levels:
        grade_levels = [""]
    grade_level_text = ", ".join([g for g in grade_levels if g]) or None

    with get_conn() as conn:
        cursor = conn.cursor()

        row = None
        if teacher_id:
            cursor.execute("SELECT teacher_id FROM teachers WHERE teacher_id = ?;", (teacher_id,))
            row = cursor.fetchone()
        if row is None:
            cursor.execute("SELECT teacher_id FROM teachers WHERE full_name = ?;", (full_name,))
            row = cursor.fetchone()
        if row:
            teacher_id = row[0]
            cursor.execute("UPDATE teachers SET email = ?, grade_level = ? WHERE teacher_id = ?;",
                       (email, grade_level_text, teacher_id))
            if username and password:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute("UPDATE teachers SET username = ?, password_hash = ? WHERE teacher_id = ?;",
                               (username, password_hash, teacher_id))
        else:
            password_hash = hashlib.sha256(password.encode()).hexdigest() if password else None
            cursor.execute("INSERT INTO teachers (full_name, email, grade_level, username, password_hash) VALUES (?, ?, ?, ?, ?);",
                           (full_name, email, grade_level_text, username, password_hash))
            teacher_id = cursor.lastrowid

        cursor.execute("DELETE FROM teacher_subjects WHERE teacher_id = ?;", (teacher_id,))
        for subject_name in subject_names:
            cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
            srow = cursor.fetchone()
            if srow:
                for g in grade_levels:
                    cursor.execute(
                        "INSERT OR IGNORE INTO teacher_subjects (teacher_id, subject_id, grade_level) VALUES (?, ?, ?);",
                        (teacher_id, srow[0], g)
                    )

        conn.commit()
    return teacher_id


def get_teacher_profile(teacher_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT teacher_id, full_name, email, grade_level FROM teachers WHERE teacher_id = ?;",
            (teacher_id,),
        )
        row = cursor.fetchone()
    return row


def get_teacher_subject_links(teacher_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.subject_name, ts.grade_level
            FROM teacher_subjects ts
            JOIN subjects s ON s.subject_id = ts.subject_id
            WHERE ts.teacher_id = ?
            ORDER BY s.subject_name;
            """,
            (teacher_id,),
        )
        rows = cursor.fetchall()
    return rows


def authenticate_teacher(username, password):
    """Verify teacher credentials."""
    if not username or not password:
        return None
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT teacher_id, full_name FROM teachers WHERE username = ? AND password_hash = ?;",
                       (username, password_hash))
        row = cursor.fetchone()
    return row if row else None
 
 
def get_teachers_for_subject(subject_name, grade_level=None):
    """List teachers who teach a given subject."""
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute("""
        SELECT DISTINCT t.teacher_id, t.full_name
        FROM teachers t
        JOIN teacher_subjects ts ON ts.teacher_id = t.teacher_id
        JOIN subjects s ON s.subject_id = ts.subject_id
        WHERE s.subject_name = ? AND (ts.grade_level = ? OR ts.grade_level = '')
        ORDER BY t.full_name;
    """, (subject_name, grade_level))
        else:
            cursor.execute("""
        SELECT t.teacher_id, t.full_name
        FROM teachers t
        JOIN teacher_subjects ts ON ts.teacher_id = t.teacher_id
        JOIN subjects s ON s.subject_id = ts.subject_id
        WHERE s.subject_name = ?
        ORDER BY t.full_name;
    """, (subject_name,))
        rows = cursor.fetchall()
    return rows  # list of (teacher_id, full_name)
 
 
def submit_preference(student_id, subject_name, preferred_teacher_id=None, priority=1):
    """Record (or update) a student's subject + preferred teacher choice."""
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = ?;", (subject_name,))
        row = cursor.fetchone()
        if row is None:
            return False, f"Subject '{subject_name}' not found."
        subject_id = row[0]

        try:
            cursor.execute("""
                INSERT INTO student_preferences (student_id, subject_id, preferred_teacher_id, priority)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(student_id, subject_id, priority)
                DO UPDATE SET preferred_teacher_id = excluded.preferred_teacher_id,
                              submitted_at = CURRENT_TIMESTAMP;
            """, (student_id, subject_id, preferred_teacher_id, priority))
            conn.commit()
            return True, "Preference saved."
        except sqlite3.IntegrityError as exc:
            return False, str(exc)
 
 
def view_student_preferences(student_id):
    """Return all preferences for a student, in priority order."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT sub.subject_name, t.full_name, sp.priority
        FROM student_preferences sp
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        LEFT JOIN teachers t ON t.teacher_id = sp.preferred_teacher_id
        WHERE sp.student_id = ?
        ORDER BY sp.priority ASC;
    """, (student_id,))
        rows = cursor.fetchall()
    return rows  # list of (subject_name, teacher_name_or_None, priority)
 
 
def get_preferences_for_teacher(teacher_id):
    """Which students picked this teacher, and for which subject."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT s.full_name, s.roll_number, sub.subject_name, sp.priority
        FROM student_preferences sp
        JOIN students s ON s.student_id = sp.student_id
        JOIN subjects sub ON sub.subject_id = sp.subject_id
        WHERE sp.preferred_teacher_id = ?
        ORDER BY sub.subject_name, sp.priority;
    """, (teacher_id,))
        rows = cursor.fetchall()
    return rows


def create_flashcards_table():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            flashcard_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            grade_level TEXT,
            question_text TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            created_by TEXT DEFAULT 'teacher',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()


def seed_flashcards():
    default_cards = {
        "Mathematics": [
            ("What is the quadratic formula?", "x = (-b ± √(b² - 4ac)) / 2a."),
            ("What is the formula for force?", "Force = mass × acceleration (F = ma)."),
            ("What does BODMAS stand for?", "Brackets, Orders, Division, Multiplication, Addition, Subtraction."),
        ],
        "Physics": [
            ("What is acceleration?", "The rate of change of velocity."),
            ("What is resistance?", "The opposition to the flow of electric current."),
            ("What is the SI unit of force?", "The newton (N)."),
        ],
        "Chemistry": [
            ("What is an atom made of?", "Protons, neutrons, and electrons."),
            ("What is a catalyst?", "A substance that speeds up a reaction without being used up."),
            ("What is the pH of a neutral solution?", "7."),
        ],
        "Biology": [
            ("What is photosynthesis?", "The process by which green plants make glucose using light energy, water, and carbon dioxide."),
            ("What is the role of mitochondria?", "They release energy from food during respiration."),
            ("What is osmosis?", "The movement of water across a selectively permeable membrane from a dilute solution to a concentrated one."),
        ],
        "Computer Science": [
            ("What is an algorithm?", "A step-by-step set of instructions to solve a problem."),
            ("What is a variable?", "A named storage location for data in a program."),
            ("What is a loop?", "A control structure that repeats actions while a condition is true."),
        ],
        "English": [
            ("What is a metaphor?", "A comparison without using like or as."),
            ("What is the purpose of a topic sentence?", "To introduce the main idea of a paragraph."),
            ("What is the difference between a simile and a metaphor?", "A simile compares using like or as; a metaphor makes a direct comparison."),
        ],
        "Economics": [
            ("What is demand?", "The amount of a good consumers are willing and able to buy."),
            ("What is supply?", "The amount of a good producers are willing and able to sell."),
            ("What is scarcity?", "A situation where limited resources cannot satisfy all wants."),
        ],
    }
    with get_conn() as conn:
        cursor = conn.cursor()
        for subject_name, cards in default_cards.items():
            for question, answer in cards:
                cursor.execute(
                    "INSERT OR IGNORE INTO flashcards (subject_name, question_text, answer_text, created_by) VALUES (?, ?, ?, ?);",
                    (subject_name, question, answer, "system"),
                )
        conn.commit()


def add_flashcard(subject_name, question, answer, created_by="teacher", grade_level=None):
    if not subject_name or not question.strip() or not answer.strip():
        return False
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO flashcards (subject_name, grade_level, question_text, answer_text, created_by) VALUES (?, ?, ?, ?, ?);",
            (subject_name.strip(), (grade_level or "").strip(), question.strip(), answer.strip(), created_by),
        )
        conn.commit()
    return True


def get_flashcards_for_subject(subject_name, grade_level=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                "SELECT flashcard_id, question_text, answer_text FROM flashcards WHERE subject_name = ? AND (grade_level = ? OR grade_level IS NULL OR grade_level = '') ORDER BY flashcard_id;",
                (subject_name, grade_level),
            )
        else:
            cursor.execute(
                "SELECT flashcard_id, question_text, answer_text FROM flashcards WHERE subject_name = ? ORDER BY flashcard_id;",
                (subject_name,),
            )
        rows = cursor.fetchall()
    return rows


def get_flashcards_for_subjects(subject_names, grade_level=None):
    if not subject_names:
        return []
    placeholders = ", ".join("?" for _ in subject_names)
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                f"SELECT flashcard_id, subject_name, question_text, answer_text FROM flashcards WHERE subject_name IN ({placeholders}) AND (grade_level = ? OR grade_level IS NULL OR grade_level = '') ORDER BY subject_name, flashcard_id;",
                subject_names + [grade_level],
            )
        else:
            cursor.execute(
                f"SELECT flashcard_id, subject_name, question_text, answer_text FROM flashcards WHERE subject_name IN ({placeholders}) ORDER BY subject_name, flashcard_id;",
                subject_names,
            )
        rows = cursor.fetchall()
    return rows


def delete_flashcard(flashcard_id):
    """Delete a flashcard by ID."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM flashcards WHERE flashcard_id = ?;", (flashcard_id,))
        conn.commit()
    return True


def create_study_streaks_table():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_streaks (
            streak_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            study_date DATE NOT NULL,
            minutes_studied INTEGER DEFAULT 0,
            pomodoros_completed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (student_id, study_date),
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        );
        """)
        conn.commit()


def create_textbooks_table():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS textbooks (
            textbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            description TEXT,
            resource_type TEXT DEFAULT 'textbook',
            file_path TEXT,
            external_url TEXT,
            added_by TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            teacher_id INTEGER,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            chapter TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS past_papers (
            paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            grade_level TEXT,
            year INTEGER,
            paper_type TEXT,
            season TEXT,
            paper_number TEXT,
            question_paper_path TEXT,
            mark_scheme_path TEXT,
            examiner_report_path TEXT,
            duration_minutes INTEGER DEFAULT 60,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS syllabus_documents (
            syllabus_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            grade_level TEXT,
            title TEXT NOT NULL,
            file_path TEXT,
            chapter_outline TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()


def add_textbook(subject_name, title, author=None, description=None, resource_type='textbook', file_path=None, external_url=None, added_by='admin'):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO textbooks (subject_name, title, author, description, resource_type, file_path, external_url, added_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?);""",
            (subject_name, title, author, description, resource_type, file_path, external_url, added_by)
        )
        conn.commit()
    return True


def get_textbooks_for_subject(subject_name):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT textbook_id, title, author, description, resource_type, file_path, external_url FROM textbooks WHERE subject_name = ? ORDER BY title;",
            (subject_name,)
        )
        rows = cursor.fetchall()
    return rows


def add_teacher_note(subject_name, teacher_id, title, content, chapter=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO teacher_notes (subject_name, teacher_id, title, content, chapter)
               VALUES (?, ?, ?, ?, ?);""",
            (subject_name, teacher_id, title, content, chapter)
        )
        conn.commit()
    return True


def get_teacher_notes_for_subject(subject_name):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT note_id, title, content, chapter, created_at FROM teacher_notes
               WHERE subject_name = ? ORDER BY created_at DESC;""",
            (subject_name,)
        )
        rows = cursor.fetchall()
    return rows


def add_past_paper(subject_name, year, paper_type, season, paper_number, question_paper_path=None, mark_scheme_path=None, examiner_report_path=None, grade_level=None, duration_minutes=60):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
              """INSERT INTO past_papers (subject_name, grade_level, year, paper_type, season, paper_number, question_paper_path, mark_scheme_path, examiner_report_path, duration_minutes)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
              (subject_name, grade_level, year, paper_type, season, paper_number, question_paper_path, mark_scheme_path, examiner_report_path, int(duration_minutes or 60))
        )
        conn.commit()
    return True


def get_past_papers_for_subject(subject_name, grade_level=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                """SELECT paper_id, year, paper_type, season, paper_number, question_paper_path, mark_scheme_path, examiner_report_path, duration_minutes
                   FROM past_papers WHERE subject_name = ? AND (grade_level = ? OR grade_level IS NULL OR grade_level = '') ORDER BY year DESC, season DESC, paper_number;""",
                (subject_name, grade_level)
            )
        else:
            cursor.execute(
                """SELECT paper_id, year, paper_type, season, paper_number, question_paper_path, mark_scheme_path, examiner_report_path, duration_minutes
                   FROM past_papers WHERE subject_name = ? ORDER BY year DESC, season DESC, paper_number;""",
                (subject_name,)
            )
        rows = cursor.fetchall()
    return rows


def add_syllabus_document(subject_name, grade_level, title, file_path=None, chapter_outline=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO syllabus_documents (subject_name, grade_level, title, file_path, chapter_outline)
               VALUES (?, ?, ?, ?, ?);""",
            (subject_name, grade_level, title, file_path, chapter_outline),
        )
        conn.commit()
    return True


def get_syllabus_documents(subject_name, grade_level=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                "SELECT syllabus_id, title, file_path, chapter_outline, created_at FROM syllabus_documents WHERE subject_name = ? AND (grade_level = ? OR grade_level IS NULL OR grade_level = '') ORDER BY created_at DESC;",
                (subject_name, grade_level),
            )
        else:
            cursor.execute(
                "SELECT syllabus_id, title, file_path, chapter_outline, created_at FROM syllabus_documents WHERE subject_name = ? ORDER BY created_at DESC;",
                (subject_name,),
            )
        rows = cursor.fetchall()
    return rows


def get_teacher_pending_checks(teacher_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.submission_id, s.student_id, s.student_name, a.title, a.subject_name, a.total_marks,
                   s.answer_text, s.answer_file_path, s.submitted_at
            FROM assessment_submissions s
            JOIN assessments a ON a.assessment_id = s.assessment_id
            WHERE s.status = 'submitted' AND (s.teacher_check_enabled = 1 OR s.grading_mode IN ('Teacher only', 'AI + teacher'))
              AND (a.teacher_id IS NULL OR a.teacher_id = ?)
            ORDER BY s.submitted_at DESC;
            """,
            (teacher_id,),
        )
        rows = cursor.fetchall()
    return rows


def record_study_session(student_id, minutes_studied=0, pomodoros_completed=0):
    """Record a study session for streak tracking."""
    if not student_id:
        return False
    from datetime import date
    today = date.today().isoformat()
    with get_conn() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO study_streaks (student_id, study_date, minutes_studied, pomodoros_completed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(student_id, study_date)
                DO UPDATE SET minutes_studied = minutes_studied + excluded.minutes_studied,
                              pomodoros_completed = pomodoros_completed + excluded.pomodoros_completed;
            """, (student_id, today, minutes_studied, pomodoros_completed))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def get_student_streak(student_id):
    """Calculate current study streak for a student."""
    if not student_id:
        return {"current_streak": 0, "total_days": 0, "total_pomodoros": 0, "total_minutes": 0}
    
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT study_date, minutes_studied, pomodoros_completed
            FROM study_streaks
            WHERE student_id = ?
            ORDER BY study_date DESC;
        """, (student_id,))
        rows = cursor.fetchall()
    
    if not rows:
        return {"current_streak": 0, "total_days": 0, "total_pomodoros": 0, "total_minutes": 0}
    
    from datetime import date, timedelta
    today = date.today()
    current_streak = 0
    total_pomodoros = sum(row[2] for row in rows)
    total_minutes = sum(row[1] for row in rows)
    
    # Calculate current streak
    for i, (study_date_str, _, _) in enumerate(rows):
        study_date = date.fromisoformat(study_date_str)
        expected_date = today - timedelta(days=i)
        if study_date == expected_date:
            current_streak += 1
        else:
            break
    
    return {
        "current_streak": current_streak,
        "total_days": len(rows),
        "total_pomodoros": total_pomodoros,
        "total_minutes": total_minutes
    }


def create_assessment_tables():
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            total_marks INTEGER NOT NULL,
            grade_level TEXT,
            exam_duration_minutes INTEGER DEFAULT 60,
            description TEXT,
            question_paper_path TEXT,
            mark_scheme_path TEXT,
            examiner_report TEXT,
            teacher_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessment_submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            student_name TEXT,
            answer_text TEXT NOT NULL,
            grading_mode TEXT NOT NULL,
            ai_score REAL,
            teacher_score REAL,
            final_score REAL,
            status TEXT DEFAULT 'submitted',
            completed_in_time INTEGER DEFAULT 1,
            answer_file_path TEXT,
            self_check_enabled INTEGER DEFAULT 0,
            teacher_check_enabled INTEGER DEFAULT 0,
            ai_check_enabled INTEGER DEFAULT 0,
            ai_feedback TEXT,
            teacher_feedback TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );
        """)
        conn.commit()


def save_uploaded_file(uploaded_file, subdir, prefix):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return str(uploaded_file)
    target_dir = DATA_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower() or ".bin"
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    path = target_dir / fname
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


def create_assessment(title, subject_name, total_marks, grade_level, description, question_paper_file, mark_scheme_file, examiner_report_text, teacher_id=None, exam_duration_minutes=60):
    if not title.strip() or not subject_name or total_marks <= 0:
        return None
    question_paper_path = save_uploaded_file(question_paper_file, "assessments", "paper")
    mark_scheme_path = save_uploaded_file(mark_scheme_file, "assessments", "scheme")
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO assessments (title, subject_name, total_marks, grade_level, exam_duration_minutes, description, question_paper_path, mark_scheme_path, examiner_report, teacher_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (title.strip(), subject_name, total_marks, grade_level.strip() if grade_level else None, int(exam_duration_minutes or 60), description.strip() if description else None, question_paper_path, mark_scheme_path, examiner_report_text.strip() if examiner_report_text else None, teacher_id),
        )
        assessment_id = cursor.lastrowid
        conn.commit()
    return assessment_id


def get_assessments_for_subject(subject_name, grade_level=None):
    with get_conn() as conn:
        cursor = conn.cursor()
        if grade_level:
            cursor.execute(
                "SELECT assessment_id, title, subject_name, total_marks, grade_level, exam_duration_minutes, description, question_paper_path, mark_scheme_path, examiner_report, created_at FROM assessments WHERE subject_name = ? AND (grade_level = ? OR grade_level IS NULL OR grade_level = '') ORDER BY created_at DESC;",
                (subject_name, grade_level),
            )
        else:
            cursor.execute(
                "SELECT assessment_id, title, subject_name, total_marks, grade_level, exam_duration_minutes, description, question_paper_path, mark_scheme_path, examiner_report, created_at FROM assessments WHERE subject_name = ? ORDER BY created_at DESC;",
                (subject_name,),
            )
        rows = cursor.fetchall()
    return rows


def save_assessment_submission(assessment_id, student_id, student_name, answer_text, grading_mode, completed_in_time=True, answer_file_path=None, self_check=False, teacher_check=False, ai_check=False):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO assessment_submissions (assessment_id, student_id, student_name, answer_text, grading_mode, completed_in_time, answer_file_path, self_check_enabled, teacher_check_enabled, ai_check_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (assessment_id, student_id, student_name, answer_text.strip(), grading_mode, 1 if completed_in_time else 0, answer_file_path, 1 if self_check else 0, 1 if teacher_check else 0, 1 if ai_check else 0),
        )
        submission_id = cursor.lastrowid
        conn.commit()
    return submission_id


def update_submission_grade(submission_id, teacher_score=None, teacher_feedback=None, ai_score=None, ai_feedback=None, status="graded"):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ai_score, teacher_score FROM assessment_submissions WHERE submission_id = ?;",
            (submission_id,),
        )
        existing = cursor.fetchone()
        if existing is None:
            return False
        ai_score = ai_score if ai_score is not None else existing[0]
        teacher_score = teacher_score if teacher_score is not None else existing[1]
        final_score = None
        if ai_score is not None and teacher_score is not None:
            final_score = round((ai_score + teacher_score) / 2, 1)
        elif ai_score is not None:
            final_score = ai_score
        elif teacher_score is not None:
            final_score = teacher_score
        cursor.execute(
            """
            UPDATE assessment_submissions
            SET ai_score = ?, teacher_score = ?, final_score = ?, ai_feedback = COALESCE(?, ai_feedback), teacher_feedback = COALESCE(?, teacher_feedback), status = ?
            WHERE submission_id = ?;
            """,
            (ai_score, teacher_score, final_score, ai_feedback, teacher_feedback, status, submission_id),
        )
        conn.commit()
        return True


def get_submissions_for_assessment(assessment_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT submission_id, student_id, student_name, answer_text, grading_mode, ai_score, teacher_score, final_score, status, completed_in_time, answer_file_path, self_check_enabled, teacher_check_enabled, ai_check_enabled, ai_feedback, teacher_feedback, submitted_at FROM assessment_submissions WHERE assessment_id = ? ORDER BY submitted_at DESC;",
            (assessment_id,),
        )
        rows = cursor.fetchall()
    return rows


def get_submissions_for_student(student_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT s.submission_id, s.assessment_id, a.title, a.subject_name, a.total_marks, s.grading_mode, s.ai_score, s.teacher_score, s.final_score, s.status, s.completed_in_time, s.ai_feedback, s.teacher_feedback, s.submitted_at FROM assessment_submissions s JOIN assessments a ON a.assessment_id = s.assessment_id WHERE s.student_id = ? ORDER BY s.submitted_at DESC;",
            (student_id,),
        )
        rows = cursor.fetchall()
    return rows


def get_grade_band(percentage):
    if percentage >= 85:
        return "A"
    if percentage >= 70:
        return "B"
    if percentage >= 55:
        return "C"
    if percentage >= 40:
        return "D"
    return "E"


def get_performance_recommendation(percentage, feedback=None):
    if feedback:
        return feedback
    if percentage >= 80:
        return "Excellent progress — keep up the strong revision routine."
    if percentage >= 60:
        return "Solid work — a bit more consistency will raise your score further."
    if percentage >= 40:
        return "You are improving — focus on weak topics and ask your teacher for extra practice."
    return "Needs more practice — revisit the basics and ask for guided support."


def get_student_threshold(student_id):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT performance_threshold FROM students WHERE student_id = ?;", (student_id,))
        row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else DEFAULT_PERFORMANCE_THRESHOLD


def update_student_threshold(student_id, threshold):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE students SET performance_threshold = ? WHERE student_id = ?;", (int(threshold), student_id))
        conn.commit()
    return True


def get_student_performance_summary(student_id):
    submissions = get_submissions_for_student(student_id)
    completed = [s for s in submissions if s[8] is not None]
    if not completed:
        return {
            "total": 0,
            "average_percentage": 0,
            "best_percentage": 0,
            "latest": None,
            "subject_breakdown": {},
            "grade_band": "—",
        }

    percentages = [round((row[8] / row[4] * 100), 1) if row[4] else 0 for row in completed]
    average_percentage = round(sum(percentages) / len(percentages), 1)
    best_percentage = max(percentages)
    latest = completed[0]
    subject_breakdown = {}
    for row in completed:
        subject = row[3]
        subject_breakdown[subject] = subject_breakdown.get(subject, []) + [row[8] / row[4] * 100 if row[4] else 0]
    subject_breakdown = {k: round(sum(v) / len(v), 1) for k, v in subject_breakdown.items()}

    return {
        "total": len(completed),
        "average_percentage": average_percentage,
        "best_percentage": best_percentage,
        "latest": latest,
        "subject_breakdown": subject_breakdown,
        "grade_band": get_grade_band(average_percentage),
    }


def generate_flashcards_with_ai(model, subject_name, count=3):
    if not ai_ready(model):
        return []
    system = "You create high-quality, curriculum-friendly flashcards for O-Level students with short, clear questions and concise answers. Return ONLY JSON as {\"flashcards\":[{\"question\":\"...\",\"answer\":\"...\"}]}"
    prompt = f"Create {count} flashcards for {subject_name} covering easy and medium difficulty concepts. Avoid vague wording and ensure the answers are accurate."
    res = _call(model, prompt, system, max_tokens=800)
    data = parse_json(res["text"]) if res.get("ok") else None
    cards = []
    if isinstance(data, dict):
        for item in data.get("flashcards", []):
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                cards.append({"question": str(item["question"]).strip(), "answer": str(item["answer"]).strip()})
    return cards


QUIZ_BANK = [
    {"subject": "Mathematics", "difficulty": "Easy", "q": "What is 7 × 8?", "options": ["54", "56", "58", "60"], "answer_index": 1, "explanation": "7 times 8 equals 56."},
    {"subject": "Mathematics", "difficulty": "Medium", "q": "Solve: 3x + 5 = 20", "options": ["3", "5", "7", "10"], "answer_index": 2, "explanation": "Subtract 5 from both sides, then divide by 3."},
    {"subject": "Mathematics", "difficulty": "Hard", "q": "What is the quadratic formula?", "options": ["x = (-b ± √(b² - 4ac)) / 2a", "x = (b ± √(b² - 4ac)) / 2a", "x = (-b ± √(b² + 4ac)) / 2a", "x = (b ± √(b² + 4ac)) / 2a"], "answer_index": 0, "explanation": "The quadratic formula is used to solve ax² + bx + c = 0."},
    {"subject": "Physics", "difficulty": "Easy", "q": "What is the SI unit of force?", "options": ["Watt", "Newton", "Joule", "Metre"], "answer_index": 1, "explanation": "Force is measured in newtons."},
    {"subject": "Physics", "difficulty": "Medium", "q": "What does acceleration measure?", "options": ["Distance per second", "Speed only", "Rate of change of velocity", "Mass per second"], "answer_index": 2, "explanation": "Acceleration is the rate at which velocity changes."},
    {"subject": "Physics", "difficulty": "Hard", "q": "If a force of 10 N acts on a 2 kg mass, what is the acceleration?", "options": ["2 m/s²", "5 m/s²", "10 m/s²", "20 m/s²"], "answer_index": 1, "explanation": "Using F = ma, a = 10/2 = 5 m/s²."},
    {"subject": "Chemistry", "difficulty": "Easy", "q": "What is the pH of a neutral solution?", "options": ["3", "5", "7", "9"], "answer_index": 2, "explanation": "Neutral solutions have pH 7."},
    {"subject": "Chemistry", "difficulty": "Medium", "q": "What is a catalyst?", "options": ["A product of a reaction", "A reactant that slows down a reaction", "A substance that speeds up a reaction without being used up", "A type of acid"], "answer_index": 2, "explanation": "Catalysts speed up reactions and are not consumed."},
    {"subject": "Chemistry", "difficulty": "Hard", "q": "Which subatomic particle has a negative charge?", "options": ["Proton", "Neutron", "Electron", "Nucleus"], "answer_index": 2, "explanation": "Electrons carry a negative charge."},
    {"subject": "Biology", "difficulty": "Easy", "q": "What is the main function of chlorophyll?", "options": ["Store food", "Absorb light energy", "Release oxygen", "Produce roots"], "answer_index": 1, "explanation": "Chlorophyll absorbs light energy for photosynthesis."},
    {"subject": "Biology", "difficulty": "Medium", "q": "Where does respiration mainly occur in cells?", "options": ["Cell wall", "Nucleus", "Mitochondria", "Ribosome"], "answer_index": 2, "explanation": "Mitochondria are the main site of respiration."},
    {"subject": "Biology", "difficulty": "Hard", "q": "What is osmosis?", "options": ["Active movement of minerals", "Movement of water across a selectively permeable membrane", "Breakdown of glucose", "Formation of proteins"], "answer_index": 1, "explanation": "Osmosis is the movement of water across a selective membrane."},
    {"subject": "Computer Science", "difficulty": "Easy", "q": "What is a variable?", "options": ["A fixed value", "A storage location for data", "A type of loop", "A software bug"], "answer_index": 1, "explanation": "A variable stores data under a name."},
    {"subject": "Computer Science", "difficulty": "Medium", "q": "What does an algorithm describe?", "options": ["A hardware device", "A set of instructions to solve a problem", "A database table", "A computer screen"], "answer_index": 1, "explanation": "An algorithm is a step-by-step solution."},
    {"subject": "Computer Science", "difficulty": "Hard", "q": "What is the purpose of a loop in programming?", "options": ["To stop the program", "To repeat actions", "To delete variables", "To display output only"], "answer_index": 1, "explanation": "Loops repeat a block of instructions."},
    {"subject": "English", "difficulty": "Easy", "q": "What is a metaphor?", "options": ["A comparison using like or as", "A direct comparison without like or as", "A punctuation mark", "A type of noun"], "answer_index": 1, "explanation": "A metaphor makes a direct comparison."},
    {"subject": "English", "difficulty": "Medium", "q": "What is the main purpose of a topic sentence?", "options": ["To end the paragraph", "To introduce the paragraph's main idea", "To add a quote", "To state a conclusion"], "answer_index": 1, "explanation": "A topic sentence introduces the main idea."},
    {"subject": "English", "difficulty": "Hard", "q": "Which of these is a complex sentence?", "options": ["I ran quickly.", "Because it was raining, we stayed inside.", "The dog barked.", "She smiled."], "answer_index": 1, "explanation": "A complex sentence includes a dependent clause."},
    {"subject": "Economics", "difficulty": "Easy", "q": "What is demand?", "options": ["The amount producers sell", "The amount consumers are willing and able to buy", "The amount of money in the bank", "The amount of labour available"], "answer_index": 1, "explanation": "Demand is consumer willingness and ability to buy."},
    {"subject": "Economics", "difficulty": "Medium", "q": "What is scarcity?", "options": ["Too much money", "Unlimited resources", "Limited resources compared to unlimited wants", "Large supply"], "answer_index": 2, "explanation": "Scarcity means needs and wants exceed available resources."},
    {"subject": "Economics", "difficulty": "Hard", "q": "What happens when supply increases and demand stays the same?", "options": ["Price rises", "Price falls", "Demand rises", "Demand falls"], "answer_index": 1, "explanation": "More supply usually pushes price down."},
    {"subject": "Accounting", "difficulty": "Easy", "q": "What is the purpose of accounting?", "options": ["To build machines", "To record and report financial information", "To design websites", "To perform surgery"], "answer_index": 1, "explanation": "Accounting records and reports financial information."},
    {"subject": "Accounting", "difficulty": "Medium", "q": "What is a debit?", "options": ["An increase in liabilities", "An entry on the left side", "An entry on the right side", "A loss-making business"], "answer_index": 1, "explanation": "A debit is recorded on the left side of an account."},
]


# --------------------------------------------------------------------------- #
# UI — Sharks Academy branding
# --------------------------------------------------------------------------- #
_ORG_HEADER_CSS = """
<style>
.sharks-header {
    background: linear-gradient(135deg, #123b5d 0%, #0f6a8f 45%, #17a1a5 100%);
    border-radius: 24px;
    padding: 1.35rem 1.45rem 1.2rem;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 14px 34px rgba(23, 25, 59, 0.25);
    border: 1px solid rgba(255,255,255,0.16);
    position: relative;
    overflow: hidden;
}
.sharks-header::after {
    content: "";
    position: absolute;
    inset: auto -40px -60px auto;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(255,255,255,0.22), transparent 70%);
    pointer-events: none;
}
.sharks-header .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .35rem .7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin-bottom: .65rem;
}
.sharks-header h1 {
    margin: 0 0 .32rem;
    font-size: 2rem;
    letter-spacing: .5px;
    font-weight: 800;
}
.sharks-header p {
    margin: .2rem 0 .8rem;
    opacity: .92;
    font-size: 1.02rem;
    max-width: 720px;
}
.role-label {
    font-size: 1rem;
    font-weight: 700;
    color: #1e1e2f;
    margin-bottom: .5rem;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button {
    width: 100%;
    border-radius: 16px !important;
    padding: 1.1rem .6rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin-bottom: .9rem !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button:hover {
    transform: translateY(-2px);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: linear-gradient(135deg, #0f6a8f, #17a1a5) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(15,106,143,.35);
}
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: #f4f9fc !important;
    color: #184056 !important;
    border: 2px solid #d2e9f4 !important;
}

/* Improve readability for horizontal tabs: avoid white text on pale backgrounds. */
div[data-baseweb="tab-list"] {
    gap: .45rem;
    flex-wrap: wrap;
}
div[data-baseweb="tab-list"] button {
    background: #eff6fb !important;
    color: #101010 !important;
    border: 1px solid #d2e9f4 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #0f6a8f, #17a1a5) !important;
    color: #101010 !important;
    border-color: transparent !important;
}
</style>
"""


def _render_org_header():
    st.markdown(_ORG_HEADER_CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="sharks-header">
        <div class="eyebrow">✨ KG to A-Level Learning Platform</div>
        <h1>🌊 ScholarWave Learning Hub</h1>
        <p>Your all-in-one study platform for KG to Class 8, O-Level/IGCSE, and A-Level learners. Connect with expert teachers, access AI-powered tutoring, and build stronger exam confidence.</p>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_seo_keywords_footer():
    st.markdown("---")
    st.caption(
        "Curriculum search keywords: Cambridge Assessment International Education (CAIE), "
        "Cambridge O-Level, IGCSE, A-Level, O Level Computer Science checklist, "
        "IGCSE Accounting flashcards, Cambridge revision notes, Cambridge past papers."
    )


def _render_role_selector():
    st.sidebar.markdown('<div class="role-label">👋 I am a…</div>', unsafe_allow_html=True)
    st.session_state.setdefault("role", "Student")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("🎓 Student", key="role_student",
                 type="primary" if st.session_state.role == "Student" else "secondary",
                 use_container_width=True):
        st.session_state.role = "Student"
        st.rerun()
    if c2.button("👩‍🏫 Teacher", key="role_teacher",
                 type="primary" if st.session_state.role == "Teacher" else "secondary",
                 use_container_width=True):
        st.session_state.role = "Teacher"
        st.rerun()

    return st.session_state.role


def _render_mode_toggle():
    st.session_state.setdefault("socratic_mode", False)
    st.sidebar.markdown('<div class="role-label">🧠 Study mode</div>', unsafe_allow_html=True)
    st.sidebar.toggle(
        "Socratic mode",
        key="socratic_mode",
        help="Guide the student with questions and hints instead of giving away answers immediately.",
    )
    if st.session_state.get("socratic_mode"):
        st.sidebar.caption("The tutor will coach you with questions and hints.")
    else:
        st.sidebar.caption("The tutor will answer directly and clearly.")


def _render_pomodoro_sidebar():
    st.sidebar.markdown("### 🍅 Pomodoro Timer")
    st.sidebar.caption("25min study → 5min break → 4 cycles → 15min long break")
    pomodoro_html = """
    <div style="padding: 0.8rem; border: 1px solid #dfe6f1; border-radius: 14px; background: #f8faff;">
      <div id="pomodoro-mode" style="font-size: 0.85rem; font-weight: 600; color: #6C5CE7; text-align: center; margin-bottom: 0.4rem;">STUDY</div>
      <div id="pomodoro-display" style="font-size: 2rem; font-weight: 800; color: #2f3b5b; text-align: center;">25:00</div>
      <div id="pomodoro-cycle" style="font-size: 0.75rem; color: #8a8f9c; text-align: center; margin-bottom: 0.6rem;">Cycle 1/4</div>
      <div style="display: flex; gap: 0.5rem; margin-top: 0.6rem;">
        <button onclick="startPomodoro()" style="flex: 1; padding: 0.45rem; border: none; border-radius: 10px; background: #6C5CE7; color: white; cursor: pointer;">Start</button>
        <button onclick="pausePomodoro()" style="flex: 1; padding: 0.45rem; border: none; border-radius: 10px; background: #e8edf7; color: #2f3b5b; cursor: pointer;">Pause</button>
        <button onclick="resetPomodoro()" style="flex: 1; padding: 0.45rem; border: none; border-radius: 10px; background: #e8edf7; color: #2f3b5b; cursor: pointer;">Reset</button>
      </div>
    </div>
    <script>
      const display = document.getElementById('pomodoro-display');
      const modeDisplay = document.getElementById('pomodoro-mode');
      const cycleDisplay = document.getElementById('pomodoro-cycle');
      
      const STUDY_TIME = 25 * 60;
      const SHORT_BREAK = 5 * 60;
      const LONG_BREAK = 15 * 60;
      
      let timeLeft = STUDY_TIME;
      let timer = null;
      let isRunning = false;
      let currentMode = 'study'; // 'study', 'short_break', 'long_break'
      let cycleCount = 1;
      let completedPomodoros = 0;
      
      function updateDisplay() {
        const minutes = String(Math.floor(timeLeft / 60)).padStart(2, '0');
        const seconds = String(timeLeft % 60).padStart(2, '0');
        display.textContent = `${minutes}:${seconds}`;
        cycleDisplay.textContent = `Cycle ${cycleCount}/4`;
      }
      
      function updateModeDisplay() {
        if (currentMode === 'study') {
          modeDisplay.textContent = '📚 STUDY';
          modeDisplay.style.color = '#6C5CE7';
        } else if (currentMode === 'short_break') {
          modeDisplay.textContent = '☕ SHORT BREAK';
          modeDisplay.style.color = '#00B894';
        } else {
          modeDisplay.textContent = '🌴 LONG BREAK';
          modeDisplay.style.color = '#fd79a8';
        }
      }
      
      function startPomodoro() {
        if (isRunning) return;
        isRunning = true;
        timer = setInterval(() => {
          if (timeLeft > 0) {
            timeLeft -= 1;
            updateDisplay();
          } else {
            clearInterval(timer);
            isRunning = false;
            handleTimerComplete();
          }
        }, 1000);
        updateDisplay();
      }
      
      function pausePomodoro() {
        if (timer) {
          clearInterval(timer);
          isRunning = false;
        }
      }
      
      function handleTimerComplete() {
        if (currentMode === 'study') {
          completedPomodoros++;
          if (completedPomodoros % 4 === 0) {
            currentMode = 'long_break';
            timeLeft = LONG_BREAK;
            window.alert('🎉 Great work! Time for a 15-minute long break!');
          } else {
            currentMode = 'short_break';
            timeLeft = SHORT_BREAK;
            window.alert('✅ Pomodoro complete! Take a 5-minute break.');
          }
        } else {
          currentMode = 'study';
          timeLeft = STUDY_TIME;
          if (currentMode === 'study' && completedPomodoros % 4 === 0) {
            cycleCount = Math.min(cycleCount + 1, 4);
          }
          window.alert('⏰ Break over! Ready to focus again?');
        }
        updateModeDisplay();
        updateDisplay();
      }
      
      function resetPomodoro() {
        if (timer) clearInterval(timer);
        isRunning = false;
        currentMode = 'study';
        timeLeft = STUDY_TIME;
        cycleCount = 1;
        completedPomodoros = 0;
        updateModeDisplay();
        updateDisplay();
      }
      
      updateModeDisplay();
      updateDisplay();
    </script>
    """
    st.components.v1.html(pomodoro_html, height=220)


def _render_study_streak_sidebar():
    student_id = st.session_state.get("student_id")
    if not student_id:
        return
    
    # Initialize session tracking
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = None
    if "session_minutes" not in st.session_state:
        st.session_state.session_minutes = 0
    
    streak_data = get_student_streak(student_id)
    st.sidebar.markdown("### 🔥 Study Streak")
    st.sidebar.markdown(f"""
    <div style="padding: 0.8rem; border: 1px solid #dfe6f1; border-radius: 14px; background: linear-gradient(135deg, #fff5f5, #ffe8e8);">
      <div style="font-size: 2rem; font-weight: 800; color: #e17055; text-align: center;">{streak_data['current_streak']}</div>
      <div style="font-size: 0.8rem; color: #636e72; text-align: center; margin-bottom: 0.4rem;">day streak</div>
      <div style="font-size: 0.75rem; color: #636e72; text-align: center;">
        📚 {streak_data['total_days']} days total<br>
        🍅 {streak_data['total_pomodoros']} pomodoros<br>
        ⏱️ {streak_data['total_minutes']} mins studied
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Session tracking
    st.sidebar.markdown("### ⏱️ Session Timer")
    if st.session_state.session_start_time is None:
        if st.sidebar.button("▶️ Start Session", key="start_session"):
            from datetime import datetime
            st.session_state.session_start_time = datetime.now()
            st.rerun()
    else:
        from datetime import datetime
        elapsed = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
        st.sidebar.info(f"Session: {int(elapsed)} minutes")
        
        col1, col2 = st.sidebar.columns(2)
        if col1.button("⏸️ Pause", key="pause_session"):
            st.session_state.session_minutes += elapsed
            st.session_state.session_start_time = None
            st.rerun()
        if col2.button("⏹️ End & Save", key="end_session"):
            total_minutes = int(st.session_state.session_minutes + elapsed)
            record_study_session(student_id, total_minutes, 0)
            st.session_state.session_minutes = 0
            st.session_state.session_start_time = None
            st.sidebar.success(f"Saved {total_minutes} minutes!")
            st.rerun()
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📝 Manual Entry")
    if st.sidebar.button("📝 Log study session", key="log_study_btn"):
        minutes = st.sidebar.number_input("Minutes studied", min_value=1, max_value=480, value=25, key="log_study_minutes")
        pomodoros = st.sidebar.number_input("Pomodoros completed", min_value=0, max_value=20, value=1, key="log_study_pomodoros")
        if st.sidebar.button("Save session", key="save_study_session", type="primary"):
            record_study_session(student_id, minutes, pomodoros)
            st.sidebar.success("Study session logged!")
            st.rerun()


def _render_mobile_install_prompt():
        """Show a lightweight Add-to-Home-Screen prompt on mobile browsers."""
        install_prompt_html = """
        <script>
            (function () {
                const parentWin = window.parent || window;
                const doc = parentWin.document;
                const installUiId = 'sw-install-cta';

                if (doc.getElementById(installUiId)) {
                    return;
                }

                let deferredPrompt = null;
                const isMobile = /Android|iPhone|iPad|iPod/i.test(parentWin.navigator.userAgent || '');
                if (!isMobile) {
                    return;
                }

                const isStandalone = parentWin.matchMedia && parentWin.matchMedia('(display-mode: standalone)').matches;
                const isIosStandalone = parentWin.navigator.standalone === true;
                if (isStandalone || isIosStandalone) {
                    return;
                }

                const box = doc.createElement('div');
                box.id = installUiId;
                box.style.position = 'fixed';
                box.style.right = '14px';
                box.style.bottom = '14px';
                box.style.zIndex = '2147483647';
                box.style.background = 'linear-gradient(135deg, #0f6a8f, #17a1a5)';
                box.style.color = '#ffffff';
                box.style.padding = '10px 12px';
                box.style.borderRadius = '12px';
                box.style.boxShadow = '0 10px 24px rgba(0,0,0,.25)';
                box.style.fontFamily = 'system-ui, -apple-system, Segoe UI, sans-serif';
                box.style.maxWidth = '280px';
                box.style.display = 'none';

                const title = doc.createElement('div');
                title.textContent = 'Install Study Hub';
                title.style.fontWeight = '700';
                title.style.marginBottom = '6px';

                const hint = doc.createElement('div');
                hint.style.fontSize = '12px';
                hint.style.lineHeight = '1.35';
                hint.style.marginBottom = '8px';

                const row = doc.createElement('div');
                row.style.display = 'flex';
                row.style.gap = '8px';

                const installBtn = doc.createElement('button');
                installBtn.textContent = 'Add to Home Screen';
                installBtn.style.border = 'none';
                installBtn.style.borderRadius = '8px';
                installBtn.style.padding = '7px 10px';
                installBtn.style.fontWeight = '700';
                installBtn.style.cursor = 'pointer';
                installBtn.style.background = '#ffffff';
                installBtn.style.color = '#0f6a8f';

                const dismissBtn = doc.createElement('button');
                dismissBtn.textContent = 'Not now';
                dismissBtn.style.border = '1px solid rgba(255,255,255,.5)';
                dismissBtn.style.borderRadius = '8px';
                dismissBtn.style.padding = '7px 10px';
                dismissBtn.style.cursor = 'pointer';
                dismissBtn.style.background = 'transparent';
                dismissBtn.style.color = '#ffffff';

                row.appendChild(installBtn);
                row.appendChild(dismissBtn);
                box.appendChild(title);
                box.appendChild(hint);
                box.appendChild(row);
                doc.body.appendChild(box);

                function showIosHint() {
                    hint.textContent = 'On iPhone/iPad: tap Share, then Add to Home Screen.';
                    installBtn.style.display = 'none';
                    box.style.display = 'block';
                }

                function showInstallButton() {
                    hint.textContent = 'Get quick access from your home screen.';
                    installBtn.style.display = 'inline-block';
                    box.style.display = 'block';
                }

                function handleBeforeInstallPrompt(e) {
                    e.preventDefault();
                    deferredPrompt = e;
                    showInstallButton();
                }

                parentWin.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
                window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

                installBtn.addEventListener('click', async function () {
                    if (!deferredPrompt) {
                        return;
                    }
                    deferredPrompt.prompt();
                    try {
                        await deferredPrompt.userChoice;
                    } catch (err) {
                        // Ignore user dismissal and browser-specific prompt errors.
                    }
                    deferredPrompt = null;
                    box.style.display = 'none';
                });

                dismissBtn.addEventListener('click', function () {
                    box.style.display = 'none';
                });

                const isIos = /iPhone|iPad|iPod/i.test(parentWin.navigator.userAgent || '');
                if (isIos) {
                    showIosHint();
                }
            })();
        </script>
        """
        st.components.v1.html(install_prompt_html, height=0)


# --------------------------------------------------------------------------- #
# UI — Teacher
# --------------------------------------------------------------------------- #
def teacher_view(model):
    if not st.session_state.get("teacher_authenticated"):
        auth_login, auth_register = st.tabs(["🔐 Login", "🆕 Register"])
        with auth_login:
            st.subheader("Teacher sign in")
            with st.form("teacher_login"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary"):
                    if login_username and login_password:
                        auth_result = authenticate_teacher(login_username, login_password)
                        if auth_result:
                            st.session_state.teacher_authenticated = True
                            st.session_state.teacher_id = auth_result[0]
                            st.session_state.teacher_name = auth_result[1]
                            st.success(f"Welcome back, {auth_result[1]}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                    else:
                        st.warning("Please enter both username and password.")

        with auth_register:
            st.subheader("Create teacher account")
            with st.form("teacher_register"):
                reg_name = st.text_input("Full name")
                reg_email = st.text_input("Email")
                reg_username = st.text_input("Choose a username")
                reg_password = st.text_input("Choose a password", type="password")
                if st.form_submit_button("Register", type="primary"):
                    if reg_name.strip() and reg_username.strip() and reg_password.strip():
                        try:
                            register_teacher_for_subjects(
                                reg_name.strip(),
                                reg_email.strip(),
                                [],
                                grade_level=[],
                                username=reg_username.strip(),
                                password=reg_password.strip(),
                            )
                            st.success("Account created. Please sign in from the Login tab.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Registration failed: {exc}")
                    else:
                        st.warning("Please complete all required fields.")
        return

    st.success(f"✅ Logged in as: {st.session_state.teacher_name}")
    if st.button("Logout"):
        for key in ["teacher_authenticated", "teacher_id", "teacher_name"]:
            st.session_state.pop(key, None)
        st.rerun()

    tab_profile, tab_upload, tab_flashcards, tab_assessments, tab_to_check, tab_resources, tab_questions, tab_admin = st.tabs([
        "🏫 My classes & subjects",
        "📤 Upload lecture",
        "🧠 Manage flashcards",
        "📝 Assessments & grading",
        "✅ To be checked",
        "📚 Resources",
        "💬 Student questions",
        "⚙️ Admin & Billing",
    ])

    with tab_profile:
        st.subheader("🏫 Register classes and subjects you teach")
        profile = get_teacher_profile(st.session_state.teacher_id)
        linked_subjects = get_teacher_subject_links(st.session_state.teacher_id)

        saved_subjects = sorted({row[0] for row in linked_subjects if row[0]})
        saved_grades = sorted({row[1] for row in linked_subjects if row[1]})
        if not saved_grades and profile and profile[3]:
            saved_grades = [g.strip() for g in str(profile[3]).split(",") if g.strip()]

        with st.form("teacher_profile"):
            t_name = st.text_input("Your full name", value=(profile[1] if profile else st.session_state.teacher_name))
            t_email = st.text_input("Your email", value=(profile[2] if profile and profile[2] else ""))
            t_grades = st.multiselect("Classes you teach", GRADE_LEVEL_OPTIONS, default=saved_grades)
            t_subjects = st.multiselect("Subjects you teach", SUBJECTS, default=saved_subjects)
            t_submitted = st.form_submit_button("Save my profile", type="primary")
            if t_submitted:
                if not t_name.strip() or not t_subjects or not t_grades:
                    st.warning("Please provide your name, at least one class, and at least one subject.")
                else:
                    register_teacher_for_subjects(
                        t_name.strip(),
                        t_email.strip(),
                        t_subjects,
                        t_grades,
                        teacher_id=st.session_state.teacher_id,
                    )
                    st.session_state.teacher_name = t_name.strip()
                    st.success("Profile updated successfully.")

        st.divider()
        st.markdown("#### Students who picked you")
        rows = get_preferences_for_teacher(st.session_state.teacher_id)
        if not rows:
            st.info("No students have picked you yet.")
        else:
            for full_name, roll, subject_name, priority in rows:
                st.write(f"**{full_name}** ({roll}) — {subject_name}, priority {priority}")

    with tab_upload:
        st.subheader("👩‍🏫 Upload lecture")
        with st.form("upload", clear_on_submit=True):
            title = st.text_input("Lecture title", placeholder="e.g. Photosynthesis — Part 1")
            subject = st.selectbox("Subject", SUBJECTS)
            grade_level = st.selectbox("Class", GRADE_LEVEL_OPTIONS)
            description = st.text_input("One-line description", placeholder="What is this lesson about?")
            notes = st.text_area("Lecture notes (used by AI)", height=180,
                                 placeholder="Paste or write key notes for this lecture…")
            video = st.file_uploader("Video lecture", type=VIDEO_TYPES)
            submitted = st.form_submit_button("⬆️ Upload lecture", type="primary")
            if submitted:
                if not title.strip() or video is None:
                    st.warning("Please give a title and choose a video file.")
                else:
                    add_lecture(title, subject, description, notes, video, teacher_id=st.session_state.teacher_id, grade_level=grade_level)
                    st.success(f"Uploaded “{title.strip()}” for {subject} · Class {grade_level}.")

    with tab_flashcards:
        st.subheader("🧠 Add flashcards for your students")
        st.caption("Flashcards are stored by subject and class.")
        flash_subject = st.selectbox("Subject", SUBJECTS, key="teacher_flashcard_subject")
        flash_grade = st.selectbox("Class", GRADE_LEVEL_OPTIONS, key="teacher_flashcard_grade")
        with st.form("flashcard_form", clear_on_submit=True):
            q = st.text_input("Question", placeholder="e.g. What is the formula for speed?")
            a = st.text_area("Answer", height=120, placeholder="Write the correct answer here…")
            submitted = st.form_submit_button("Add flashcard", type="primary")
            if submitted:
                if q.strip() and a.strip():
                    add_flashcard(flash_subject, q, a, created_by="teacher", grade_level=flash_grade)
                    st.success("Flashcard saved.")
                else:
                    st.warning("Please enter both a question and an answer.")

        if st.button("✨ Generate 3 with AI", type="secondary"):
            with st.spinner("Generating flashcards…"):
                generated = generate_flashcards_with_ai(model, flash_subject, count=3)
            if generated:
                for item in generated:
                    add_flashcard(flash_subject, item["question"], item["answer"], created_by="AI", grade_level=flash_grade)
                st.success(f"Added {len(generated)} AI-generated flashcards for {flash_subject} · Class {flash_grade}.")
            else:
                st.info("The AI generator is not configured right now.")

        st.divider()
        st.markdown("#### Saved flashcards")
        cards = get_flashcards_for_subject(flash_subject, flash_grade)
        if cards:
            for flashcard_id, question, answer in cards:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**Q:** {question}")
                c1.caption(f"**A:** {answer}")
                if c2.button("🗑️", key=f"del_flash_{flashcard_id}"):
                    delete_flashcard(flashcard_id)
                    st.rerun()
                st.divider()
        else:
            st.info("No flashcards saved for this subject yet.")

    with tab_assessments:
        st.subheader("📝 Assign past papers and worksheets")
        st.caption("Set class, paper duration, and grading workflow for students.")
        with st.form("assessment_form", clear_on_submit=True):
            assessment_title = st.text_input("Assessment title", placeholder="e.g. Unit 3 revision test")
            assessment_subject = st.selectbox("Subject", SUBJECTS, key="teacher_assessment_subject")
            assessment_grade = st.selectbox("Class", GRADE_LEVEL_OPTIONS, key="teacher_assessment_grade")
            total_marks = st.number_input("Total marks", min_value=1, max_value=100, value=20)
            exam_duration = st.number_input("Time allowed (minutes)", min_value=5, max_value=300, value=60)
            description = st.text_area("Instructions or summary", height=120, placeholder="Tell students what to do")
            question_paper = st.file_uploader("Question paper (PDF or text)", type=["pdf", "txt", "docx"])
            mark_scheme = st.file_uploader("Mark scheme (optional)", type=["pdf", "txt", "docx"])
            examiner_report = st.text_area("Examiner report (optional)", height=120, placeholder="Optional guidance for grading")
            submitted = st.form_submit_button("Save assessment", type="primary")
            if submitted:
                if not assessment_title.strip():
                    st.warning("Please give the assessment a title.")
                else:
                    create_assessment(
                        assessment_title,
                        assessment_subject,
                        int(total_marks),
                        assessment_grade,
                        description,
                        question_paper,
                        mark_scheme,
                        examiner_report,
                        teacher_id=st.session_state.teacher_id,
                        exam_duration_minutes=int(exam_duration),
                    )
                    st.success("Assessment saved.")

        st.divider()
        st.markdown("#### Current assessments")
        assessments = get_assessments_for_subject(assessment_subject, assessment_grade)
        if assessments:
            for assessment_id, title, subject_name, total_marks, grade_level, exam_duration_minutes, description, question_paper_path, mark_scheme_path, examiner_report, created_at in assessments:
                with st.expander(f"{title} · {total_marks} marks"):
                    st.caption(f"{created_at} · Class {grade_level or 'Any'} · {exam_duration_minutes} mins")
                    if description:
                        st.write(description)
                    st.markdown("#### Submission summary")
                    submissions = get_submissions_for_assessment(assessment_id)
                    st.caption(f"Total submissions: {len(submissions)}")
        else:
            st.info("No assessments saved yet.")

    with tab_to_check:
        st.subheader("✅ Student assessments to check")
        pending = get_teacher_pending_checks(st.session_state.teacher_id)
        if not pending:
            st.info("No pending submissions right now.")
        else:
            for submission_id, student_id, student_name, title, subject_name, total_marks, answer_text, answer_file_path, submitted_at in pending:
                with st.expander(f"{student_name or 'Student'} · {title} ({subject_name})"):
                    st.caption(f"Submitted: {submitted_at}")
                    st.write(answer_text)
                    if answer_file_path and Path(answer_file_path).exists():
                        with open(answer_file_path, "rb") as fh:
                            st.download_button("Download student solved paper", fh.read(), file_name=Path(answer_file_path).name, mime="application/octet-stream", key=f"dl_ans_{submission_id}")

                    with st.form(f"grade_pending_{submission_id}"):
                        teacher_mark = st.number_input(
                            "Teacher score",
                            min_value=0.0,
                            max_value=float(total_marks),
                            value=float(total_marks) / 2.0,
                            key=f"teacher_mark_pending_{submission_id}",
                            step=0.5,
                        )
                        feedback = st.text_area("Teacher feedback", key=f"teacher_feedback_pending_{submission_id}")
                        if st.form_submit_button("Save teacher grade", type="primary"):
                            update_submission_grade(submission_id, teacher_score=float(teacher_mark), teacher_feedback=feedback)
                            st.success("Teacher grade saved.")

    with tab_questions:
        st.subheader("💬 Questions from students")
        st.caption("Students can ask you directly from the study view when they need teacher help.")
        questions = get_teacher_questions(st.session_state.teacher_id)
        if not questions:
            st.info("No questions sent to you yet.")
        else:
            for question_id, student_name, subject_name, question_text, answer_text, answered_by, answered_at, created_at in questions:
                st.markdown(f"**{student_name or 'Student'}** · {subject_name}")
                st.write(question_text)
                st.caption(created_at)
                if answer_text:
                    st.success(f"Answered by {answered_by or 'Teacher'}{f' at {answered_at}' if answered_at else ''}")
                    st.write(answer_text)
                with st.form(f"answer_question_{question_id}"):
                    reply = st.text_area("Teacher reply", value=answer_text or "", height=120, key=f"teacher_reply_{question_id}")
                    if st.form_submit_button("Save reply", type="primary"):
                        if not reply.strip():
                            st.warning("Please write an answer before saving.")
                        else:
                            save_teacher_question_answer(question_id, reply, st.session_state.get("teacher_name", "Teacher"))
                            st.success("Reply saved.")
                st.divider()

    with tab_admin:
        st.subheader("⚙️ Admin & Billing")
        st.caption("Review pending premium payments and recent student feedback.")

        if not supabase:
            st.warning("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY to enable cloud admin workflows.")

        pending_requests = []
        feedback_items = []
        if supabase:
            try:
                pending_response = (
                    supabase.table("payment_requests")
                    .select("request_id, user_role, user_identifier, user_name, user_email, reference, note, status, created_at")
                    .eq("status", "pending")
                    .order("created_at", desc=True)
                    .limit(200)
                    .execute()
                )
                pending_requests = pending_response.data or []
            except Exception as exc:
                st.error(f"Could not load pending payments from Supabase: {exc}")

            try:
                feedback_response = (
                    supabase.table("suggestions")
                    .select("suggestion_id, user_role, user_name, user_email, category, message, created_at")
                    .order("created_at", desc=True)
                    .limit(200)
                    .execute()
                )
                feedback_items = feedback_response.data or []
            except Exception as exc:
                st.error(f"Could not load feedback from Supabase: {exc}")
        else:
            pending_requests = [
                req for req in get_payment_requests(limit=200)
                if str(req.get("status") or "").lower() == "pending"
            ]
            feedback_items = get_recent_suggestions(limit=200)

        st.markdown("#### Pending premium approvals")
        if not pending_requests:
            st.info("No pending payment requests right now.")
        else:
            for request in pending_requests:
                request_id = request.get("request_id")
                user_name = request.get("user_name") or "Unknown user"
                user_role = request.get("user_role") or "Student"
                created_at = request.get("created_at") or ""
                with st.expander(f"{user_name} · {user_role} · {created_at}"):
                    st.write(f"Reference: {request.get('reference') or 'N/A'}")
                    st.write(f"Identifier: {request.get('user_identifier') or 'N/A'}")
                    st.write(f"Email: {request.get('user_email') or 'N/A'}")
                    if request.get("note"):
                        st.caption(request.get("note"))

                    approve_key = f"approve_payment_{request_id}"
                    if st.button("Approve pending premium key", key=approve_key, type="primary"):
                        ok, message = grant_premium_access(
                            request.get("user_role") or "Student",
                            user_identifier=request.get("user_identifier"),
                            user_email=request.get("user_email"),
                            days=PREMIUM_DAYS_DEFAULT,
                        )
                        if ok:
                            reviewer = st.session_state.get("teacher_name") or "Admin"
                            update_payment_request_status(
                                request_id,
                                "approved",
                                reviewed_by=reviewer,
                                reviewed_note="Approved in Admin & Billing tab",
                            )
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        st.divider()
        st.markdown("#### Student feedback inbox")
        if not feedback_items:
            st.info("No feedback entries yet.")
        else:
            for item in feedback_items:
                category = item.get("category") or "Feedback"
                user_name = item.get("user_name") or "Anonymous"
                created_at = item.get("created_at") or ""
                with st.expander(f"{category} · {user_name} · {created_at}"):
                    st.write(item.get("message") or "")
                    st.caption(f"Role: {item.get('user_role') or 'Unknown'} · Email: {item.get('user_email') or 'N/A'}")
 
 
# --------------------------------------------------------------------------- #
# UI — Student: flashcards + study tools
# --------------------------------------------------------------------------- #

def _render_flashcards(subject_names, grade_level=None):
    st.subheader("🧠 Flashcards")
    st.caption("Flip each card to test yourself with quick definitions.")
    cards = get_flashcards_for_subjects(subject_names, grade_level=grade_level)
    if not cards:
        st.info("No flashcards are available for your enrolled subjects yet.")
        return

    subject_cards = []
    for _, subject_name, question_text, answer_text in cards:
        subject_cards.append({"subject": subject_name, "front": question_text, "back": answer_text})

    if not subject_cards:
        st.info("No flashcards are available for your enrolled subjects yet.")
        return

    if "flashcard_subject" not in st.session_state or st.session_state.flashcard_subject != str(subject_names):
        st.session_state.flashcard_subject = str(subject_names)
        st.session_state.flashcard_index = 0
        st.session_state.flashcard_flipped = False

    card = subject_cards[st.session_state.flashcard_index]
    face = card["back"] if st.session_state.flashcard_flipped else card["front"]
    st.markdown(
        f"""
        <div style="border:1px solid #dfe6f1; border-radius: 18px; padding: 1.3rem; background: linear-gradient(135deg, #f8faff, #eef3ff); min-height: 170px; display: flex; align-items: center; justify-content: center; text-align: center; font-size: 1.1rem; font-weight: 600; color: #2f3b5b;">
            {face}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if card.get("subject"):
        st.caption(f"Subject: {card['subject']}")

    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("← Previous"):
        st.session_state.flashcard_index = (st.session_state.flashcard_index - 1) % len(subject_cards)
        st.session_state.flashcard_flipped = False
    if c2.button("Flip card"):
        st.session_state.flashcard_flipped = not st.session_state.flashcard_flipped
    if c3.button("Next →"):
        st.session_state.flashcard_index = (st.session_state.flashcard_index + 1) % len(subject_cards)
        st.session_state.flashcard_flipped = False


def _render_syllabus_checklist(subject_name, grade_level=None):
    st.subheader("📘 Syllabus checklist")
    st.caption("Track your chapter-by-chapter progress and keep your revision moving.")
    chapters = get_syllabus_chapters(subject_name, grade_level=grade_level)
    if not chapters:
        st.info(f"No chapters have been saved for {subject_name} yet.")
        return

    completed = 0
    for chapter in chapters:
        key = f"chapter_complete_{safe_name(subject_name)}_{safe_name(chapter)}"
        st.session_state.setdefault(key, False)
        if st.session_state.get(key):
            completed += 1
    progress = int(completed / len(chapters) * 100) if chapters else 0
    st.progress(max(0.0, min(progress / 100.0, 1.0)))
    st.caption(f"{completed}/{len(chapters)} chapters completed")

    cols = st.columns(2)
    for idx, chapter in enumerate(chapters):
        with cols[idx % 2]:
            key = f"chapter_complete_{safe_name(subject_name)}_{safe_name(chapter)}"
            st.session_state.setdefault(key, False)
            st.checkbox(chapter, value=st.session_state.get(key, False), key=key)

    if progress == 100:
        st.success("Excellent work — you have completed the full checklist.")


def _render_past_paper_grader(model, subject_name, grade_level=None):
    st.subheader("📝 Past paper & worksheet grader")
    st.caption("Choose a past paper, submit your answer, and pick whether AI, your teacher, or both should grade it.")
    student_id = st.session_state.get("student_id")
    if not student_id:
        st.info("Sign in first to submit work for grading.")
        return

    # Get both teacher assessments and past papers
    assessments = get_assessments_for_subject(subject_name, grade_level=grade_level)
    past_papers = get_past_papers_for_subject(subject_name, grade_level=grade_level)
    
    if not assessments and not past_papers:
        st.info(f"No assessments or past papers are available for {subject_name} yet. Ask your teacher to add some.")
        return

    # Combine options
    paper_options = []
    if assessments:
        for assessment_id, title, _, total_marks, _, _, _, _, _, _, _ in assessments:
            paper_options.append(("assessment", assessment_id, f"{title} · {total_marks} marks (Teacher Assessment)"))
    if past_papers:
        for paper_id, year, paper_type, season, paper_number, _, _, _, _ in past_papers:
            paper_options.append(("past_paper", paper_id, f"{year} {season} - {paper_type} Paper {paper_number} (Past Paper)"))
    
    if not paper_options:
        st.info("No papers available.")
        return
    
    selected_idx = st.selectbox("Choose a paper to solve", range(len(paper_options)), format_func=lambda i: paper_options[i][2])
    paper_type, paper_id, paper_name = paper_options[selected_idx]
    
    if paper_type == "assessment":
        # Use existing teacher assessment flow
        assessment = [a for a in assessments if a[0] == paper_id][0]
        assessment_id, title, _, total_marks, _, exam_duration_minutes, description, question_paper_path, mark_scheme_path, examiner_report, created_at = assessment
        
        st.write(f"**{title}**")
        if description:
            st.caption(description)
        if question_paper_path and Path(question_paper_path).exists():
            with open(question_paper_path, "rb") as fh:
                st.download_button("Download question paper", fh.read(), file_name=Path(question_paper_path).name, mime="application/octet-stream")
        st.caption(f"Time allowed: {exam_duration_minutes} minutes")
        timer_key = f"assessment_timer_start_{assessment_id}"
        if st.button("Start timer", key=f"start_timer_{assessment_id}"):
            st.session_state[timer_key] = datetime.now().timestamp()
        if st.session_state.get(timer_key):
            elapsed = int((datetime.now().timestamp() - st.session_state[timer_key]) / 60)
            remaining = max(int(exam_duration_minutes) - elapsed, 0)
            st.info(f"Time remaining: {remaining} minute(s)")
        
        answer = st.text_area("Your answer", height=260, placeholder="Write your full response here…")
        checking_choices = st.multiselect(
            "Choose how to check this paper",
            ["Self check", "Teacher check", "AI check"],
            default=["Teacher check", "AI check"],
        )
        solved_file = st.file_uploader("Upload your solved paper (optional)", type=["pdf", "jpg", "jpeg", "png"], key=f"assessment_solved_{assessment_id}")
        completed_in_time = st.radio("Did you complete it in the given time?", ["Yes", "No"], horizontal=True, key=f"in_time_{assessment_id}")
        if st.button("Submit for grading", type="primary"):
            if not answer.strip():
                st.warning("Please enter an answer before submitting it.")
            else:
                answer_file_path = save_uploaded_file(solved_file, "student_submissions", "answer") if solved_file else None
                grading_mode = " + ".join(checking_choices) if checking_choices else "Teacher check"
                submission_id = save_assessment_submission(
                    assessment_id,
                    student_id,
                    st.session_state.get("student_name"),
                    answer,
                    grading_mode,
                    completed_in_time=(completed_in_time == "Yes"),
                    answer_file_path=answer_file_path,
                    self_check=("Self check" in checking_choices),
                    teacher_check=("Teacher check" in checking_choices),
                    ai_check=("AI check" in checking_choices),
                )
                ai_score = None
                ai_feedback = None
                if "AI check" in checking_choices and ai_ready(model):
                    with st.spinner("The AI examiner is marking your work…"):
                        system = "You are an experienced O-Level examiner. Grade the student's answer fairly, using the available instructions and total marks. Return ONLY JSON with keys: score_out_of_total, feedback, strengths, improvement_points."
                        prompt = f"Assessment: {title}\nTotal marks: {total_marks}\nInstructions: {description or 'None'}\nStudent answer:\n{answer}\nReturn a score out of the total marks and useful feedback."
                        result = _call(model, prompt, system, max_tokens=900)
                        data = parse_json(result["text"]) if result.get("ok") else None
                    if isinstance(data, dict):
                        ai_score = parse_score_value(data.get("score_out_of_total", 0), total_marks)
                        ai_feedback = str(data.get("feedback") or "")
                update_submission_grade(submission_id, ai_score=ai_score, ai_feedback=ai_feedback, status="submitted")
                if "Self check" in checking_choices:
                    if mark_scheme_path and Path(mark_scheme_path).exists():
                        with open(mark_scheme_path, "rb") as fh:
                            st.download_button("Download mark scheme", fh.read(), file_name=Path(mark_scheme_path).name, mime="application/octet-stream", key=f"ms_after_{submission_id}")
                    if examiner_report:
                        st.info(examiner_report)
                st.success("Your work has been submitted for grading.")
    else:
        # Past paper flow
        paper = [p for p in past_papers if p[0] == paper_id][0]
        paper_id, year, paper_type, season, paper_number, qp_path, ms_path, er_path, duration_minutes = paper
        
        st.write(f"**{year} {season} - {paper_type} Paper {paper_number}**")
        if qp_path and Path(qp_path).exists():
            with open(qp_path, "rb") as fh:
                st.download_button("Download question paper", fh.read(), file_name=Path(qp_path).name, mime="application/octet-stream")
        st.caption(f"Time allowed: {duration_minutes or 60} minutes")
        timer_key = f"pastpaper_timer_start_{paper_id}"
        if st.button("Start timer", key=f"start_pp_timer_{paper_id}"):
            st.session_state[timer_key] = datetime.now().timestamp()
        if st.session_state.get(timer_key):
            elapsed = int((datetime.now().timestamp() - st.session_state[timer_key]) / 60)
            remaining = max(int(duration_minutes or 60) - elapsed, 0)
            st.info(f"Time remaining: {remaining} minute(s)")
        
        answer = st.text_area("Your answer", height=260, placeholder="Write your full response here…")
        checking_choices = st.multiselect(
            "Choose how to check this paper",
            ["Self check", "Teacher check", "AI check"],
            default=["Teacher check", "AI check"],
            key=f"pastpaper_checks_{paper_id}",
        )
        solved_file = st.file_uploader("Upload your solved paper (optional)", type=["pdf", "jpg", "jpeg", "png"], key=f"pastpaper_solved_{paper_id}")
        completed_in_time = st.radio("Did you complete it in the given time?", ["Yes", "No"], horizontal=True, key=f"pp_in_time_{paper_id}")
        if st.button("Submit for grading", type="primary"):
            if not answer.strip():
                st.warning("Please enter an answer before submitting it.")
            else:
                # Create a temporary assessment for the past paper
                temp_title = f"{year} {season} {paper_type} Paper {paper_number}"
                temp_assessment_id = create_assessment(temp_title, subject_name, 50, grade_level or "", "Past paper submission", qp_path, ms_path, er_path if er_path else "", exam_duration_minutes=int(duration_minutes or 60))
                answer_file_path = save_uploaded_file(solved_file, "student_submissions", "answer") if solved_file else None
                grading_mode = " + ".join(checking_choices) if checking_choices else "Teacher check"
                submission_id = save_assessment_submission(
                    temp_assessment_id,
                    student_id,
                    st.session_state.get("student_name"),
                    answer,
                    grading_mode,
                    completed_in_time=(completed_in_time == "Yes"),
                    answer_file_path=answer_file_path,
                    self_check=("Self check" in checking_choices),
                    teacher_check=("Teacher check" in checking_choices),
                    ai_check=("AI check" in checking_choices),
                )
                ai_score = None
                ai_feedback = None
                if "AI check" in checking_choices and ai_ready(model):
                    with st.spinner("The AI examiner is marking your work…"):
                        system = "You are an experienced O-Level examiner. Grade the student's answer fairly. Return ONLY JSON with keys: score_out_of_total, feedback, strengths, improvement_points."
                        prompt = f"Past Paper: {temp_title}\nSubject: {subject_name}\nStudent answer:\n{answer}\nReturn a score out of 50 and useful feedback."
                        result = _call(model, prompt, system, max_tokens=900)
                        data = parse_json(result["text"]) if result.get("ok") else None
                    if isinstance(data, dict):
                        ai_score = parse_score_value(data.get("score_out_of_total", 0), 50)
                        ai_feedback = str(data.get("feedback") or "")
                update_submission_grade(submission_id, ai_score=ai_score, ai_feedback=ai_feedback, status="submitted")
                if "Self check" in checking_choices:
                    if ms_path and Path(ms_path).exists():
                        with open(ms_path, "rb") as fh:
                            st.download_button("Download mark scheme", fh.read(), file_name=Path(ms_path).name, mime="application/octet-stream", key=f"pp_ms_after_{submission_id}")
                    if er_path and Path(er_path).exists():
                        with open(er_path, "rb") as fh:
                            st.download_button("Download examiner report", fh.read(), file_name=Path(er_path).name, mime="application/octet-stream", key=f"pp_er_after_{submission_id}")
                st.success("Your work has been submitted for grading.")

    st.divider()
    st.markdown("#### Your past submissions")
    submissions = get_submissions_for_student(student_id)
    if not submissions:
        st.info("You have not submitted anything yet.")
    else:
        for submission_id, assessment_id, title, subject_name, total_marks, grading_mode, ai_score, teacher_score, final_score, status, completed_in_time, ai_feedback, teacher_feedback, submitted_at in submissions:
            st.markdown(f"**{title}** · {subject_name}")
            st.caption(f"Submitted {submitted_at} · {grading_mode} · Completed in time: {'Yes' if completed_in_time else 'No'}")
            if final_score is not None:
                st.metric("Score", f"{final_score}/{total_marks}")
            else:
                st.caption("Waiting for grading.")
            if teacher_feedback:
                st.write(f"**Teacher feedback:** {teacher_feedback}")
            if ai_feedback:
                st.write(f"**AI feedback:** {ai_feedback}")
            st.divider()


def _render_study_modes(model, subject_name, grade_level=None):
    st.subheader("🛡️ Study modes")
    st.caption("Turn a topic into a challenge, a simpler explanation, a concept map, or a one-page cheat sheet.")

    student_id = st.session_state.get("student_id")
    grade_level = ""
    if student_id:
        student_row = get_student_by_id(student_id)
        if student_row:
            grade_level = student_row[3] or ""

    topic = st.text_input("Topic or concept", placeholder="e.g. photosynthesis", key="study_mode_topic")
    if not topic.strip():
        st.info("Type a topic to unlock the study tools.")
        return

    tab_boss, tab_eli5, tab_map, tab_sheet = st.tabs(["⚔️ Boss Battle", "🧒 ELI5", "🧩 Concept map", "📋 Cheat sheet"])

    with tab_boss:
        if not ai_ready(model):
            st.info("Study-mode AI tools need the AI endpoint configured.")
        else:
            if st.button("Start Boss Battle", type="primary", key="boss_battle_btn"):
                with st.spinner("Building your challenge…"):
                    st.session_state["boss_battle_data"] = generate_boss_battle_challenge(model, subject_name, grade_level, topic)
                    st.session_state["boss_battle_answers"] = {}
                    st.session_state["boss_battle_submitted"] = False

            if st.session_state.get("boss_battle_data"):
                data = st.session_state["boss_battle_data"]
                questions = data.get("questions", [])
                final_mission = data.get("final_mission", "")

                if not questions:
                    st.info(final_mission)
                else:
                    st.markdown(f"### ⚔️ Boss Battle Challenge")
                    st.caption(f"Topic: {topic}")

                    for i, q in enumerate(questions):
                        st.markdown(f"**Q{i+1}. {q.get('question', '')}**")
                        options = q.get("options", [])
                        if options:
                            answer = st.radio("Select your answer:", options, key=f"boss_q_{i}", index=None)
                            st.session_state["boss_battle_answers"][i] = answer

                    if st.button("Submit Answers", type="primary", key="boss_submit"):
                        st.session_state["boss_battle_submitted"] = True

                    if st.session_state.get("boss_battle_submitted"):
                        correct = 0
                        for i, q in enumerate(questions):
                            user_answer = st.session_state["boss_battle_answers"].get(i)
                            correct_answer = q["options"][q.get("correct_index", 0)]
                            is_correct = user_answer == correct_answer
                            if is_correct:
                                correct += 1

                            st.markdown(f"**Q{i+1} Result:** {'✅ Correct!' if is_correct else '❌ Incorrect'}")
                            if not is_correct:
                                st.caption(f"💡 Hint: {q.get('hint', 'No hint available')}")
                                st.info(f"📚 Explanation: {q.get('explanation', 'No explanation available')}")

                        st.markdown(f"### Score: {correct}/{len(questions)}")
                        if correct == len(questions):
                            st.balloons()
                            st.success("🎉 Perfect! You defeated the boss!")
                        else:
                            st.warning(f"You got {correct} out of {len(questions)} correct. Keep practicing!")

                        st.markdown(f"### 🏆 Final Mission")
                        st.markdown(final_mission)

    with tab_eli5:
        if not ai_ready(model):
            st.info("Study-mode AI tools need the AI endpoint configured.")
        else:
            explanation = st.text_area("Paste the explanation you want simplified", height=140, placeholder="Write a short explanation here…")
            if st.button("Make it ELI5", key="eli5_btn"):
                if explanation.strip():
                    with st.spinner("Simplifying your explanation…"):
                        st.session_state["eli5_output"] = simplify_for_eli5(model, subject_name, topic, explanation)
                    st.markdown(st.session_state["eli5_output"])
                else:
                    st.warning("Please enter a short explanation to simplify.")

    with tab_map:
        if not ai_ready(model):
            st.info("Study-mode AI tools need the AI endpoint configured.")
        else:
            related_topic = st.text_input("Related concept", placeholder="e.g. respiration", key="concept_map_topic")
            if st.button("Weave the concept map", key="concept_map_btn"):
                if related_topic.strip():
                    with st.spinner("Connecting the concepts…"):
                        st.session_state["concept_map_data"] = connect_concepts(model, subject_name, topic, related_topic)
                else:
                    st.warning("Please enter a related concept.")

            if st.session_state.get("concept_map_data"):
                data = st.session_state["concept_map_data"]
                central_topic = data.get("central_topic", topic)
                connections = data.get("connections", [])
                key_points = data.get("key_points", [])

                # Visual concept map as blocks
                st.markdown("### 🧩 Visual Concept Map")

                # Central topic block
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffd9a8, #b6f1d8); color: #1b2733; padding: 1rem; border-radius: 12px; text-align: center; font-weight: 700; font-size: 1.2rem; margin: 1rem 0; border: 1px solid #9ed9c0;">
                    {central_topic}
                </div>
                """, unsafe_allow_html=True)

                # Connection blocks
                if connections:
                    for i, conn in enumerate(connections):
                        from_topic = conn.get("from", "")
                        to_topic = conn.get("to", "")
                        relationship = conn.get("relationship", "")

                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 0.8rem 0;">
                            <div style="background: #ffffff; color: #13293d; border: 2px solid #5d9dd8; padding: 0.8rem; border-radius: 10px; flex: 1; text-align: center; font-weight: 600;">
                                {from_topic}
                            </div>
                            <div style="flex: 0 0 100px; text-align: center; color: #1d4f73; font-weight: bold; font-size: 0.9rem;">
                                → {relationship} →
                            </div>
                            <div style="background: #ffffff; color: #13293d; border: 2px solid #4fc08d; padding: 0.8rem; border-radius: 10px; flex: 1; text-align: center; font-weight: 600;">
                                {to_topic}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Key points
                if key_points:
                    st.markdown("### 📌 Key Points")
                    for point in key_points:
                        st.markdown(f"""
                        <div style="background: #f9fcff; color: #13293d; border-left: 4px solid #2f80ed; padding: 0.8rem; margin: 0.5rem 0; border-radius: 4px;">
                            {point}
                        </div>
                        """, unsafe_allow_html=True)

    with tab_sheet:
        if not ai_ready(model):
            st.info("Study-mode AI tools need the AI endpoint configured.")
        else:
            if st.button("Generate cheat sheet", type="primary", key="cheat_sheet_btn"):
                with st.spinner("Creating your revision sheet…"):
                    st.session_state["cheat_sheet_output"] = generate_cheat_sheet(model, subject_name, topic)
            if st.session_state.get("cheat_sheet_output"):
                st.markdown(st.session_state["cheat_sheet_output"])


def _play_lecture(lec, model, active_subject, grade_level=None):
    st.markdown(f"### {lec['title']}")
    if lec.get("description"):
        st.caption(lec["description"])
    path = VIDEO_DIR / lec["video"]
    if path.exists():
        st.video(str(path))
    else:
        st.error("Video file is missing (it may have been reset on redeploy).")

    tab_notes, tab_ai, tab_teacher, tab_quiz = st.tabs(["📄 Notes & summary", "🤖 Ask AI tutor", "👩‍🏫 Ask your teacher", "🧠 Quiz me"])

    with tab_notes:
        st.markdown(lec.get("notes") or "_No notes were added for this lecture._")
        if ai_ready(model) and lec.get("notes"):
            if st.button("✨ Summarise for revision", key=f"sum_{lec['id']}"):
                with st.spinner("Summarising…"):
                    st.session_state[f"summary_{lec['id']}"] = summarize(model, lec)
            if st.session_state.get(f"summary_{lec['id']}"):
                st.info(st.session_state[f"summary_{lec['id']}"])

    with tab_ai:
        if not ai_ready(model):
            st.info("The AI tutor isn't configured in this environment.")
        else:
            q = st.text_input("Ask anything about this lecture",
                              key=f"q_{lec['id']}", placeholder="e.g. Why is chlorophyll important?")
            if st.button("Ask AI tutor", key=f"ask_{lec['id']}", type="primary") and q.strip():
                with st.spinner("Thinking…"):
                    st.session_state[f"ans_{lec['id']}"] = ask_tutor(model, lec, q)
            if st.session_state.get(f"ans_{lec['id']}"):
                st.markdown(st.session_state[f"ans_{lec['id']}"])

    with tab_teacher:
        st.caption("Send a question directly to your teacher about this lecture or topic.")
        teacher_subject = lec.get("subject") or active_subject
        teachers = get_teachers_for_subject(teacher_subject, grade_level=grade_level)
        teacher_names = ["No specific teacher"] + [t[1] for t in teachers]
        teacher_choice = st.selectbox("Send to", teacher_names, key=f"teacher_choice_{lec['id']}")
        teacher_id = None
        if teacher_choice != "No specific teacher":
            teacher_id = next(t[0] for t in teachers if t[1] == teacher_choice)
        question = st.text_area("Your teacher question", height=160,
                                 key=f"teacher_q_{lec['id']}",
                                 placeholder="e.g. I’m stuck on this part — can you explain it a little more?")
        if st.button("Send to teacher", key=f"teacher_submit_{lec['id']}", type="primary"):
            if not question.strip():
                st.warning("Please write a question before sending it.")
            else:
                save_teacher_question(
                    st.session_state.get("student_id"),
                    teacher_subject,
                    teacher_id,
                    question,
                    st.session_state.get("student_name"),
                )
                st.success("Your question has been sent to your teacher.")

        st.divider()
        st.markdown("#### Your questions and replies")
        previous_questions = get_student_teacher_questions(st.session_state.get("student_id"))
        subject_questions = [row for row in previous_questions if row[1] == teacher_subject]
        if not subject_questions:
            st.info("No replies yet for this subject.")
        else:
            for question_id, subject_name, question_text, answer_text, answered_by, answered_at, created_at in subject_questions:
                with st.expander(f"{subject_name} · {created_at}"):
                    st.write(question_text)
                    if answer_text:
                        st.success(f"Answered by {answered_by or 'Teacher'}{f' at {answered_at}' if answered_at else ''}")
                        st.write(answer_text)
                    else:
                        st.info("Waiting for a teacher reply.")

    with tab_quiz:
        if not ai_ready(model):
            st.info("Quizzes need the AI, which isn't configured here.")
        else:
            _quiz_ui(lec, model)


def _quiz_ui(lec, model):
    qkey = f"quiz_{lec['id']}"
    if st.button("🎯 Make me a quiz", key=f"mkquiz_{lec['id']}"):
        with st.spinner("Writing your quiz…"):
            quiz = make_quiz(model, lec)
            if not quiz:
                quiz = _quiz_bank_fallback(lec.get("subject"), count=10)
            st.session_state[qkey] = quiz
            st.session_state[f"{qkey}_submitted"] = False
    quiz = st.session_state.get(qkey)
    if not quiz:
        return
    normalized_quiz = []
    for item in quiz:
        if not isinstance(item, dict):
            continue
        options = item.get("options") or []
        if not options:
            continue
        try:
            answer_index = int(item.get("answer_index", 0))
        except (TypeError, ValueError):
            answer_index = 0
        if answer_index < 0 or answer_index >= len(options):
            answer_index = 0
        normalized_quiz.append({
            "q": str(item.get("q") or item.get("question") or "").strip(),
            "options": [str(opt) for opt in options],
            "answer_index": answer_index,
            "explanation": str(item.get("explanation") or item.get("hint") or "Review the related topic notes and key definitions."),
        })
    quiz = normalized_quiz
    if not quiz:
        st.info("No quiz questions could be generated for this lecture right now.")
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


# --------------------------------------------------------------------------- #
# UI — Student sign-up wizard
#
# A branded, 3-step onboarding flow: details -> subjects & teachers -> a
# generated "digital enrollment card" + an AI concierge welcome note. This is
# the piece that's meant to stand out from a plain sign-up form.
# --------------------------------------------------------------------------- #
_HERO_CSS = """
<style>
.sh-hero {
    background: linear-gradient(135deg, #0f6a8f 0%, #17a1a5 100%);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(108,92,231,0.30);
}
.sh-hero h1 { margin: 0; font-size: 1.9rem; }
.sh-hero p { opacity: .92; margin-top: .5rem; font-size: 1rem; }
.sh-step-pill {
    display: inline-block; padding: .32rem 1rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; margin-right: .4rem; margin-bottom: .6rem;
}
.sh-step-active { background: #0f6a8f; color: white; }
.sh-step-done   { background: #17a1a5; color: white; }
.sh-step-todo   { background: #eaf4fa; color: #5c7182; }
.sh-id-card {
    background: linear-gradient(135deg, #15374d, #0f6a8f);
    border-radius: 18px; padding: 1.6rem 1.7rem; color: white;
    max-width: 460px; box-shadow: 0 8px 26px rgba(0,0,0,.28);
}
.sh-id-avatar {
    width: 54px; height: 54px; border-radius: 50%;
    background: linear-gradient(135deg, #17a1a5, #f39c12);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.25rem; margin-bottom: .7rem;
}
.sh-id-name { font-size: 1.15rem; font-weight: 700; }
.sh-id-meta { opacity: .7; font-size: .82rem; margin-bottom: .7rem; }
.sh-subject-badge {
    display: inline-block; background: rgba(255,255,255,.12);
    padding: .28rem .75rem; border-radius: 999px; font-size: .78rem;
    margin: .2rem .3rem .2rem 0;
}
.sh-note {
    border-left: 4px solid #0f6a8f; background: rgba(15,106,143,.08);
    padding: .9rem 1.1rem; border-radius: 8px; margin-top: .8rem;
}
</style>
"""
 
_STEP_LABELS = ["1 · Your details", "2 · Subjects & teachers", "3 · Confirm"]
 
 
def _inject_hero_css():
    st.markdown(_HERO_CSS, unsafe_allow_html=True)
 
 
def _render_hero():
    st.markdown(
        """<div class="sh-hero">
        <h1>🚀 Join the Study Hub</h1>
        <p>Set up your profile once — pick your subjects, choose the teachers you vibe with,
        and get a personalized welcome note from our AI concierge.</p>
        </div>""",
        unsafe_allow_html=True,
    )
 
 
def _render_steps(current):
    html = ""
    for i, label in enumerate(_STEP_LABELS, start=1):
        if i == current:
            cls = "sh-step-active"
        elif i < current:
            cls = "sh-step-done"
        else:
            cls = "sh-step-todo"
        html += f'<span class="sh-step-pill {cls}">{label}</span>'
    st.markdown(html, unsafe_allow_html=True)
 
 
def _init_signup_state():
    st.session_state.setdefault("signup_step", 1)
    st.session_state.setdefault("signup_data", {})
    st.session_state.setdefault("editing_prefs", False)
 
 
def _step_details():
    st.subheader("Step 1 — Tell us about you")
    st.info("Your roll number will be assigned automatically when you confirm registration.")

    with st.expander("Already registered? Continue with your roll number", expanded=False):
        with st.form("student_login"):
            existing_roll = st.text_input("Roll number", placeholder="e.g. 2026002")
            existing_password = st.text_input("Password", type="password")
            login_pressed = st.form_submit_button("Login as student", type="primary")

        if login_pressed:
            if not existing_roll.strip() or not existing_password:
                st.warning("Please enter both roll number and password.")
            else:
                existing = authenticate_student(existing_roll.strip(), existing_password)
                if existing:
                    st.success(f"Welcome back, {existing[1]}.")
                    st.session_state.student_id = existing[0]
                    st.session_state.student_name = existing[1]
                    st.rerun()
                else:
                    st.error("Invalid roll number or password.")

    data = st.session_state.signup_data
    roll_preview = data.get("roll") or "Will be generated on confirmation"
    st.caption(f"Roll number: {roll_preview}")

    name = st.text_input("Full name", value=data.get("name", ""))
    email = st.text_input("Email address", value=data.get("email", ""),
                          placeholder="you@example.com")
    password = st.text_input("Create password", type="password", value=data.get("password", ""))
    confirm_password = st.text_input("Confirm password", type="password", value=data.get("confirm_password", ""))
    grade_options = GRADE_LEVEL_OPTIONS
    existing_grade = data.get("grade", "")
    default_grade_idx = grade_options.index(existing_grade) if existing_grade in grade_options else 0
    grade_choice = st.selectbox("Grade level", grade_options, index=default_grade_idx, key="signup_grade_choice")
    grade = grade_choice
 
    _, c2 = st.columns([1, 1])
    if c2.button("Next: Subjects →", type="primary"):
        if not (name.strip() and email.strip()):
            st.warning("Name and email are required.")
        elif "@" not in email:
            st.warning("Please enter a valid email address.")
        elif len(password) < 6:
            st.warning("Please create a password with at least 6 characters.")
        elif password != confirm_password:
            st.warning("Passwords do not match.")
        else:
            st.session_state.signup_data.update({
                "name": name.strip(),
                "email": email.strip(),
                "password": password,
                "confirm_password": confirm_password,
                "grade": (grade or "").strip(),
            })
            st.session_state.signup_step = 2
            st.rerun()
 
 
def _step_subjects():
    st.subheader("Step 2 — Pick your subjects & preferred teachers")
    st.caption("Choose every subject you're taking. For each one you can optionally pick "
              "the teacher you'd like, and rank how important that pick is to you.")
 
    existing_map = st.session_state.signup_data.get("subjects", {})
    chosen = st.multiselect("Which subjects are you taking?", SUBJECTS,
                            default=list(existing_map.keys()))
 
    new_map = {}
    for subj in chosen:
        selected_grade = st.session_state.signup_data.get("grade", "")
        teachers = get_teachers_for_subject(subj, grade_level=selected_grade)
        names = ["No preference"] + [t[1] for t in teachers]
        prev = existing_map.get(subj, {})
        default_idx = names.index(prev["teacher_name"]) if prev.get("teacher_name") in names else 0
 
        col1, col2 = st.columns([2, 1])
        with col1:
            t_choice = st.selectbox(f"Preferred teacher — {subj}", names,
                                    index=default_idx, key=f"tch_{subj}")
        with col2:
            pr = st.number_input("Priority", min_value=1, max_value=5,
                                 value=prev.get("priority", 1), step=1, key=f"pr_{subj}")
 
        teacher_id = None
        if t_choice != "No preference":
            teacher_id = next(t[0] for t in teachers if t[1] == t_choice)
        new_map[subj] = {"teacher_id": teacher_id, "teacher_name": t_choice, "priority": int(pr)}
 
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        if st.session_state.editing_prefs:
            st.session_state.editing_prefs = False
        else:
            st.session_state.signup_step = 1
        st.rerun()
    if c2.button("Next: Review →", type="primary"):
        if not chosen:
            st.warning("Pick at least one subject.")
        else:
            st.session_state.signup_data["subjects"] = new_map
            st.session_state.signup_step = 3
            st.rerun()
 
 
def _step_confirm(model):
    st.subheader("Step 3 — Review & confirm")
    data = st.session_state.signup_data
    subjects = data.get("subjects", {})
 
    initials = "".join(p[0].upper() for p in (data.get("name") or "?").split()[:2]) or "?"
    badges = "".join(
        f'<span class="sh-subject-badge">{s} · {v["teacher_name"]}</span>'
        for s, v in subjects.items()
    )
    st.markdown(f"""
    <div class="sh-id-card">
      <div class="sh-id-avatar">{initials}</div>
      <div class="sh-id-name">{data.get('name', '')}</div>
            <div class="sh-id-meta">Roll: {data.get('roll') or 'Auto-generated on confirmation'} · {data.get('grade') or '—'} · {data.get('email', '')}</div>
      <div>{badges}</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.write("")
    c1, c2 = st.columns([1, 1])
    if c1.button("← Back"):
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("✅ Confirm & join", type="primary"):
        if st.session_state.get("student_id"):
            student_id = st.session_state.student_id
        else:
            student_id, assigned_roll = register_student(
                data["name"],
                None,
                data.get("grade", ""),
                data["email"],
                data.get("password", ""),
            )
            if student_id is None:
                st.error("We could not create your registration right now. Please try again.")
                return
            st.session_state.signup_data["roll"] = assigned_roll
            st.session_state.student_id = student_id
            st.session_state.student_name = data["name"]
            st.success(f"Your roll number is {assigned_roll}.")
 
        for subj, v in subjects.items():
            submit_preference(student_id, subj, v["teacher_id"], v["priority"])
 
        if ai_ready(model) and not st.session_state.get("welcome_note"):
            with st.spinner("Your AI concierge is writing you a welcome note…"):
                note = generate_welcome_note(model, data.get("name", ""), list(subjects.keys()))
            if note:
                st.session_state["welcome_note"] = note
 
        st.session_state.editing_prefs = False
        st.balloons()
        st.rerun()
 
 
def _signup_dashboard():
    st.markdown(f"### 🎉 You're all set, {st.session_state.student_name}!")

    premium_until = get_user_premium_until("Student")
    if premium_until and premium_until >= _today_date():
        days_left = (premium_until - _today_date()).days
        st.success(f"⭐ Premium active until {premium_until.isoformat()} ({days_left} day(s) left)")
    else:
        st.info("Free plan active: 10 AI questions per day. Upgrade from the sidebar payment panel for unlimited AI.")
 
    if st.session_state.get("welcome_note"):
        st.markdown(f'<div class="sh-note">💬 {st.session_state["welcome_note"]}</div>',
                    unsafe_allow_html=True)
 
    rows = view_student_preferences(st.session_state.student_id)
    if rows:
        st.markdown("#### Your enrolled subjects")
        for subject_name, teacher_name, priority in rows:
            st.write(f"{priority}. **{subject_name}** → {teacher_name or 'No preference'}")
    else:
        st.info("You haven't picked any subjects yet.")
 
    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("➕ Update subjects / teachers"):
        srow = get_student_by_id(st.session_state.student_id)
        if srow:
            st.session_state.signup_data = {
                "name": srow[1], "roll": srow[2], "grade": srow[3] or "", "email": srow[4] or "",
                "subjects": {
                    subj: {"teacher_id": None, "teacher_name": teacher or "No preference", "priority": pr}
                    for subj, teacher, pr in rows
                },
            }
        st.session_state.editing_prefs = True
        st.session_state.signup_step = 2
        st.rerun()
    if c2.button("Switch student"):
        for k in ("student_id", "student_name", "signup_step", "signup_data",
                  "editing_prefs", "welcome_note"):
            st.session_state.pop(k, None)
        st.rerun()
 
 
def student_signup_wizard(model):
    """Entry point for the sign-up / preferences tab."""
    _inject_hero_css()
    _render_hero()
    _init_signup_state()
 
    if st.session_state.get("student_id") and not st.session_state.editing_prefs:
        _signup_dashboard()
        return
 
    step = st.session_state.signup_step
    _render_steps(step)
    st.write("")
 
    if step == 1:
        _step_details()
    elif step == 2:
        _step_subjects()
    elif step == 3:
        _step_confirm(model)
 
 
# --------------------------------------------------------------------------- #
def student_view(model):
    # Lectures are locked behind sign-up: a student must have an account and
    # have gone through "Sign up & preferences" before the lecture library
    # becomes visible.
    if not st.session_state.get("student_id"):
        st.info("👋 **Welcome!** Please sign up below to unlock your lectures.")
        student_signup_wizard(model)
        return

    preferences = view_student_preferences(st.session_state.student_id)
    student_subjects = [subject_name for subject_name, _, _ in preferences if subject_name in SUBJECTS]
    if not student_subjects:
        student_subjects = SUBJECTS[:3]
    student_row = get_student_by_id(st.session_state.student_id)
    student_grade = student_row[3] if student_row else None
    subject_options = ["All selected subjects"] + student_subjects
    st.session_state.setdefault("selected_subject", subject_options[0])
    active_subject = st.selectbox("📚 Choose a subject", subject_options, key="selected_subject")
    is_all_subjects = active_subject == "All selected subjects"

    tab_lectures, tab_flashcards, tab_modes, tab_checklist, tab_grader, tab_performance, tab_signup, tab_resources = st.tabs([
        "🎬 Lectures",
        "🧠 Flashcards",
        "🛡️ Study modes",
        "📘 Syllabus checklist",
        "📝 Past Paper Grader",
        "📊 Performance",
        "🚀 Sign up & preferences",
        "📚 Resources",
    ])

    with tab_lectures:
        items = load_index()
        if not items:
            st.info("📭 No lectures yet. Ask your teacher to switch to **Teacher** mode and "
                    "upload one!")
        else:
            if is_all_subjects:
                in_subject = [it for it in items if it["subject"] in student_subjects and (not it.get("grade_level") or it.get("grade_level") == student_grade)]
            else:
                in_subject = [it for it in items if it["subject"] == active_subject and (not it.get("grade_level") or it.get("grade_level") == student_grade)]
            if not in_subject:
                st.info("No lectures are available for this selection yet.")
            else:
                titles = [it["title"] for it in in_subject]
                picked = st.selectbox("🎬 Choose a lecture", range(len(in_subject)),
                                      format_func=lambda i: titles[i])
                current_lecture_id = in_subject[picked]["id"]
                previous_lecture_id = st.session_state.get("active_lecture_id")
                if previous_lecture_id != current_lecture_id:
                    if previous_lecture_id:
                        st.session_state.pop(f"summary_{previous_lecture_id}", None)
                        st.session_state.pop(f"quiz_{previous_lecture_id}", None)
                        st.session_state.pop(f"quiz_{previous_lecture_id}_submitted", None)
                    st.session_state["active_lecture_id"] = current_lecture_id
                st.divider()
                _play_lecture(in_subject[picked], model, in_subject[picked]["subject"], grade_level=student_grade)

    with tab_flashcards:
        _render_flashcards(student_subjects if is_all_subjects else [active_subject], grade_level=student_grade)

    with tab_modes:
        if is_all_subjects:
            st.info("Choose one subject (not 'All') to use Study Modes.")
        else:
            _render_study_modes(model, active_subject, grade_level=student_grade)

    with tab_checklist:
        if is_all_subjects:
            for subj in student_subjects:
                with st.expander(f"{subj} checklist", expanded=False):
                    _render_syllabus_checklist(subj, grade_level=student_grade)
        else:
            _render_syllabus_checklist(active_subject, grade_level=student_grade)

    with tab_grader:
        if is_all_subjects:
            st.info("Choose one subject (not 'All') to open the paper grader.")
        else:
            _render_past_paper_grader(model, active_subject, grade_level=student_grade)

    with tab_performance:
        st.subheader("📊 Student performance")
        summary = get_student_performance_summary(st.session_state.student_id)
        if summary["total"] == 0:
            st.info("No graded work yet. Submit a paper or worksheet to start building your performance record.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Average %", f"{summary['average_percentage']}%")
            metric_cols[1].metric("Grade", summary['grade_band'])
            metric_cols[2].metric("Best score", f"{summary['best_percentage']}%")
            metric_cols[3].metric("Assessments", summary['total'])

            st.progress(max(0.0, min(summary['average_percentage'] / 100.0, 1.0)))
            threshold = get_student_threshold(st.session_state.student_id)
            st.caption(f"Your target threshold: {threshold}%")
            st.caption(get_performance_recommendation(summary['average_percentage']))
            if summary['average_percentage'] < threshold:
                st.warning("You are under your target. Try the next intervention step below.")
                for item in build_intervention_plan(summary['average_percentage'], threshold):
                    st.markdown(f"- {item['message']}")

            with st.expander("Adjust your intervention target"):
                desired_threshold = st.slider("Target performance (%)", min_value=40, max_value=90, value=threshold, key="student_threshold_slider")
                if st.button("Save target", key="save_threshold"):
                    update_student_threshold(st.session_state.student_id, desired_threshold)
                    st.success(f"Target saved at {desired_threshold}%")

            if summary['latest']:
                latest_title, latest_subject, latest_total, latest_final = summary['latest'][2], summary['latest'][3], summary['latest'][4], summary['latest'][8]
                latest_percentage = round((latest_final / latest_total * 100), 1) if latest_final is not None and latest_total else 0
                st.info(f"Latest update: {latest_title} in {latest_subject} is at {latest_percentage}%.")

            st.divider()
            st.markdown("#### Subject overview")
            for subject, pct in sorted(summary['subject_breakdown'].items()):
                st.write(f"**{subject}**")
                st.progress(max(0.0, min(pct / 100.0, 1.0)))
                st.caption(f"Average performance: {pct}%")

            st.divider()
            st.markdown("#### Recent submissions")
            submissions = get_submissions_for_student(st.session_state.student_id)
            for submission_id, assessment_id, title, subject_name, total_marks, grading_mode, ai_score, teacher_score, final_score, status, completed_in_time, ai_feedback, teacher_feedback, submitted_at in submissions:
                percentage = round((final_score / total_marks * 100), 1) if final_score is not None and total_marks else 0
                grade = get_grade_band(percentage)
                with st.container():
                    st.markdown(f"**{title}** · {subject_name}")
                    st.caption(f"Submitted {submitted_at} · {grading_mode} · Completed in time: {'Yes' if completed_in_time else 'No'}")
                    st.write(f"Score: {final_score}/{total_marks}  ·  Percentage: {percentage}%  ·  Grade: {grade}")
                    if teacher_feedback or ai_feedback:
                        st.write(get_performance_recommendation(percentage, teacher_feedback or ai_feedback))
                    st.divider()

    with tab_signup:
        student_signup_wizard(model)

    with tab_resources:
        st.subheader("📚 Study Resources")
        st.caption("Free textbooks, teacher notes, and past papers for your subjects")

        res_tab_syllabus, res_tab_textbooks, res_tab_notes, res_tab_papers = st.tabs(["📘 Syllabus", "📖 Textbooks", "📝 Teacher Notes", "📄 Past Papers"])

        with res_tab_syllabus:
            target_subjects = student_subjects if is_all_subjects else [active_subject]
            for subj in target_subjects:
                st.markdown(f"#### {subj}")
                docs = get_syllabus_documents(subj, grade_level=student_grade)
                if not docs:
                    st.info("No syllabus uploaded yet.")
                    continue
                for syllabus_id, title, file_path, chapter_outline, created_at in docs:
                    with st.expander(f"{title} ({created_at})"):
                        if chapter_outline:
                            st.text(chapter_outline)
                        if file_path and Path(file_path).exists():
                            with open(file_path, "rb") as fh:
                                st.download_button("Download syllabus", fh.read(), file_name=Path(file_path).name, mime="application/octet-stream", key=f"syllabus_dl_{syllabus_id}")

        with res_tab_textbooks:
            if is_all_subjects:
                textbooks = []
                for subj in student_subjects:
                    textbooks.extend(get_textbooks_for_subject(subj))
            else:
                textbooks = get_textbooks_for_subject(active_subject)
            if not textbooks:
                st.info("No textbooks available for this selection yet.")
            else:
                for textbook_id, title, author, description, resource_type, file_path, external_url in textbooks:
                    with st.expander(f"📖 {title}"):
                        if author:
                            st.caption(f"Author: {author}")
                        if description:
                            st.write(description)
                        if external_url:
                            st.link_button("Open Resource", external_url)
                        if file_path and Path(file_path).exists():
                            with open(file_path, "rb") as fh:
                                st.download_button("Download", fh.read(), file_name=Path(file_path).name, mime="application/octet-stream")

        with res_tab_notes:
            if is_all_subjects:
                notes = []
                for subj in student_subjects:
                    notes.extend(get_teacher_notes_for_subject(subj))
            else:
                notes = get_teacher_notes_for_subject(active_subject)
            if not notes:
                st.info("No teacher notes available for this selection yet.")
            else:
                for note_id, title, content, chapter, created_at in notes:
                    with st.expander(f"📝 {title}"):
                        if chapter:
                            st.caption(f"Chapter: {chapter}")
                        st.caption(f"Added: {created_at}")
                        st.write(content)

        with res_tab_papers:
            if is_all_subjects:
                papers = []
                for subj in student_subjects:
                    papers.extend(get_past_papers_for_subject(subj, grade_level=student_grade))
            else:
                papers = get_past_papers_for_subject(active_subject, grade_level=student_grade)
            if not papers:
                st.info("No past papers available for this selection yet.")
            else:
                for paper_id, year, paper_type, season, paper_number, qp_path, ms_path, er_path, duration_minutes in papers:
                    with st.expander(f"{year} {season} - {paper_type} Paper {paper_number}"):
                        col1 = st.columns(1)[0]
                        if qp_path and Path(qp_path).exists():
                            with open(qp_path, "rb") as fh:
                                col1.download_button("Question Paper", fh.read(), file_name=Path(qp_path).name, mime="application/octet-stream")
                        st.caption("Mark scheme and examiner report unlock after you submit in the grader with self-check enabled.")
 
 
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="ScholarWave Learning Hub | KG to A-Level", page_icon="🌊", layout="wide")
    _render_mobile_install_prompt()

    _backup_data_files()

    create_tables()
    create_syllabus_table()
    create_teacher_questions_table()
    create_flashcards_table()
    create_study_streaks_table()
    create_textbooks_table()
    create_assessment_tables()
    seed_subjects()
    seed_syllabus_chapters()
    seed_flashcards()

    _render_org_header()
    _render_corner_suggestion_box()

    st.sidebar.title("🌊 ScholarWave Hub")
    role = _render_role_selector()
    _render_mode_toggle()
    if role == "Student":
        _render_feedback_and_payment_panel()
    
    if role == "Student":
        _render_pomodoro_sidebar()
        _render_study_streak_sidebar()
    
    model = DEFAULT_MODEL
    if not supabase:
        st.sidebar.info("Supabase is not configured. App is using local SQLite fallback for data storage.")

    if not ai_ready(model):
        st.sidebar.warning("AI features are off (no key set) — video + notes still work.")
    else:
        if is_user_premium(role):
            premium_until = get_user_premium_until(role)
            if premium_until:
                st.sidebar.caption(f"AI daily quota ({role}): Premium active until {premium_until.isoformat()} (unlimited AI)")
            else:
                st.sidebar.caption(f"AI daily quota ({role}): Premium active (unlimited AI)")
        else:
            role_usage = get_ai_usage_count(role=role)
            remaining = max(DAILY_AI_QUOTA - role_usage, 0)
            st.sidebar.caption(f"AI daily quota ({role}): {role_usage}/{DAILY_AI_QUOTA} used ({remaining} left)")
    st.sidebar.caption(f"{len(load_index())} lecture(s) available.")

    if role == "Teacher":
        st.title("👩‍🏫 Teacher dashboard")
        teacher_view(model)
    else:
        st.title("🎯 Study smarter")
        st.caption("Pick a subject, watch the lecture, and let your AI tutor make revision feel lighter and brighter.")
        student_view(model)

    _render_seo_keywords_footer()
 
 
if __name__ == "__main__":
    main()