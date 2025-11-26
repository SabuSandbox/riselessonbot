# app.py
import os
import tempfile
import json
import time
import nltk
from flask import Flask, request, jsonify
import requests
from docx import Document
from duckduckgo_search import ddg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# ================== NLTK setup ==================
NLTK_DATA_DIR = os.environ.get("NLTK_DATA_DIR", "/opt/render/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DEFAULT_TEMPLATE_PATH = os.environ.get("DEFAULT_TEMPLATE_PATH", "./Sample Lesson Plan.docx")
PORT = int(os.environ.get("PORT", 5000))
# Admin ID (can be overridden via env var). Default from your message.
ADMIN_ID = int(os.environ.get("ADMIN_ID", "7925575742"))
# Target user ID to message (recommended to set as Render secret). Can be None.
TARGET_USER_ID_ENV = os.environ.get("TARGET_USER_ID")  # string or None

if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN env var")

BASE_TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
app = Flask(__name__)

# In-memory sessions (ephemeral)
SESS = {}  # chat_id -> state dict
# runtime override for target (admin can set during runtime)
RUNTIME_TARGET = {}

# ================== Telegram helpers ==================
def telegram_api(method, params=None, files=None, json_payload=None):
    url = f"{BASE_TELEGRAM_URL}/{method}"
    try:
        if files:
            return requests.post(url, params=params, files=files, timeout=30)
        if json_payload:
            return requests.post(url, json=json_payload, timeout=30)
        return requests.post(url, data=params, timeout=30)
    except Exception as e:
        print("telegram_api error:", e)
        raise

def send_message(chat_id, text, reply_markup=None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        telegram_api("sendMessage", params=payload)
    except Exception as e:
        print("send_message error:", e)

def download_file(file_id, dest_path):
    r = telegram_api("getFile", params={"file_id": file_id})
    r.raise_for_status()
    data = r.json()
    file_path = data["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    r2 = requests.get(file_url, timeout=60)
    r2.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r2.content)
    return dest_path

# ================== Text extraction helpers ==================
def extract_text_from_pdf(path):
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
            except Exception:
                continue
    except Exception as e:
        print("PDF read error:", e)
        raise
    return "\n".join(text_parts)

def extract_text_from_url(url, max_chars=20000):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        print("fetch error for", url, ":", e)
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n")
    else:
        ps = soup.find_all("p")
        filtered = []
        for p in ps:
            t = p.get_text(strip=True)
            if not t:
                continue
            if len(t) < 30:
                continue
            filtered.append(t)
        text = "\n\n".join(filtered)
    if not text or len(text.strip()) < 100:
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc_tag = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
        meta = desc_tag.get("content").strip() if desc_tag and desc_tag.get("content") else ""
        text = (title + "\n" + meta).strip()
    return (text or "")[:max_chars]

# ================== Summarization & heuristics ==================
def summarize_text(text, sentences_count=6):
    if not text:
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)
        return "\n".join(str(s) for s in summary_sentences)
    except Exception as e:
        print("summarize_text error:", e)
        # fallback: return first N lines/segments
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines[:sentences_count]) if lines else text[:1000]

def extract_objectives_from_text(text, max_points=5):
    candidates = []
    for sent in (text or "").split("."):
        s = sent.strip()
        if not s:
            continue
        sl = s.lower()
        if any(k in sl for k in ("able to", "will", "understand", "learn", "identify", "describe")):
            candidates.append(s.strip())
    if candidates:
        return "\n".join(f"• {c}" for c in candidates[:max_points])
    summ = summarize_text(text, sentences_count=max_points)
    if summ:
        return "\n".join(f"• {s.strip()}" for s in summ.split("\n") if s.strip())
    return "• Objective 1\n• Objective 2"

def generate_activities(text, max_items=4):
    return (
        "1. Read the summary and discuss key terms.\n"
        "2. Small-group activity: identify examples from the text.\n"
        "3. Hands-on/demo (if applicable): follow the experiment steps.\n"
        "4. Exit ticket: one short question to assess learning."
    )

def generate_assessment_questions(text, max_q=4):
    summary = summarize_text(text, sentences_count=4)
    qs = []
    for i, s in enumerate(summary.split("\n")[:max_q], start=1):
        s = s.strip().rstrip(".")
        if len(s.split()) < 3:
            continue
        qs.append(f"Q{i}. Explain: {s}?")
    if not qs:
        qs = ["Q1. What is the main idea of the chapter?", "Q2. List two key points."]
    return "\n".join(qs)

# ================== Template fill & send ==================
def fill_template_and_send(chat_id, title, summary, objectives, activities, assessment, references=""):
    template_path = SESS.get(chat_id, {}).get("template_path") or DEFAULT_TEMPLATE_PATH
    if not os.path.exists(template_path):
        send_message(chat_id, "Template not found on server. Please upload a .docx template or set DEFAULT_TEMPLATE_PATH.")
        return
    replacements = {
        "{{ChapterTitle}}": title,
        "{{Summary}}": summary,
        "{{LearningObjectives}}": objectives,
        "{{Activities}}": activities,
        "{{Assessment}}": assessment,
        "{{References}}": references
    }
    out_tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    out_path = out_tmp.name
    doc = Document(template_path)
    for p in doc.paragraphs:
        for k, v in replacements.items():
            if k in p.text:
                p.text = p.text.replace(k, v)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for k, v in replacements.items():
                    if k in cell.text:
                        cell.text = cell.text.replace(k, v)
    doc.save(out_path)
    try:
        with open(out_path, "rb") as f:
            files = {"document": (os.path.basename(out_path), f)}
            telegram_api("sendDocument", params={"chat_id": chat_id}, files=files)
        send_message(chat_id, "Lesson plan generated ✅")
    except Exception as e:
        print("sendDocument error:", e)
        send_message(chat_id, f"Failed to send generated file: {e}")

# ================== Admin helpers ==================
def is_admin(chat_id):
    return int(chat_id) == int(ADMIN_ID)

def get_current_target():
    # runtime override first, then env var
    rt = RUNTIME_TARGET.get("target")
    if rt:
        return str(rt)
    if TARGET_USER_ID_ENV:
        return str(TARGET_USER_ID_ENV)
    return None

# ================== Webhook & conversational flow ==================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(force=True)
    if not update or "message" not in update:
        return jsonify({"ok": True})
    msg = update["message"]
    chat_id = msg["chat"]["id"]
    SESS.setdefault(chat_id, {"state": "idle", "tmp": {}, "template_path": None})

    # Admin-only commands: /admin, /settarget, /showtarget, /sendtarget, /settemplate
    if "text" in msg:
        text = msg["text"].strip()
        # /admin opens admin menu
        if text.startswith("/admin"):
            if not is_admin(chat_id):
                send_message(chat_id, "Unauthorized. Only admin can use this command.")
                return jsonify({"ok": True})
            kb = {"keyboard":[["Send Message to Target"],["Show Target"],["Set Target"],["Set Template Path"],["Exit Admin"]],"one_time_keyboard":True,"resize_keyboard":True}
            send_message(chat_id, "Admin menu — choose an action:", reply_markup=kb)
            SESS[chat_id]["state"] = "admin_menu"
            return jsonify({"ok": True})

        # Short admin CLI commands: /settarget <id>, /showtarget, /sendtarget <message>
        if text.startswith("/settarget"):
            if not is_admin(chat_id):
                send_message(chat_id, "Unauthorized.")
                return jsonify({"ok": True})
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                RUNTIME_TARGET["target"] = parts[1].strip()
                send_message(chat_id, f"Runtime target set to: {RUNTIME_TARGET['target']}")
            else:
                send_message(chat_id, "Usage: /settarget <chat_id>")
            return jsonify({"ok": True})

        if text.startswith("/showtarget"):
            if not is_admin(chat_id):
                send_message(chat_id, "Unauthorized.")
                return jsonify({"ok": True})
            cur = get_current_target()
            send_message(chat_id, f"Current target: {cur or 'None'}")
            return jsonify({"ok": True})

        if text.startswith("/sendtarget"):
            if not is_admin(chat_id):
                send_message(chat_id, "Unauthorized.")
                return jsonify({"ok": True})
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                send_message(chat_id, "Usage: /sendtarget <message>")
                return jsonify({"ok": True})
            target = get_current_target()
            if not target:
                send_message(chat_id, "No target set. Use /settarget or set TARGET_USER_ID env var.")
                return jsonify({"ok": True})
            # send the message to target
            send_message(target, parts[1].strip())
            send_message(chat_id, f"Message sent to {target}")
            return jsonify({"ok": True})

    # Handle admin menu options (simple text routing)
    if "text" in msg and SESS[chat_id].get("state") == "admin_menu" and is_admin(chat_id):
        choice = msg["text"].strip()
        if choice == "Send Message to Target":
            send_message(chat_id, "Please send the message you want to forward to the target (single message).")
            SESS[chat_id]["state"] = "admin_send_message"
            return jsonify({"ok": True})
        if choice == "Show Target":
            cur = get_current_target()
            send_message(chat_id, f"Current target: {cur or 'None'}")
            SESS[chat_id]["state"] = "admin_menu"
            return jsonify({"ok": True})
        if choice == "Set Target":
            send_message(chat_id, "Send the chat_id to set as runtime target (digits only).")
            SESS[chat_id]["state"] = "admin_set_target"
            return jsonify({"ok": True})
        if choice == "Set Template Path":
            send_message(chat_id, "Send the absolute path (inside container) to set as default template for admin (e.g. ./Sample Lesson Plan.docx).")
            SESS[chat_id]["state"] = "admin_set_template"
            return jsonify({"ok": True})
        if choice == "Exit Admin":
            send_message(chat_id, "Exiting admin menu.")
            SESS[chat_id]["state"] = "idle"
            return jsonify({"ok": True})
        # unknown choice -> back to menu
        send_message(chat_id, "Unknown option. /admin to open menu.")
        SESS[chat_id]["state"] = "idle"
        return jsonify({"ok": True})

    # Admin menu follow-ups
    if "text" in msg and is_admin(chat_id):
        state = SESS[chat_id].get("state")
        if state == "admin_send_message":
            message_to_send = msg["text"].strip()
            target = get_current_target()
            if not target:
                send_message(chat_id, "No target set. Use Set Target option or /settarget <chat_id>.")
                SESS[chat_id]["state"] = "admin_menu"
                return jsonify({"ok": True})
            send_message(target, message_to_send)
            send_message(chat_id, f"Message sent to {target}")
            SESS[chat_id]["state"] = "admin_menu"
            return jsonify({"ok": True})
        if state == "admin_set_target":
            cand = msg["text"].strip()
            if cand.isdigit():
                RUNTIME_TARGET["target"] = cand
                send_message(chat_id, f"Runtime target set to {cand}")
            else:
                send_message(chat_id, "Invalid chat_id. It must be digits only.")
            SESS[chat_id]["state"] = "admin_menu"
            return jsonify({"ok": True})
        if state == "admin_set_template":
            cand = msg["text"].strip()
            if os.path.exists(cand):
                SESS[chat_id]["template_path"] = cand
                send_message(chat_id, f"Admin template path set to: {cand}")
            else:
                send_message(chat_id, f"Path does not exist in container: {cand}")
            SESS[chat_id]["state"] = "admin_menu"
            return jsonify({"ok": True})

    # ========== Normal flows (non-admin) ==========
    # initialize per-chat if missing
    SESS.setdefault(chat_id, {"state": "idle", "tmp": {}, "template_path": None})
    # entry command /hi_rise (lowercase)
    if "text" in msg and msg["text"].strip().lower() == "/hi_rise":
        kb = {"keyboard":[["Upload PDF"],["Paste Text"],["Ask Bot to Find Lesson"]],"one_time_keyboard":True,"resize_keyboard":True}
        send_message(chat_id, "Hi! For which lesson shall we create a lesson plan today? Choose how you'd like to provide the lesson:", reply_markup=kb)
        SESS[chat_id]["state"] = "idle"
        SESS[chat_id]["tmp"] = {}
        return jsonify({"ok": True})

    # If user clicked one of the options (plain text)
    if "text" in msg and SESS[chat_id]["state"] == "idle":
        txt = msg["text"].strip()
        if txt == "Upload PDF":
            SESS[chat_id]["state"] = "await_pdf"
            send_message(chat_id, "Please upload the lesson PDF as a document now (send as Telegram document).")
            return jsonify({"ok": True})
        if txt == "Paste Text":
            SESS[chat_id]["state"] = "await_text"
            send_message(chat_id, "Please paste the chapter text now.")
            return jsonify({"ok": True})
        if txt == "Ask Bot to Find Lesson":
            SESS[chat_id]["state"] = "await_grade"
            send_message(chat_id, "Okay — which Grade? (e.g., Grade 6)")
            return jsonify({"ok": True})
        if len(txt) > 120:
            send_message(chat_id, "I detected pasted text — shall I create a lesson plan from this? Reply 'Yes' to proceed.")
            SESS[chat_id]["state"] = "confirm_from_text"
            SESS[chat_id]["tmp"]["text_candidate"] = txt
            return jsonify({"ok": True})

    # Handle document uploads
    if "document" in msg:
        doc = msg["document"]
        fname = doc.get("file_name", "file")
        file_id = doc["file_id"]
        tmpdir = tempfile.mkdtemp()
        local_path = os.path.join(tmpdir, fname)
        try:
            download_file(file_id, local_path)
        except Exception as e:
            send_message(chat_id, f"Failed to download file: {e}")
            SESS[chat_id]["state"] = "idle"
            return jsonify({"ok": True})

        # If .docx: template
        if fname.lower().endswith(".docx"):
            SESS[chat_id]["template_path"] = local_path
            send_message(chat_id, "Template uploaded and saved for your session. Now upload PDF, paste text, or use /hi_rise to start again.")
            SESS[chat_id]["state"] = "idle"
            return jsonify({"ok": True})

        # If PDF and expecting PDF
        if fname.lower().endswith(".pdf") and SESS[chat_id]["state"] == "await_pdf":
            try:
                text = extract_text_from_pdf(local_path)
            except Exception as e:
                send_message(chat_id, f"PDF extraction failed: {e}")
                SESS[chat_id]["state"] = "idle"
                return jsonify({"ok": True})
            send_message(chat_id, "PDF received. Generating lesson plan...")
            title = "Extracted Lesson"
            summary = summarize_text(text, sentences_count=6)
            objectives = extract_objectives_from_text(text)
            activities = generate_activities(text)
            assessment = generate_assessment_questions(text)
            references = "Source: uploaded PDF"
            fill_template_and_send(chat_id, title, summary, objectives, activities, assessment, references)
            SESS[chat_id]["state"] = "idle"
            return jsonify({"ok": True})

        send_message(chat_id, "Document received. If this is a PDF for lesson content, please choose 'Upload PDF' first. If this is a template (.docx), it has been saved.")
        SESS[chat_id]["state"] = "idle"
        return jsonify({"ok": True})

    # Handle plain text during flows
    if "text" in msg:
        txt = msg["text"].strip()
        state = SESS[chat_id]["state"]

        if state == "confirm_from_text":
            if txt.lower() in ("yes","y"):
                candidate = SESS[chat_id]["tmp"].get("text_candidate", "")
                SESS[chat_id]["state"] = "idle"
                send_message(chat_id, "Generating lesson plan from pasted text...")
                summary = summarize_text(candidate, sentences_count=6)
                objectives = extract_objectives_from_text(candidate)
                activities = generate_activities(candidate)
                assessment = generate_assessment_questions(candidate)
                fill_template_and_send(chat_id, "Pasted Lesson", summary, objectives, activities, assessment, "Source: pasted by user")
                return jsonify({"ok": True})
            else:
                SESS[chat_id]["state"] = "idle"
                send_message(chat_id, "Cancelled. Send /hi_rise to begin again.")
                return jsonify({"ok": True})

        if state == "await_text":
            text = txt
            SESS[chat_id]["state"] = "idle"
            send_message(chat_id, "Generating lesson plan from pasted text...")
            summary = summarize_text(text, sentences_count=6)
            objectives = extract_objectives_from_text(text)
            activities = generate_activities(text)
            assessment = generate_assessment_questions(text)
            fill_template_and_send(chat_id, "Pasted Lesson", summary, objectives, activities, assessment, "Source: pasted by user")
            return jsonify({"ok": True}")

        if state == "await_grade":
            SESS[chat_id]["tmp"] = {"grade": txt}
            SESS[chat_id]["state"] = "await_subject"
            send_message(chat_id, "Which Subject? (e.g., Mathematics, Science, English)")
            return jsonify({"ok": True})
        if state == "await_subject":
            SESS[chat_id]["tmp"]["subject"] = txt
            SESS[chat_id]["state"] = "await_chapter"
            send_message(chat_id, "Which Chapter name or number should I search for?")
            return jsonify({"ok": True})
        if state == "await_chapter":
            SESS[chat_id]["tmp"]["chapter"] = txt
            grade = SESS[chat_id]["tmp"].get("grade")
            subject = SESS[chat_id]["tmp"].get("subject")
            chapter = SESS[chat_id]["tmp"].get("chapter")
            query = f"{grade} {subject} {chapter} summary lesson"
            send_message(chat_id, f"Searching web for: {query}")
            SESS[chat_id]["state"] = "idle"
            try:
                hits = ddg(query, max_results=5) or []
            except Exception as e:
                print("ddg error:", e)
                hits = []
            combined_texts = []
            references = []
            for h in hits[:3]:
                url = h.get("href") or h.get("url")
                title = h.get("title") or url
                snippet = h.get("body") or h.get("snippet") or ""
                txt = extract_text_from_url(url) if url else snippet
                if txt:
                    combined_texts.append(txt)
                references.append(f"{title} — {url}")
            big_text = "\n\n".join(combined_texts) or " ".join([h.get("body","") or h.get("snippet","") for h in hits]) or chapter
            summary = summarize_text(big_text, sentences_count=6)
            objectives = extract_objectives_from_text(big_text)
            activities = generate_activities(big_text)
            assessment = generate_assessment_questions(big_text)
            ref_text = "\n".join(references) if references else "No web refs found"
            fill_template_and_send(chat_id, f"{chapter} ({subject})", summary, objectives, activities, assessment, ref_text)
            return jsonify({"ok": True})

        # fallback
        send_message(chat_id, "Send /hi_rise to start the lesson-plan flow, or upload a .docx template.")
        return jsonify({"ok": True})

    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
