# app.py
import os, tempfile, json, time
from flask import Flask, request, jsonify
import requests
from docx import Document
from duckduckgo_search import ddg
from readability import Document as ReadabilityDoc
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# CONFIG
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
DEFAULT_TEMPLATE_PATH = os.environ.get("DEFAULT_TEMPLATE_PATH", "./Sample Lesson Plan.docx")
PORT = int(os.environ.get("PORT", 5000))

if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN env var")

BASE_TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
app = Flask(__name__)

# In-memory sessions (ephemeral)
SESS = {}  # chat_id -> state dict

def telegram_api(method, params=None, files=None, json_payload=None):
    url = f"{BASE_TELEGRAM_URL}/{method}"
    if files:
        return requests.post(url, params=params, files=files, timeout=30)
    if json_payload:
        return requests.post(url, json=json_payload, timeout=30)
    return requests.post(url, data=params, timeout=30)

def send_message(chat_id, text, reply_markup=None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    telegram_api("sendMessage", params=payload)

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

def extract_text_from_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)

def extract_text_from_url(url, max_chars=20000):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.
