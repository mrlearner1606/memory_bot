import os
import json
import re
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from dotenv import load_dotenv
import requests

load_dotenv()

currdate = datetime.now().strftime("%Y-%m-%d")
# --- Config from environment ---
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID")

# OpenRouter keys
OPENROUTER_KEYS = [os.environ[k] for k in os.environ if k.startswith("OPENROUTER_API_KEY_")]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

# Gemini keys
GEMINI_KEYS = [os.environ[k] for k in os.environ if k.startswith("GEMINI_API_KEY_")]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

UI_PASSWORD = os.environ.get("UI_PASSWORD")
FLASK_SECRET = os.environ.get("FLASK_SECRET")

if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Set AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID in env")
if not OPENROUTER_KEYS and not GEMINI_KEYS:
    raise RuntimeError("Provide at least one OPENROUTER_API_KEY_x or GEMINI_API_KEY_x")

app = Flask(__name__)
app.secret_key = FLASK_SECRET

# global indexes
openrouter_index = 0
gemini_index = 0
session_req = requests.Session()

def get_next_openrouter_key():
    global openrouter_index
    key = OPENROUTER_KEYS[openrouter_index % len(OPENROUTER_KEYS)]
    openrouter_index += 1
    return key

def get_next_gemini_key():
    global gemini_index
    key = GEMINI_KEYS[gemini_index % len(GEMINI_KEYS)]
    gemini_index += 1
    return key

# Lightning fast LLM call with fallback
def call_openrouter_llm(messages, timeout=15):
    url = "https://openrouter.ai/api/v1/chat/completions"
    last_err = None
    for _ in range(len(OPENROUTER_KEYS)):
        api_key = get_next_openrouter_key()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.4}
        try:
            resp = session_req.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text", "")
                if msg:
                    return msg.strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All OpenRouter keys failed. Last error: {last_err}")

def call_gemini_llm(messages, timeout=15):
    url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    text_input = "\n".join([m["content"] for m in messages if m.get("content")])
    payload = {"contents": [{"parts": [{"text": text_input}]}]}
    last_err = None
    for _ in range(len(GEMINI_KEYS)):
        api_key = get_next_gemini_key()
        url = url_template.format(model=GEMINI_MODEL, api_key=api_key)
        try:
            resp = session_req.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["candidates"][0]["content"]["parts"][0]["text"]
                if msg:
                    return msg.strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All Gemini keys failed. Last error: {last_err}")

def call_llm(messages):
    try:
        if GEMINI_KEYS:
            return call_gemini_llm(messages)
        raise RuntimeError("No Gemini keys configured.")
    except Exception:
        if OPENROUTER_KEYS:
            return call_openrouter_llm(messages)
        raise

# ----------------- Fast Classification & Extraction -----------------
def classify_intent(query: str) -> str:
    query_lower = query.lower().strip()
    # one-word queries are QUERY
    if len(query_lower.split()) == 1:
        return "QUERY"
    # common phrases indicating insert
    insert_triggers = ["i am", "i have", "my", "i was", "i learned", "i did"]
    if any(query_lower.startswith(p) for p in insert_triggers):
        return "INSERT"
    return "QUERY"

def extract_insert_fields(query: str) -> dict:
    # Extract simple date if present
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if not date_match:
        # look for common date patterns like July 10, 2015
        date_match2 = re.search(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4})", query)
        if date_match2:
            from dateutil import parser
            try:
                date_val = parser.parse(date_match2.group(1)).date().isoformat()
            except:
                date_val = currdate
        else:
            date_val = currdate
    else:
        date_val = date_match.group(1)
    # Keywords: pick nouns / words >3 chars as reference
    words = re.findall(r'\b\w{4,}\b', query)
    reference = ",".join(words[:5])
    return {"Knowledge": query, "Reference": reference, "Date": date_val}

def extract_reference_keywords(query: str) -> list:
    words = re.findall(r'\b\w{2,}\b', query)
    return words[:5]  # take first 5 words as keywords

def insert_airtable(fields: dict) -> dict:
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}
    payload = {"records": [{"fields": fields}]}
    resp = session_req.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def search_airtable_by_reference(keywords: list) -> list:
    results, seen_ids = [], set()
    base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}
    for kw in keywords:
        formula = f"FIND('{kw.lower()}', LOWER({{Reference}}))"
        resp = session_req.get(base_url, headers=headers, params={"filterByFormula": formula})
        resp.raise_for_status()
        for r in resp.json().get("records", []):
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(r)
    return results

def llm_answer_using_records(query: str, records: list) -> str:
    context = "\n".join([json.dumps(r.get("fields", {}), ensure_ascii=False) for r in records]) or "No records."
    system_prompt = (
        "You are an assistant that answers ONLY using Airtable records provided.\n"
        "- 'I', 'me', 'myself' = Krishna (the user). Address him in second person ('you').\n"
        "- Use only facts from records. If missing, say you don’t know.\n"
        "- Keep answer concise."
    )
    return call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}\n\nRecords:\n{context}"}
    ])

# ----------------- Flask UI -----------------
LOGIN_HTML = """
<!doctype html>
<title>Memory Bot - Login</title>
<style>
body { background:#121212; color:#eee; font-family:Arial; padding:2em; }
input,button { padding:0.5em; border-radius:5px; border:none; }
button { background:#1e88e5; color:white; cursor:pointer; }
</style>
<h2>Memory Bot - Sign in</h2>
{% with messages = get_flashed_messages() %}
  {% if messages %} <ul style="color: red;">{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul> {% endif %}
{% endwith %}
<form method="post" action="{{ url_for('login') }}">
  <input type="password" name="password" placeholder="Password" autofocus required>
  <button type="submit">Sign in</button>
</form>
"""

MAIN_HTML = """
<!doctype html>
<title>Memory Bot</title>
<style>
body { background:#121212; color:#eee; font-family:Arial; padding:2em; }
textarea { width:100%; padding:0.7em; border-radius:8px; border:none; resize:vertical; background:#1e1e1e; color:#fff; }
button { margin-top:1em; padding:0.7em 1.2em; border:none; border-radius:8px; background:#1e88e5; color:white; cursor:pointer; }
pre { background:#1e1e1e; padding:1em; border-radius:8px; white-space:pre-wrap; }
</style>
<h2>Memory Bot</h2>
<p>Signed in as <strong>Krishna</strong>. <a href="{{ url_for('logout') }}" style="color:#90caf9;">Logout</a></p>
<form method="post" action="{{ url_for('ask') }}" id="queryForm">
  <label>Enter query or statement:</label><br>
  <textarea name="query" rows="4" placeholder="e.g. My graduation day was on July 10, 2015 OR When did I graduate?" required></textarea><br>
  <button type="submit">Submit</button>
</form>
{% if result %}
<hr>
<h3>Answer</h3>
<pre>{{ result }}</pre>
{% endif %}
<script>
document.getElementById("queryForm").addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.submit(); }
});
</script>
"""

@app.route("/", methods=["GET"])
def index():
    if not session.get("authed"): return render_template_string(LOGIN_HTML)
    return render_template_string(MAIN_HTML, result=None)

@app.route("/login", methods=["POST"])
def login():
    if request.form.get("password") == UI_PASSWORD:
        session["authed"] = True
        return redirect(url_for("index"))
    flash("Invalid password")
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    if not session.get("authed"): return redirect(url_for("index"))
    query = request.form.get("query", "").strip()
    if not query:
        flash("Query required")
        return redirect(url_for("index"))
    try:
        intent = classify_intent(query)
        if intent == "INSERT":
            fields = extract_insert_fields(query)
            insert_airtable(fields)
            result = "✅ Saved your memory successfully."
        else:
            keywords = extract_reference_keywords(query)
            records = search_airtable_by_reference(keywords)
            result = llm_answer_using_records(query, records)
        return render_template_string(MAIN_HTML, result=result)
    except Exception as e:
        return render_template_string(MAIN_HTML, result=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
