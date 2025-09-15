# app.py
import os
import ast
import json
import requests
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from dotenv import load_dotenv
load_dotenv()

currdate = datetime.now().strftime("%Y-%m-%d")
# --- Config from environment ---
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID")

# OpenRouter keys
OPENROUTER_KEYS = [os.environ.get(f"OPENROUTER_API_KEY_{i}") for i in range(1, 4)]
OPENROUTER_KEYS = [k for k in OPENROUTER_KEYS if k]  # filter out None
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

UI_PASSWORD = os.environ.get("UI_PASSWORD")
FLASK_SECRET = os.environ.get("FLASK_SECRET")

if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Set AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID in env")
if not OPENROUTER_KEYS:
    raise RuntimeError("Provide at least one OPENROUTER_API_KEY_x")

app = Flask(__name__)
app.secret_key = FLASK_SECRET
key_index = 0


def get_next_openrouter_key():
    global key_index
    key = OPENROUTER_KEYS[key_index % len(OPENROUTER_KEYS)]
    key_index += 1
    return key


def call_openrouter_llm(messages, timeout=30):
    api_key = get_next_openrouter_key()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.0}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {}).get("content")
        return msg.strip() if msg else data["choices"][0].get("text", "").strip()
    return json.dumps(data)


def classify_intent(query: str) -> str:
    system_prompt = (
        "You are a classifier. If the user is adding/storing/saving new information, reply exactly: INSERT. "
        "If the user is asking/retrieving/searching for info, reply exactly: QUERY. Reply only with INSERT or QUERY."
    )
    return call_openrouter_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    ).strip().upper()


def extract_insert_fields(query: str) -> dict:
    system_prompt = (
        "You are an extractor. Given the user's statement, output valid JSON (use double quotes) with keys:\n"
        f"- Knowledge\n- Reference\n- Date (ISO YYYY-MM-DD or {currdate})\n\n"
        "Return only JSON. Example:\n"
        '{"Knowledge":"I graduated on July 10, 2015","Reference":"graduation,education,college","Date":"2015-07-10"}'
    )
    out = call_openrouter_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    )
    try:
        parsed = json.loads(out)
        for k in ("Knowledge", "Reference", "Date"):
            parsed.setdefault(k, "")
        return parsed
    except Exception:
        try:
            parsed = ast.literal_eval(out)
            return {k: parsed.get(k, "") for k in ("Knowledge", "Reference", "Date")}
        except Exception:
            return {"Knowledge": query, "Reference": "", "Date": ""}


def insert_airtable(fields: dict) -> dict:
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}
    payload = {"records": [{"fields": fields}]}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def extract_reference_keywords(query: str) -> list:
    out = call_openrouter_llm(
        [{"role": "system", "content": "Extract key reference words from the query. Reply comma-separated."},
         {"role": "user", "content": query}]
    )
    return [w.strip() for w in out.split(",") if w.strip()]


def search_airtable_by_reference(keywords: list) -> list:
    results, seen_ids = [], set()
    base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}
    for kw in keywords:
        safe_kw = kw.replace("'", "\\'")
        formula = f"FIND('{safe_kw.lower()}', LOWER({{Reference}}))"
        resp = requests.get(base_url, headers=headers, params={"filterByFormula": formula})
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
    return call_openrouter_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}\n\nRecords:\n{context}"}
    ])


# ---------------------------
# Flask UI
# ---------------------------

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
  {% if messages %}
    <ul style="color: red;">{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul>
  {% endif %}
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
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    this.submit();
  }
});
</script>
"""

@app.route("/", methods=["GET"])
def index():
    if not session.get("authed"):
        return render_template_string(LOGIN_HTML)
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
    if not session.get("authed"):
        return redirect(url_for("index"))
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
