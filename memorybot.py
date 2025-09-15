# app.py
import os
import ast
import json
import time
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
OPENROUTER_KEYS = []
i = 1
while True:
    try:
        key = os.environ[f"OPENROUTER_API_KEY_{i}"]
        OPENROUTER_KEYS.append(key)
        i += 1
    except:
        break
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


# global index for rotation
key_index = 0

def get_next_openrouter_key():
    """Return the next key in round-robin order."""
    global key_index
    key = OPENROUTER_KEYS[key_index % len(OPENROUTER_KEYS)]
    key_index += 1
    return key

def call_openrouter_llm(messages, per_key_retries=3, timeout=30, backoff_factor=1.0):
    """
    Round-robin key rotation across requests.
    For each request: start with the next key in rotation.
    If that key fails, fall back to the remaining keys in order.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    last_err = None

    # start from the next key in rotation
    start_index = (key_index - 1) % len(OPENROUTER_KEYS)
    ordered_keys = OPENROUTER_KEYS[start_index:] + OPENROUTER_KEYS[:start_index]

    for idx, api_key in enumerate(ordered_keys, start=1):
        attempt = 0
        while attempt < per_key_retries:
            attempt += 1
            wait = backoff_factor * (2 ** (attempt - 1))
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.4}

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("choices"):
                        msg = data["choices"][0].get("message", {}).get("content") \
                              or data["choices"][0].get("text", "")
                        if msg:
                            return msg.strip()
                        last_err = RuntimeError(f"Key #{idx} response missing message content.")
                    else:
                        last_err = RuntimeError(f"Key #{idx} response missing 'choices'.")
                else:
                    last_err = RuntimeError(f"Key #{idx} returned status {resp.status_code}: {resp.text}")

                # retry on transient issues
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < per_key_retries:
                    time.sleep(wait)
                    continue
                else:
                    break

            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt < per_key_retries:
                    time.sleep(wait)
                    continue
                else:
                    break

        # move to next key if this one exhausted
        continue

    raise RuntimeError(f"All OpenRouter keys failed. Last error: {last_err}")


def classify_intent(query: str) -> str:
    system_prompt = (
        "You are a classifier. If the user is adding/storing/saving new information, reply exactly: INSERT. "
        "If the user is asking/retrieving/searching for info, reply exactly: QUERY. Reply only with INSERT or QUERY."
        "If the input is just one word then it is only QUERY"
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
        [{"role": "system", "content": "Extract key reference words from the query. Reply comma-separated. if the user input is only one word then oonly use that"},
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
        # show full traceback-ish message for debugging in UI
        return render_template_string(MAIN_HTML, result=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
