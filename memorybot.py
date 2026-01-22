import os
import json
import requests
import markdown
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash, jsonify
from markupsafe import Markup
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()
currdate = datetime.now().strftime("%Y-%m-%d")


# --- Config from environment ---
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID")

# Pollinations.AI config (only AI service)
POLLINATION_API_KEY = os.environ.get("POLLINATION")
POLLINATION_MODEL = "openai-fast"

UI_PASSWORD = os.environ.get("UI_PASSWORD")
FLASK_SECRET = os.environ.get("FLASK_SECRET")


if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Set AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID in env")


app = Flask(__name__)
app.secret_key = FLASK_SECRET


# Initialize markdown with extensions
md = markdown.Markdown(extensions=[
    'markdown.extensions.extra',
    'markdown.extensions.codehilite',
    'markdown.extensions.toc',
    'markdown.extensions.nl2br',
    'markdown.extensions.fenced_code'
], extension_configs={
    'codehilite': {
        'css_class': 'highlight',
        'use_pygments': False
    }
})


def markdown_filter(text):
    return Markup(md.convert(text))


app.jinja_env.filters['markdown'] = markdown_filter


# Reusable session with retries and bigger pools
session_req = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
session_req.mount("https://", adapter)
session_req.mount("http://", adapter)


def call_llm(prompt_text, timeout=180):
    """Single AI service using Pollinations.AI"""
    try:
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt_text)

        url = f"https://gen.pollinations.ai/text/{encoded_prompt}?model={POLLINATION_MODEL}"
        if POLLINATION_API_KEY:
            url += f"&key={POLLINATION_API_KEY}"
        
        headers = {}

        resp = session_req.get(url, headers=headers, timeout=timeout)
        
        if resp.status_code == 200:
            result = resp.text.strip()
            if result:
                return result
            else:
                raise RuntimeError("Empty response from Pollinations AI")
        else:
            raise RuntimeError(f"Pollinations AI error {resp.status_code}: {resp.text}")

    except Exception as e:
        raise RuntimeError(f"AI service failed: {e}")


def process_query(query: str) -> dict:
    """
    Single AI call to handle everything: classify intent and extract/answer in one go.
    Returns: {"intent": "INSERT"|"QUERY", "data": {...}}
    """
    prompt = f"""Analyze this user query and respond with ONLY valid JSON (no markdown, no extra text).

Current date: {currdate}

User query: "{query}"

If the user is storing/saving information, respond with:
{{"intent": "INSERT", "knowledge": "first-person version of the input", "reference": "keywords for retrieval", "date": "YYYY-MM-DD or {currdate}"}}

If the user is asking/searching for information, respond with:
{{"intent": "QUERY", "keywords": ["keyword1", "keyword2", "keyword3"]}}

Rules:
- For INSERT: Convert to first person, exclude word "today", extract relevant keywords
- For QUERY: Extract 2-5 relevant search keywords
- Single-word inputs are always QUERY
- Output ONLY the JSON object, nothing else

JSON response:"""

    try:
        result = call_llm(prompt)
        # Clean up potential markdown or extra text
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join([l for l in lines if not l.startswith("```")])
        result = result.strip()
        
        parsed = json.loads(result)
        return parsed
    except Exception as e:
        # Fallback: treat as query if parsing fails
        print(f"Parse error: {e}, treating as QUERY")
        return {"intent": "QUERY", "keywords": [query]}


def insert_airtable(fields: dict) -> dict:
    """Insert record into Airtable"""
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}
    payload = {"records": [{"fields": fields}]}
    resp = session_req.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def search_airtable_by_reference(keywords: list) -> list:
    """Search Airtable records by keywords"""
    results, seen_ids = [], set()
    base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}

    for kw in keywords:
        kw_clean = str(kw).strip()
        if not kw_clean:
            continue
        formula = f"FIND('{kw_clean.lower()}', LOWER({{Reference}}))"
        resp = session_req.get(base_url, headers=headers, params={"filterByFormula": formula}, timeout=60)
        resp.raise_for_status()
        for r in resp.json().get("records", []):
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(r)
    return results


def llm_answer_using_records(query: str, records: list) -> str:
    """Generate answer using retrieved records"""
    context = "\n".join([json.dumps(r.get("fields", {}), ensure_ascii=False) for r in records]) or "No records found."
    
    prompt = f"""You are Krishna's memory assistant. Answer the question using ONLY the provided records.

Rules:
- 'I', 'me', 'myself' in records = Krishna (the user). Address him as 'you'
- Use only facts from the records below
- If information is missing, say you don't know
- Format response in clean Markdown (headers ##, bullets -, **bold**, *italic*)
- Keep it concise but well-formatted

User question: {query}

Available records:
{context}

Your answer (in Markdown):"""

    return call_llm(prompt)


LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Bot - Login</title>
    <style>
        body {
            background: linear-gradient(135deg, #121212, #1e1e1e);
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 2em;
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: rgba(30, 30, 30, 0.8);
            padding: 2em;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            max-width: 400px;
            width: 100%;
        }
        input, button {
            padding: 0.8em;
            border-radius: 8px;
            border: none;
            width: 100%;
            margin: 0.5em 0;
            box-sizing: border-box;
        }
        input {
            background: #2a2a2a;
            color: #fff;
        }
        button {
            background: linear-gradient(135deg, #1e88e5, #1976d2);
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        button:hover {
            background: linear-gradient(135deg, #1976d2, #1565c0);
            transform: translateY(-2px);
        }
        h2 {
            text-align: center;
            margin-bottom: 1.5em;
            color: #1e88e5;
        }
        .error {
            color: #f44336;
            background: rgba(244, 67, 54, 0.1);
            padding: 0.5em;
            border-radius: 5px;
            margin: 1em 0;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>🧠 Memory Bot - Sign in</h2>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="error">
                {% for m in messages %}<div>{{ m }}</div>{% endfor %}
            </div>
          {% endif %}
        {% endwith %}
        <form method="post" action="{{ url_for('login') }}">
          <input type="password" name="password" placeholder="Enter Password" autofocus required>
          <button type="submit">🚀 Sign in</button>
        </form>
    </div>
</body>
</html>
"""


MAIN_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧠 Memory Bot</title>
<style>
    body {
        background: linear-gradient(135deg, #121212, #1e1e1e);
        color: #eee;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0; padding: 0;
        display: flex; justify-content: center; align-items: center;
        min-height: 100vh;
    }
    .chat-container {
        width: 95%; max-width: 1000px;
        background: rgba(30,30,30,0.85);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
        display: flex; flex-direction: column;
        height: 92vh;
    }
    .header {
        padding: 1em; text-align: center;
        border-bottom: 1px solid #333;
    }
    .header h2 { margin: 0; color: #1e88e5; }
    .user-info { color: #aaa; margin-top: 6px; }
    .user-info a { color: #90caf9; text-decoration: none; }
    .user-info a:hover { text-decoration: underline; }
    #chatbox {
        flex: 1; overflow-y: auto; padding: 1em;
    }
    .msg {
        margin: 8px 0; padding: 10px 14px;
        border-radius: 10px; max-width: 75%;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .user { background: #1e88e5; color: white; margin-left: auto; }
    .bot { background: #2a2a2a; border: 1px solid #444; margin-right: auto; }
    .loading {
        background: #2a2a2a;
        border: 1px dashed #555;
        color: #bbb;
        font-style: italic;
    }
    .input-area {
        display: flex; padding: 1em; border-top: 1px solid #333;
        gap: 10px;
    }
    textarea {
        flex: 1;
        padding: 10px; border-radius: 10px;
        border: 2px solid #333; resize: none;
        background: #2a2a2a; color: #fff; font-size: 14px;
        font-family: inherit;
    }
    textarea:focus { border-color: #1e88e5; outline: none; }
    button {
        padding: 10px 18px;
        border: none; border-radius: 10px;
        background: linear-gradient(135deg, #1e88e5, #1976d2);
        color: white; font-weight: bold; cursor: pointer;
        transition: 0.3s; height: 42px;
    }
    button:hover { background: linear-gradient(135deg, #1976d2, #1565c0); transform: translateY(-2px); }
</style>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
<div class="chat-container">
    <div class="header">
        <h2>🧠 Memory Bot</h2>
        <div class="user-info">
            Signed in as <strong>Krishna</strong> |
            <a href="{{ url_for('logout') }}">🚪 Logout</a>
        </div>
    </div>
    <div id="chatbox"></div>
    <div class="input-area">
        <textarea id="message" rows="1" placeholder="Type a memory or ask a question..."></textarea>
        <button id="sendBtn">Send</button>
    </div>
</div>

<script>
function addMessage(role, text, isLoading=false) {
    const chatbox = document.getElementById("chatbox");
    const div = document.createElement("div");
    div.className = "msg " + role + (isLoading ? " loading" : "");
    if (role === "bot") {
        try {
            div.innerHTML = marked.parse(text || "");
        } catch (_) {
            div.textContent = text || "";
        }
    } else {
        div.textContent = text || "";
    }
    chatbox.appendChild(div);
    div.scrollIntoView({behavior: "smooth", block: "start"});
    return div;
}

function sleep(ms){ return new Promise(res => setTimeout(res, ms)); }

async function fetchWithRetry(url, options, retries=2) {
    let attempt = 0;
    let lastError = null;
    while (attempt <= retries) {
        try {
            const resp = await fetch(url, options);
            const ct = resp.headers.get("content-type") || "";
            if (!ct.includes("application/json")) {
                const text = await resp.text();
                if (resp.ok) return { ok: true, json: { reply: text } };
                throw new Error("Non-JSON response: " + text);
            }
            const json = await resp.json();
            if (!resp.ok) {
                if (resp.status >= 500 && resp.status < 600) {
                    throw new Error("Server error " + resp.status);
                }
                return { ok: false, json };
            }
            return { ok: true, json };
        } catch (e) {
            lastError = e;
            attempt++;
            if (attempt > retries) break;
            await sleep(1000 * attempt);
        }
    }
    throw lastError || new Error("Unknown network error");
}

async function sendMessage() {
    const textarea = document.getElementById("message");
    const btn = document.getElementById("sendBtn");
    const message = textarea.value.trim();
    if (!message) return;

    addMessage("user", message);
    textarea.value = "";
    textarea.style.height = "auto";

    btn.disabled = true;
    const loader = addMessage("bot", "Thinking…", true);

    let dotTimer = setInterval(() => {
        if (loader) loader.textContent += ".";
    }, 3000);

    try {
        const res = await fetchWithRetry("{{ url_for('ask') }}", {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"},
            credentials: "same-origin",
            body: JSON.stringify({query: message})
        }, 2);

        loader.remove();
        clearInterval(dotTimer);

        if (res.ok) {
            addMessage("bot", res.json.reply || "✅ Done.");
        } else {
            addMessage("bot", res.json.reply || "⚠️ Request failed.");
        }
    } catch (e) {
        loader.remove();
        clearInterval(dotTimer);
        addMessage("bot", "❌ Error contacting server. Please try again.");
    } finally {
        btn.disabled = false;
    }
}

const textarea = document.getElementById('message');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

textarea.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

document.getElementById("sendBtn").addEventListener("click", sendMessage);
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    if not session.get("authed"):
        return render_template_string(LOGIN_HTML)
    return render_template_string(MAIN_HTML)


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
        return jsonify({"reply": "🔒 Not authorized. Please log in."}), 401

    query = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
    else:
        query = (request.form.get("query") or "").strip()

    if not query:
        return jsonify({"reply": "⚠️ Query required"}), 400

    try:
        # Single AI call handles classification and extraction/keywords
        ai_result = process_query(query)
        
        if ai_result.get("intent") == "INSERT":
            # Store the memory
            fields = {
                "Knowledge": ai_result.get("knowledge", query),
                "Reference": ai_result.get("reference", ""),
                "Date": ai_result.get("date", currdate)
            }
            insert_airtable(fields)
            result = "✅ **Memory saved successfully!**\n\nYour information has been stored."
        else:
            # Retrieve and answer
            keywords = ai_result.get("keywords", [query])
            records = search_airtable_by_reference(keywords)
            result = llm_answer_using_records(query, records)

        return jsonify({"reply": result})
    except Exception as e:
        print(f"Error in ask route: {e}")
        error_result = f"❌ **Error occurred:**\n\n```\n{str(e)}\n```"
        return jsonify({"reply": error_result}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True, use_reloader=False)