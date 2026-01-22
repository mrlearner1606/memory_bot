import os
import ast
import json
import time
import requests
import markdown
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash, jsonify
from markupsafe import Markup  # Changed from flask import to markupsafe import!
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


load_dotenv()
currdate = datetime.now().strftime("%Y-%m-%d")


# --- Config from environment ---
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID")


# Pollinations.AI config (primary AI)
POLLINATION_API_KEY = os.environ.get("POLLINATION")
POLLINATION_MODEL = "openai-fast"


# OpenRouter keys (backup #1)
OPENROUTER_KEYS = [os.environ[k] for k in os.environ if k.startswith("OPENROUTER_API_KEY_")]
OPENROUTER_KEYS = [k for k in OPENROUTER_KEYS if k]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# Gemini keys (backup #2)
GEMINI_KEYS = [os.environ[k] for k in os.environ if k.startswith("GEMINI_API_KEY_")]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")


UI_PASSWORD = os.environ.get("UI_PASSWORD")
FLASK_SECRET = os.environ.get("FLASK_SECRET")


if not AIRTABLE_TOKEN or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_ID:
    raise RuntimeError("Set AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID in env")


# At least one AI service should be available
if not POLLINATION_API_KEY and not OPENROUTER_KEYS and not GEMINI_KEYS:
    raise RuntimeError("Provide at least POLLINATION or OPENROUTER_API_KEY_x or GEMINI_API_KEY_x")


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


# global indexes
openrouter_index = 0
gemini_index = 0

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


def call_pollinations_llm(messages, timeout=180):
    """Primary AI service with long timeout and token query param support"""
    try:
        # Convert messages to a single prompt for pollinations
        if len(messages) == 1:
            prompt = messages[0]["content"]
        else:
            prompt_parts = []
            for msg in messages:
                if msg.get("role") == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg.get("role") == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                else:
                    prompt_parts.append(msg.get("content", ""))
            prompt = "\n".join(prompt_parts)

        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)

        # Prefer token as query param (some gateways expect this)
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
                raise RuntimeError("Empty response from Pollinations")
        else:
            raise RuntimeError(f"Pollinations status {resp.status_code}: {resp.text}")

    except Exception as e:
        raise RuntimeError(f"Pollinations failed: {e}")


def get_next_openrouter_key():
    global openrouter_index
    if not OPENROUTER_KEYS:
        raise RuntimeError("No OpenRouter keys available")
    key = OPENROUTER_KEYS[openrouter_index % len(OPENROUTER_KEYS)]
    openrouter_index += 1
    return key


def get_next_gemini_key():
    global gemini_index
    if not GEMINI_KEYS:
        raise RuntimeError("No Gemini keys available")
    key = GEMINI_KEYS[gemini_index % len(GEMINI_KEYS)]
    gemini_index += 1
    return key


def call_openrouter_llm(messages, timeout=120):
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
                if data.get("choices") and len(data["choices"]) > 0:
                    msg = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text", "")
                    if msg:
                        return msg.strip()
                last_err = RuntimeError("Response missing message content.")
            else:
                last_err = RuntimeError(f"Status {resp.status_code}: {resp.text}")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"All OpenRouter keys failed. Last error: {last_err}")


def call_gemini_llm(messages, timeout=120):
    url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last_err = None

    # Convert messages to Gemini format properly
    gemini_contents = []
    for msg in messages:
        if msg.get("content"):
            role = "user" if msg.get("role") == "user" else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

    if not gemini_contents:
        text_input = "\n".join([m["content"] for m in messages if m.get("content")])
        gemini_contents = [{"parts": [{"text": text_input}]}]

    payload = {
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": 0.4,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }

    for _ in range(len(GEMINI_KEYS)):
        api_key = get_next_gemini_key()
        url = f"{url_template.format(model=GEMINI_MODEL)}?key={api_key}"
        headers = {"Content-Type": "application/json"}

        try:
            resp = session_req.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    msg = data["candidates"][0]["content"]["parts"][0]["text"]
                    if msg:
                        return msg.strip()
                except (KeyError, IndexError) as e:
                    last_err = RuntimeError(f"Gemini response parsing failed: {e}")
            else:
                last_err = RuntimeError(f"Gemini status {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            last_err = RuntimeError(f"Gemini request failed: {e}")
        except Exception as e:
            last_err = RuntimeError(f"Gemini unexpected error: {e}")

    raise RuntimeError(f"All Gemini keys failed. Last error: {last_err}")


def call_llm(messages):
    """
    Try services in order:
    1. Pollinations.AI
    2. Gemini
    3. OpenRouter
    """
    errors = []

    # Try Pollinations first
    if POLLINATION_API_KEY or True:
        try:
            return call_pollinations_llm(messages)
        except Exception as e:
            errors.append(f"Pollinations: {e}")

    # Try Gemini second
    if GEMINI_KEYS:
        try:
            return call_gemini_llm(messages)
        except Exception as e:
            errors.append(f"Gemini: {e}")

    # Try OpenRouter last
    if OPENROUTER_KEYS:
        try:
            return call_openrouter_llm(messages)
        except Exception as e:
            errors.append(f"OpenRouter: {e}")

    raise RuntimeError(f"All AI services failed! Errors: {'; '.join(errors)}")


def classify_intent(query: str) -> str:
    system_prompt = (
        "You are a classifier. If the user is adding/storing/saving new information, reply exactly: INSERT. "
        "If the user is asking/retrieving/searching for info, reply exactly: QUERY. Reply only with INSERT or QUERY. "
        "If the input is just one word then it is only QUERY"
    )
    return call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]).strip().upper()


def extract_insert_fields(query: str) -> dict:
    system_prompt = (
        "You are an extractor. Given the user's statement, output valid JSON (use double quotes) with keys:\n"
        f"- Knowledge make the user input as first person perspective and exclude the word today from the user input\n- Reference put words that relate to users input, that makes it easier to retrieve\n- Date (ISO YYYY-MM-DD or {currdate} if today is mentioned in user input)\n\n"
        "Return only JSON."
    )
    out = call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": query}])
    try:
        parsed = json.loads(out)
        for k in ("Knowledge", "Reference", "Date"):
            parsed.setdefault(k, "")
        return parsed
    except Exception:
        return {"Knowledge": query, "Reference": "", "Date": currdate}


def insert_airtable(fields: dict) -> dict:
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}", "Content-Type": "application/json"}
    payload = {"records": [{"fields": fields}]}
    resp = session_req.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_reference_keywords(query: str) -> list:
    out = call_llm(
        [{"role": "system", "content": "Extract key reference words from the query. Reply comma-separated. If input is one word, just return that word."},
         {"role": "user", "content": query}]
    )
    return [w.strip() for w in out.split(",") if w.strip()]


def search_airtable_by_reference(keywords: list) -> list:
    results, seen_ids = [], set()
    base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}

    for kw in keywords:
        formula = f"FIND('{kw.lower()}', LOWER({{Reference}}))"
        resp = session_req.get(base_url, headers=headers, params={"filterByFormula": formula}, timeout=60)
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
        "- Use only facts from records. If missing, say you don't know.\n"
        "- Format your response using Markdown syntax for better readability.\n"
        "- Use headers (##), bullet points (-), **bold text**, *italic text* where appropriate.\n"
        "- Keep answer concise but well-formatted."
    )
    return call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}\n\nRecords:\n{context}"}
    ])


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


# Static, non-reloading UI using fetch with retries and cookie credentials
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
            // Try to parse JSON; if it fails, throw to trigger retry/catch
            const ct = resp.headers.get("content-type") || "";
            if (!ct.includes("application/json")) {
                const text = await resp.text();
                // If non-JSON but OK, return as reply text
                if (resp.ok) return { ok: true, json: { reply: text } };
                throw new Error("Non-JSON response: " + text);
            }
            const json = await resp.json();
            if (!resp.ok) {
                // Retry on transient 5xx
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
            // Exponential backoff: 1s, 2s
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

    // Periodically nudge the user that it's still working for long responses
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
        addMessage("bot", "❌ Error contacting server. Still waiting can help if the model is slow. Please try again.");
    } finally {
        btn.disabled = false;
    }
}

// Auto-resize textarea
const textarea = document.getElementById('message');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// Send on Enter (no Shift)
textarea.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Button click
document.getElementById("sendBtn").addEventListener("click", sendMessage);
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    if not session.get("authed"):
        return render_template_string(LOGIN_HTML)
    # Serve static, single-page UI (no server-side result rendering)
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


# JSON in/out; long-running friendly
@app.route("/ask", methods=["POST"])
def ask():
    if not session.get("authed"):
        return jsonify({"reply": "🔒 Not authorized. Please log in."}), 401

    # Accept JSON body like the sample; fallback to form field if needed
    query = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
    else:
        query = (request.form.get("query") or "").strip()

    if not query:
        return jsonify({"reply": "⚠️ Query required"}), 400

    try:
        intent = classify_intent(query)
        if intent == "INSERT":
            fields = extract_insert_fields(query)
            insert_airtable(fields)
            result = "✅ **Memory saved successfully!**\n\nYour information has been stored."
        else:
            keywords = extract_reference_keywords(query)
            records = search_airtable_by_reference(keywords)
            result = llm_answer_using_records(query, records)

        return jsonify({"reply": result})
    except Exception as e:
        print(f"Error in ask route: {e}")
        error_result = "❌ **Error occurred:**\n\n``````"
        return jsonify({"reply": error_result}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # threaded=True to keep UI responsive; use_reloader=False to avoid first-request reload hiccup
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True, use_reloader=False)
