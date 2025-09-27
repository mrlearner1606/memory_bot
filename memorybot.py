import os
import ast
import json
import time
import requests
import markdown
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from markupsafe import Markup  # Changed from flask import to markupsafe import!
from dotenv import load_dotenv

load_dotenv()
currdate = datetime.now().strftime("%Y-%m-%d")

# --- Config from environment ---
AIRTABLE_TOKEN = os.environ.get("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID")

# Pollinations.AI config (primary AI - like Rajni, the main hero!)
POLLINATION_API_KEY = os.environ.get("POLLINATION")  # May not be needed, but keeping as requested
POLLINATION_MODEL = "gemini-search"  # Using the reasoning model like Enthiran's brain!

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

# At least one AI service should be available (like having at least one superhero in the team!)
if not POLLINATION_API_KEY and not OPENROUTER_KEYS and not GEMINI_KEYS:
    raise RuntimeError("Provide at least POLLINATION or OPENROUTER_API_KEY_x or GEMINI_API_KEY_x")

app = Flask(__name__)
app.secret_key = FLASK_SECRET

# Initialize markdown with extensions - like giving Batman more gadgets!
md = markdown.Markdown(extensions=[
    'markdown.extensions.extra',      # Tables, footnotes, etc.
    'markdown.extensions.codehilite', # Code syntax highlighting
    'markdown.extensions.toc',        # Table of contents
    'markdown.extensions.nl2br',      # New line to break
    'markdown.extensions.fenced_code' # Fenced code blocks
], extension_configs={
    'codehilite': {
        'css_class': 'highlight',
        'use_pygments': False
    }
})

# Custom markdown filter - like Sivaji's special skills!
def markdown_filter(text):
    return Markup(md.convert(text))

app.jinja_env.filters['markdown'] = markdown_filter

# global indexes
openrouter_index = 0
gemini_index = 0
session_req = requests.Session()

def call_pollinations_llm(messages, timeout=30):
    """Primary AI service - like Sivaji leading the charge!"""
    try:
        # Convert messages to a single prompt for pollinations
        if len(messages) == 1:
            prompt = messages[0]["content"]
        else:
            # Combine system and user messages
            prompt_parts = []
            for msg in messages:
                if msg.get("role") == "system":
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg.get("role") == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                else:
                    prompt_parts.append(msg.get("content", ""))
            prompt = "\n".join(prompt_parts)
        
        # URL encode the prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Pollinations.AI text endpoint with reasoning model
        url = f"https://text.pollinations.ai/{encoded_prompt}?model={POLLINATION_MODEL}"
        
        # Add API key in headers if provided (though pollinations might not need it)
        headers = {}
        if POLLINATION_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATION_API_KEY}"
        
        resp = session_req.get(url, headers=headers, timeout=timeout)
        print(f"Pollinations Response Status: {resp.status_code}")  # Debug line
        print(f"Pollinations Response: {resp.text[:500]}")  # Debug line
        
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

def call_openrouter_llm(messages, timeout=20):
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

def call_gemini_llm(messages, timeout=30):
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
    
    # If no proper conversation, combine all content
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
            print(f"Gemini Response Status: {resp.status_code}")  # Debug line
            print(f"Gemini Response: {resp.text[:500]}")  # Debug line
            
            if resp.status_code == 200:
                data = resp.json()
                try:
                    msg = data["candidates"][0]["content"]["parts"][0]["text"]
                    if msg:
                        return msg.strip()
                except (KeyError, IndexError) as e:
                    print(f"Gemini parsing error: {e}")
                    print(f"Response data: {data}")
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
    Try services in order like Batman's contingency plans:
    1. Pollinations.AI (primary - like Rajni taking charge!)
    2. Gemini (backup #1 - like Robin stepping in)
    3. OpenRouter (backup #2 - like Alfred as last resort)
    """
    errors = []
    
    # Try Pollinations first (main hero!)
    if POLLINATION_API_KEY or True:  # Pollinations is free, so try even without key
        try:
            print("Trying Pollinations.AI first...")
            return call_pollinations_llm(messages)
        except Exception as e:
            print(f"Pollinations failed: {e}")
            errors.append(f"Pollinations: {e}")
    
    # Try Gemini second (backup hero #1)
    if GEMINI_KEYS:
        try:
            print("Fallback to Gemini...")
            return call_gemini_llm(messages)
        except Exception as e:
            print(f"Gemini failed: {e}")
            errors.append(f"Gemini: {e}")
    
    # Try OpenRouter last (backup hero #2)
    if OPENROUTER_KEYS:
        try:
            print("Final fallback to OpenRouter...")
            return call_openrouter_llm(messages)
        except Exception as e:
            print(f"OpenRouter failed: {e}")
            errors.append(f"OpenRouter: {e}")
    
    # All failed - like when all superheroes are down!
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
    resp = session_req.post(url, headers=headers, json=payload)
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

MAIN_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Bot</title>
    <style>
        body { 
            background: linear-gradient(135deg, #121212, #1e1e1e); 
            color: #eee; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            padding: 2em; 
            margin: 0;
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2em;
            padding: 1em;
            background: rgba(30, 30, 30, 0.5);
            border-radius: 15px;
        }
        
        .header h2 {
            color: #1e88e5;
            margin: 0;
        }
        
        .user-info {
            margin-top: 0.5em;
            color: #aaa;
        }
        
        .user-info a {
            color: #90caf9;
            text-decoration: none;
        }
        
        .user-info a:hover {
            text-decoration: underline;
        }
        
        .query-section {
            background: rgba(30, 30, 30, 0.8);
            padding: 2em;
            border-radius: 15px;
            margin-bottom: 2em;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        textarea { 
            width: 100%; 
            padding: 1em; 
            border-radius: 10px; 
            border: 2px solid #333; 
            resize: vertical; 
            background: #2a2a2a; 
            color: #fff; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            border-color: #1e88e5;
            outline: none;
        }
        
        button { 
            margin-top: 1em; 
            padding: 1em 2em; 
            border: none; 
            border-radius: 10px; 
            background: linear-gradient(135deg, #1e88e5, #1976d2); 
            color: white; 
            cursor: pointer; 
            font-weight: bold;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        button:hover {
            background: linear-gradient(135deg, #1976d2, #1565c0);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
        }
        
        .result-section {
            background: rgba(30, 30, 30, 0.8);
            padding: 2em;
            border-radius: 15px;
            margin-top: 2em;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .result-section h3 {
            color: #1e88e5;
            margin-top: 0;
            border-bottom: 2px solid #333;
            padding-bottom: 0.5em;
        }
        
        /* Markdown Styling - like Tony Stark's HUD! */
        .markdown-content {
            background: #1a1a1a;
            padding: 1.5em;
            border-radius: 10px;
            border: 1px solid #333;
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .markdown-content h1, .markdown-content h2, .markdown-content h3, 
        .markdown-content h4, .markdown-content h5, .markdown-content h6 {
            color: #1e88e5;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        
        .markdown-content h1 { font-size: 2em; border-bottom: 2px solid #1e88e5; padding-bottom: 0.3em; }
        .markdown-content h2 { font-size: 1.5em; border-bottom: 1px solid #666; padding-bottom: 0.2em; }
        .markdown-content h3 { font-size: 1.3em; }
        
        .markdown-content p {
            margin: 1em 0;
            line-height: 1.7;
        }
        
        .markdown-content ul, .markdown-content ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        
        .markdown-content li {
            margin: 0.5em 0;
        }
        
        .markdown-content strong {
            color: #ffeb3b;
            font-weight: bold;
        }
        
        .markdown-content em {
            color: #81c784;
            font-style: italic;
        }
        
        .markdown-content code {
            background: #2a2a2a;
            color: #f8f8f2;
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }
        
        .markdown-content pre {
            background: #2a2a2a;
            color: #f8f8f2;
            padding: 1em;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #444;
        }
        
        .markdown-content pre code {
            background: none;
            padding: 0;
        }
        
        .markdown-content blockquote {
            border-left: 4px solid #1e88e5;
            margin: 1em 0;
            padding-left: 1em;
            color: #ccc;
            font-style: italic;
        }
        
        .markdown-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            background: #2a2a2a;
        }
        
        .markdown-content table th, .markdown-content table td {
            border: 1px solid #444;
            padding: 0.8em;
            text-align: left;
        }
        
        .markdown-content table th {
            background: #1e88e5;
            color: white;
            font-weight: bold;
        }
        
        .markdown-content table tr:nth-child(even) {
            background: #333;
        }
        
        .markdown-content a {
            color: #90caf9;
            text-decoration: none;
        }
        
        .markdown-content a:hover {
            text-decoration: underline;
        }
        
        .markdown-content hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, transparent, #1e88e5, transparent);
            margin: 2em 0;
        }
        
        /* Success message styling */
        .success-message {
            color: #4caf50;
            background: rgba(76, 175, 80, 0.1);
            padding: 1em;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            body {
                padding: 1em;
            }
            
            .query-section, .result-section {
                padding: 1.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🧠 Memory Bot</h2>
            <div class="user-info">
                Signed in as <strong>Krishna</strong> | 
                <a href="{{ url_for('logout') }}">🚪 Logout</a>
            </div>
        </div>
        
        <div class="query-section">
            <form method="post" action="{{ url_for('ask') }}" id="queryForm">
                <label><strong>Enter your query or statement:</strong></label><br><br>
                <textarea name="query" rows="5" placeholder="e.g., My graduation day was on July 10, 2015 OR When did I graduate?" required></textarea><br>
                <button type="submit">🚀 Submit</button>
            </form>
        </div>
        
        {% if result %}
        <div class="result-section">
            <h3>📝 Answer</h3>
            <div class="markdown-content">
                {{ result|markdown }}
            </div>
        </div>
        {% endif %}
    </div>
    
    <script>
        document.getElementById("queryForm").addEventListener("keydown", function(e) {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            this.submit();
          }
        });
        
        // Auto-resize textarea like Jarvis expanding interface
        const textarea = document.querySelector('textarea');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    </script>
</body>
</html>
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
            result = "✅ **Memory saved successfully!**\n\nYour information has been stored like Jarvis filing away Tony Stark's data."
        else:
            keywords = extract_reference_keywords(query)
            records = search_airtable_by_reference(keywords)
            result = llm_answer_using_records(query, records)
        return render_template_string(MAIN_HTML, result=result)
    except Exception as e:
        print(f"Error in ask route: {e}")  # Debug line
        error_result = f"❌ **Error occurred:**\n\n``````\n\nLike when Enthiran faces a system malfunction!"
        return render_template_string(MAIN_HTML, result=error_result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
