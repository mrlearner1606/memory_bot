import os, json, requests, time
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, url_for, session, flash
from dotenv import load_dotenv
load_dotenv()

# ---------- Config & helpers ----------
currdate = datetime.now().strftime("%Y-%m-%d")

AIRTABLE_TOKEN   = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID= os.getenv("AIRTABLE_TABLE_ID")

OPENROUTER_KEYS  = [os.getenv(k) for k in os.environ if k.startswith("OPENROUTER_API_KEY_") and os.getenv(k)]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL","openai/gpt-oss-20b:free")

GEMINI_KEYS      = [os.getenv(k) for k in os.environ if k.startswith("GEMINI_API_KEY_") and os.getenv(k)]
GEMINI_MODEL     = os.getenv("GEMINI_MODEL","gemini-2.0-flash")

UI_PASSWORD      = os.getenv("UI_PASSWORD")
FLASK_SECRET     = os.getenv("FLASK_SECRET")

if not (AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_ID):
    raise RuntimeError("Missing Airtable env vars.")
if not (OPENROUTER_KEYS or GEMINI_KEYS):
    raise RuntimeError("Need at least one LLM API key.")

app = Flask(__name__)
app.secret_key = FLASK_SECRET

session_req = requests.Session()
or_index = gm_index = 0

def _next_or_key():
    global or_index
    key = OPENROUTER_KEYS[or_index % len(OPENROUTER_KEYS)]
    or_index += 1
    return key

def _next_gm_key():
    global gm_index
    key = GEMINI_KEYS[gm_index % len(GEMINI_KEYS)]
    gm_index += 1
    return key

# ---------- LLM wrappers ----------
def _call_openrouter(msgs, timeout=20):
    url = "https://openrouter.ai/api/v1/chat/completions"
    for _ in range(len(OPENROUTER_KEYS)):
        hdrs = {"Authorization":f"Bearer {_next_or_key()}","Content-Type":"application/json"}
        payload = {"model":OPENROUTER_MODEL,"messages":msgs,"temperature":0.4}
        r = session_req.post(url,headers=hdrs,json=payload,timeout=timeout)
        if r.status_code==200 and r.json().get("choices"):
            return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("All OpenRouter keys failed.")

def _call_gemini(msgs, timeout=20):
    text = "\n".join(m["content"] for m in msgs if m.get("content"))
    payload = {"contents":[{"parts":[{"text":text}]}]}
    url_tpl = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"
    for _ in range(len(GEMINI_KEYS)):
        r = session_req.post(url_tpl.format(m=GEMINI_MODEL,k=_next_gm_key()),
                             json=payload,timeout=timeout)
        if r.status_code==200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raise RuntimeError("All Gemini keys failed.")

def call_llm(msgs):
    try:
        if GEMINI_KEYS:  return _call_gemini(msgs)
        raise RuntimeError
    except Exception:
        if OPENROUTER_KEYS: return _call_openrouter(msgs)
        raise

# --- ONE-SHOT helper ---------------------------------------------------------
def llm_one_shot(user_input:str)->dict:
    sys_prompt =  f"""
Return ONE JSON object exactly like the example below (no markdown fences):

{{
  "user_intent": "INSERT" | "QUERY",
  "insert": {{
    "Knowledge": "text supplied by user",
    "Reference": "any reference words",
    "Date": "YYYY-MM-DD"
  }},
  "query": "re-phrased question for lookup"
}}

Rules:
• user_intent = "INSERT" when user adds/saves info, else "QUERY".
• If intent is QUERY, set "insert" to {{}} and fill "query".
• If intent is INSERT, set "query" to "" and fill "insert".
• If Date is missing, default to "{currdate}".
"""
    raw = call_llm([{"role":"system","content":sys_prompt},
                    {"role":"user","content":user_input}])
    try:
        data = json.loads(raw)
        if all(k in data for k in ("user_intent","insert","query")):
            return data
    except Exception: pass   # fall back if LLM mis-formatted

    # Fallback – classify quickly
    fallback_intent = "INSERT" if any(w in user_input.lower() for w in ["add","save","remember","store"]) and len(user_input.split())>1 else "QUERY"
    return {"user_intent":fallback_intent,
            "insert": {"Knowledge":user_input,"Reference":"","Date":currdate} if fallback_intent=="INSERT" else {},
            "query": user_input if fallback_intent=="QUERY" else ""}

# ---------- Airtable ----------
def airtable_insert(fields):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    hdr = {"Authorization":f"Bearer {AIRTABLE_TOKEN}","Content-Type":"application/json"}
    session_req.post(url,headers=hdr,json={"records":[{"fields":fields}]}).raise_for_status()

def airtable_search(words):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    hdr = {"Authorization":f"Bearer {AIRTABLE_TOKEN}"}
    seen, out = set(), []
    for w in words:
        formula = f"FIND('{w.lower()}', LOWER({{Reference}}))"
        r = session_req.get(url,headers=hdr,params={"filterByFormula":formula}); r.raise_for_status()
        for rec in r.json().get("records",[]):
            if rec["id"] not in seen:
                seen.add(rec["id"]); out.append(rec)
    return out

def extract_keywords(query):
    raw = call_llm([{"role":"system","content":"Give comma-separated key words."},
                    {"role":"user","content":query}])
    return [w.strip() for w in raw.split(",") if w.strip()]

def answer_from_records(query, records):
    context = "\n".join(json.dumps(r["fields"],ensure_ascii=False) for r in records) or "No records."
    sys = ("Answer ONLY from these Airtable records.\n"
           "Address Krishna as 'you'.  If answer not found, say you don't know.")
    return call_llm([{"role":"system","content":sys},
                     {"role":"user","content":f"{query}\n\nRecords:\n{context}"}])

# ---------- HTML ----------
LOGIN_HTML = """<!doctype html><title>Login</title><form method=post>
<input type=password name=password placeholder=Password autofocus required>
<button>Sign in</button></form>"""

MAIN_HTML = """<!doctype html><title>MemoryBot</title>
<form method=post><textarea name=query rows=4 style=width:100%%></textarea>
<button>Submit</button></form>
{% if result %}<pre>{{ result }}</pre>{% endif %}"""

# ---------- Routes ----------
@app.route("/",methods=["GET"])
def index():
    return render_template_string(LOGIN_HTML if not session.get("ok") else MAIN_HTML,result=None)

@app.route("/login",methods=["POST"])
def login():
    if request.form.get("password")==UI_PASSWORD:
        session["ok"]=True; return redirect("/")
    flash("Wrong password"); return redirect("/")

@app.route("/logout")
def logout(): session.clear(); return redirect("/")

@app.route("/",methods=["POST"])
def ask():
    if not session.get("ok"): return redirect("/")
    q = request.form.get("query","").strip()
    if not q: flash("Enter something"); return redirect("/")

    try:
        data = llm_one_shot(q)
        if data["user_intent"]=="INSERT":
            airtable_insert(data["insert"])
            result = "✅ Memory saved, macha!"
        else:
            keywords = extract_keywords(data["query"])
            recs = airtable_search(keywords)
            result = answer_from_records(data["query"], recs)
    except Exception as e:
        result = f"Error: {e}"

    return render_template_string(MAIN_HTML,result=result)

# ---------- Run ----------
if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)))
