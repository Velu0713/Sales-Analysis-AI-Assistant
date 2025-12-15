# llm.py
import os
import json
import requests
from typing import List, Dict

LMSTUDIO_BASE = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1")
MODEL_NAME = os.getenv("LM_MODEL_NAME", "qwen2.5-3b-instruct")
HEADERS = {"Content-Type": "application/json"}

TEMPERATURE = float(os.getenv("LM_TEMP", "0.2"))
MAX_TOKENS = int(os.getenv("LM_MAX_TOKENS", "512"))

SYSTEM_PROMPT = """
You are SalesGPT — a helpful assistant that answers using ONLY the provided CONTEXT when present.
If the context lacks required facts, say: "The dataset does not provide this information. Want me to estimate or analyze?"
Be concise and professional. Do not hallucinate facts.
"""

def build_messages(system_prompt: str, retrieved: List[Dict], chat_history: List[Dict], user_question: str):
    # messages must only use roles: system | user | assistant | tool
    messages = [{"role": "system", "content": system_prompt}]
    if retrieved:
        ctx_lines = []
        for r in retrieved:
            rid = r.get("id", "?")
            txt = r.get("text", "")[:800]
            ctx_lines.append(f"ROW {rid}: {txt}")
        messages.append({"role": "system", "content": "CONTEXT:\n" + "\n\n".join(ctx_lines)})

    # chat_history: list of {"role":"user"/"assistant", "content": "..."}
    for m in (chat_history or []):
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
        # ignore other roles if present

    # finally add the user's current question as 'user'
    messages.append({"role":"user","content": user_question})
    return messages

def call_local_llm(messages):
    url = f"{LMSTUDIO_BASE}/chat/completions"
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    r = requests.post(url, headers=HEADERS, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()

def generate_answer_with_rag(user_question: str, retrieved: List[Dict], chat_history=None):
    if chat_history is None:
        chat_history = []
    messages = build_messages(SYSTEM_PROMPT, retrieved or [], chat_history, user_question)
    try:
        resp = call_local_llm(messages)
    except Exception as e:
        return {"answer": f"LLM API HTTP error: {e}", "raw": None}
    # extract assistant content
    try:
        answer = resp["choices"][0]["message"]["content"]
    except Exception:
        answer = json.dumps(resp)
    if isinstance(answer, str) and answer.lower().startswith("assistant:"):
        answer = answer[len("assistant:"):].strip()
    return {"answer": answer, "raw": resp}

def get_models(timeout=5):
    try:
        r = requests.get(f"{LMSTUDIO_BASE}/models", headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}