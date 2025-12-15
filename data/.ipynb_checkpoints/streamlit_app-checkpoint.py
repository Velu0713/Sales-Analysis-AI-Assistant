# streamlit_app.py
import os
import time
import io
from typing import List, Optional, Any, Tuple

import pandas as pd
import streamlit as st

from rag import load_index, build_index_from_text_rows, retrieve, rerank_with_cosine, INDEX_DIR
from llm import generate_answer_with_rag, get_models
import analysis_engine as ae

# ==============================================================
# CONFIG — use your requested dataset path
# ==============================================================
CSV_PATH = "data/Sales.csv"
TOP_K = 5

st.set_page_config(page_title="SalesGPT — Chat + Data Analyzer",
                   layout="centered",
                   initial_sidebar_state="collapsed")

# ==============================================================
# CSS (keeps your design)
# ==============================================================
CHAT_CSS = """
<style>
body { background:#0d0f14; color:#e9ecef; }
.chat-shell { width:900px; margin:0 auto; padding:18px 8px 140px 8px; }
.header { text-align:center; margin-bottom:12px; }
.sub { color:#9aa0a6; font-size:14px; margin-top:-8px; }
.msg-row{display:flex; gap:12px; margin:10px 0; align-items:flex-end;}
.msg-bot{background:#13131a;border-radius:14px;padding:12px 14px;
         max-width:78%;border-left:4px solid rgba(122,92,255,0.95);}
.msg-user{background:linear-gradient(135deg,#00d1b2,#00f4d8);
          color:#041412;border-radius:14px;padding:12px 14px;
          max-width:78%;margin-left:auto;}
.avatar-bot{width:36px;height:36px;border-radius:50%;
            display:flex;align-items:center;justify-content:center;
            background:linear-gradient(135deg,#6d5dfc,#8a7dff);
            color:white;margin-right:6px;}
.avatar-user{width:36px;height:36px;border-radius:50%;
             display:flex;align-items:center;justify-content:center;
             background:linear-gradient(135deg,#00d1b2,#00f4d8);
             color:white;margin-left:6px;}
.input-bar{position:fixed;left:0;right:0;bottom:18px;
           display:flex;justify-content:center;z-index:9999;}
.input-inner{width:900px;max-width:calc(100% - 40px);
             background:#121317;border-radius:999px;padding:10px 14px;
             display:flex;gap:10px;align-items:center;
             border:1px solid #24262b;}
.sources{background:#0b0c0e;border-radius:8px;padding:10px;margin-top:12px;
         border:1px solid #1f2227;color:#dbe3ea;}
.small-muted { color:#9aa0a6; font-size:13px; margin-top:6px; }
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

# ==============================================================
# Helpers
# ==============================================================
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

def load_dataset(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception:
            return None

def fig_to_png_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes for persistence."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf.read()
    finally:
        try:
            buf.close()
        except Exception:
            pass

# safe minimal data-question detector — only triggers on whole-word matches to reduce false positives
def is_data_question(q: str) -> bool:
    q = (q or "").lower().strip()
    if not q:
        return False
    words = set([w.strip(".,?") for w in q.split()])
    keywords = {
        "sales", "profit", "order", "orders", "category", "region", "segment",
        "discount", "quantity", "product", "customer", "trend", "forecast", "top",
        "total", "sum", "average", "count", "month", "year"
    }
    return len(words.intersection(keywords)) > 0

# ==============================================================
# Load data + index
# ==============================================================
df = load_dataset(CSV_PATH)

index_pack = load_index()
if index_pack is None:
    index = None
    meta_ids: List[Any] = []
    meta_texts: List[str] = []
else:
    index, meta_ids, meta_texts, _ = index_pack

# ==============================================================
# Sidebar
# ==============================================================
with st.sidebar:
    st.header("Dataset & Index")
    st.write(f"Dataset path: {CSV_PATH}")
    if df is None:
        st.error("Dataset not found at this path.")
        uploaded = st.file_uploader("Upload Sales CSV (Superstore-like)", type=["csv", "xlsx"])
        if uploaded:
            try:
                df2 = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
                df2.to_csv(CSV_PATH, index=False)
                st.success("Saved dataset. Please reload the app.")
            except Exception as e:
                st.error("Save failed: " + str(e))
    else:
        st.success(f"Dataset loaded — rows: {len(df)}, cols: {len(df.columns)}")
        if st.button("Preview head"):
            st.dataframe(df.head(8), width="stretch")

    st.write("---")
    st.markdown("### FAISS Index")
    idx_exists = os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and os.path.exists(os.path.join(INDEX_DIR, "meta.pkl"))
    if idx_exists:
        st.success("FAISS index present")
        if st.button("Delete index files"):
            try:
                os.remove(os.path.join(INDEX_DIR, "index.faiss"))
                os.remove(os.path.join(INDEX_DIR, "meta.pkl"))
                st.success("Deleted index files")
            except Exception as e:
                st.error("Delete failed: " + str(e))
    else:
        st.warning("No FAISS index found")

    if st.button("Build / Rebuild index"):
        if df is None:
            st.error("No dataset to build index from")
        else:
            rows = []
            for i, r in df.iterrows():
                combined = " | ".join([f"{c}: {r[c]}" for c in df.columns])
                rows.append((i, combined))
            build_index_from_text_rows(rows)
            st.success("Index built")

    st.write("---")
    st.markdown("### LM Studio / Local LLM")
    try:
        st.json(get_models())
    except Exception:
        st.info("LM Studio not reachable or not configured")

# ==============================================================
# Session state initialization
# ==============================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"|"assistant", "text": "...", optional "img_key": idx}
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "img_store" not in st.session_state:
    st.session_state.img_store = {}  # map image_key -> png bytes (persist figures)
if "img_next" not in st.session_state:
    st.session_state.img_next = 0

# ==============================================================
# UI header + chat history rendering
# ==============================================================
st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
st.markdown("<div class='header'><h2 style='margin:0'>SalesGPT — Chat + Data Analyzer</h2>"
            "<div class='sub'>Ask totals, top-k, trends, forecasts. Local analysis → LLM fallback.</div></div>", unsafe_allow_html=True)

# Render conversation
for msg in st.session_state.history:
    role = msg.get("role", "assistant")
    text = msg.get("text", "")
    if role == "user":
        st.markdown(f"<div class='msg-row'><div class='msg-user'>{text}</div><div class='avatar-user'>U</div></div>", unsafe_allow_html=True)
    else:
        # assistant message — may have an image
        st.markdown(f"<div class='msg-row'><div class='avatar-bot'>AI</div><div class='msg-bot'>{text}</div></div>", unsafe_allow_html=True)
        img_key = msg.get("img_key", None)
        if img_key is not None:
            png = st.session_state.img_store.get(img_key)
            if png:
                st.image(png, use_column_width=True)

# Show sources ONLY if present (dataset question that had retrieval)
if st.session_state.last_sources:
    src_lines = []
    for s in st.session_state.last_sources:
        txt = s.get("text", "") if isinstance(s, dict) else str(s)
        preview = txt[:260].replace("\n", " ")
        src_lines.append(f"<li>{preview}</li>")
    st.markdown("<div class='sources'><b>Sources (top matches):</b><ul>" + "".join(src_lines) + "</ul></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================
# Input form (clear_on_submit ensures text input resets cleanly)
# ==============================================================
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask about your sales data (press Enter or Send)", key="chat_input", placeholder="Write a message…", label_visibility="collapsed")
    submit = st.form_submit_button("SEND")

# ==============================================================
# Main processing when the form is submitted
# ==============================================================
if submit and user_input and user_input.strip():
    user_q = user_input.strip()

    # append user message
    st.session_state.history.append({"role": "user", "text": user_q})
    st.session_state.last_sources = []

    # 1) Local analysis engine (fast) — prefer direct answers & charts
    analysis_text: Optional[str] = None
    analysis_fig = None
    try:
        res = ae.analyze_query(user_q, df.copy()) if df is not None else (None, None)
        # analysis_engine may return (text, fig) or (text, fig, ...)
        if isinstance(res, tuple):
            if len(res) >= 1:
                analysis_text = res[0]
            if len(res) >= 2:
                analysis_fig = res[1]
    except Exception as e:
        # log small error in sidebar — don't crash UI
        st.sidebar.error("Analysis engine error: " + str(e))
        analysis_text, analysis_fig = None, None

    # If analysis produced text or figure, present that and skip retrieval
    if (analysis_text and str(analysis_text).strip()) or (analysis_fig is not None):
        bot_text = analysis_text or "I generated a chart for your query."
        # if chart exists, convert to bytes and store to session so it persists
        img_key = None
        if analysis_fig is not None:
            try:
                png = fig_to_png_bytes(analysis_fig)
                img_key = f"img_{st.session_state.img_next}"
                st.session_state.img_store[img_key] = png
                st.session_state.img_next += 1
            except Exception as e:
                # if conversion fails, don't break; only text will be shown
                st.sidebar.error("Failed to persist figure: " + str(e))
                img_key = None

        # append assistant message (with optional img_key)
        msg = {"role": "assistant", "text": bot_text}
        if img_key:
            msg["img_key"] = img_key
        st.session_state.history.append(msg)

        # keep last_sources empty (local result)
        st.session_state.last_sources = []
        safe_rerun()

    # 2) Non-data question -> route directly to LLM (full chat_history passed)
    if not is_data_question(user_q):
        # assemble chat_history for LLM (map roles correctly)
        chat_history_messages = []
        for m in st.session_state.history:
            r = m.get("role")
            t = m.get("text", "")
            if r == "user":
                chat_history_messages.append({"role": "user", "content": t})
            else:
                chat_history_messages.append({"role": "assistant", "content": t})

        try:
            resp = generate_answer_with_rag(user_q, retrieved=[], chat_history=chat_history_messages)
            answer = resp.get("answer", "No response from model.")
        except Exception as e:
            answer = f"LLM error: {e}"

        st.session_state.history.append({"role": "assistant", "text": answer})
        safe_rerun()

    # 3) Data-question -> retrieval + rerank -> LLM
    try:
        # retrieval only if index present
        if index is None:
            retrieved = []
        else:
            retrieved = retrieve(user_q, None, index, meta_ids, meta_texts, k=TOP_K * 2)

        # rerank
        if retrieved:
            reranked = rerank_with_cosine(user_q, None, retrieved, top_k=TOP_K)
        else:
            reranked = []

        # evidence heuristic: use reranked[0].rerank_score if present
        evidence_present = False
        if reranked:
            top = reranked[0]
            score = top.get("rerank_score", None)
            if score is None:
                evidence_present = True
            else:
                try:
                    evidence_present = float(score) >= 0.18
                except Exception:
                    evidence_present = False

        # Typing placeholder
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<div class='msg-row'><div class='avatar-bot'>AI</div><div class='msg-bot'>Thinking…</div></div>", unsafe_allow_html=True)
            time.sleep(0.4)

        # build chat history messages (map roles)
        chat_history_messages = []
        for m in st.session_state.history:
            r = m.get("role")
            t = m.get("text", "")
            if r == "user":
                chat_history_messages.append({"role": "user", "content": t})
            else:
                chat_history_messages.append({"role": "assistant", "content": t})

        if evidence_present:
            resp = generate_answer_with_rag(user_q, retrieved=reranked, chat_history=chat_history_messages)
            st.session_state.last_sources = reranked
        else:
            resp = generate_answer_with_rag(user_q, retrieved=[], chat_history=chat_history_messages)
            st.session_state.last_sources = []

        ans = resp.get("answer", "No response from model.")
    except Exception as e:
        ans = f"LLM error: {e}"

    st.session_state.history.append({"role": "assistant", "text": ans})
    safe_rerun()