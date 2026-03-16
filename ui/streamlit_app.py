"""
Streamlit UI — companion memory chat system with multi-session support.
"""

import os
import sys
import time
import uuid
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import streamlit as st
import pandas as pd

from memory_engine.memory_store import MemoryStore
from memory_engine.memory_schema import MemoryStatus
from chat_system.conversation_manager import ConversationManager
from evals.eval_runner import EvalRunner

# --- Page Config ---
st.set_page_config(
    page_title="Companion AI",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e1e2e; }
    [data-testid="stSidebar"] * { color: #cdd6f4 !important; }
    [data-testid="stSidebar"] hr { border-color: #45475a; }

    .new-chat-btn button {
        background: #89b4fa !important;
        color: #1e1e2e !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100%;
    }

    .chat-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 2px 0;
        cursor: pointer;
        font-size: 0.88em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .chat-item-active {
        background: #45475a;
    }
    .chat-item:hover {
        background: #313244;
    }
    .chat-date {
        font-size: 0.72em;
        color: #6c7086 !important;
        padding: 8px 12px 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .memory-pill {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #b8daff;
        border-radius: 16px;
        padding: 2px 10px;
        margin: 2px 4px;
        font-size: 0.82em;
        color: #004085;
    }
    .memory-pill-extracted {
        background: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .memory-pill-correction {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .strategy-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 1px 8px;
        font-size: 0.75em;
        font-weight: 600;
        margin-left: 6px;
    }
    .strategy-recall { background: #d4edda; color: #155724; }
    .strategy-honest_missing { background: #f8d7da; color: #721c24; }
    .strategy-general { background: #e2e3e5; color: #383d41; }
    .strategy-history_recall { background: #cce5ff; color: #004085; }
    .strategy-ask_confirm { background: #fff3cd; color: #856404; }

    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .welcome-card h2 { color: white; margin-bottom: 0.5rem; }
    .welcome-card p { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SESSION / CHAT MANAGEMENT
# ============================================================

CHATS_FILE = str(PROJECT_ROOT / "chat_sessions.json")


def _init_chats_table():
    """Create chat sessions table in the memory store DB."""
    if "store" in st.session_state:
        try:
            st.session_state.store.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created TEXT NOT NULL,
                    messages TEXT NOT NULL DEFAULT '[]'
                )
            """)
            st.session_state.store.conn.commit()
        except Exception:
            pass


def load_all_chats() -> dict:
    """Load chat sessions from database, fallback to JSON file."""
    # Try database first
    if "store" in st.session_state:
        try:
            _init_chats_table()
            rows = st.session_state.store.conn.execute(
                "SELECT chat_id, user_id, created, messages FROM chat_sessions"
            ).fetchall()
            chats = {}
            for row in rows:
                chats[row[0]] = {
                    "user_id": row[1],
                    "created": row[2],
                    "messages": json.loads(row[3]),
                }
            return chats
        except Exception:
            pass

    # Fallback to JSON file
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_all_chats(chats: dict):
    """Persist chat sessions to database and JSON file."""
    # Save to database
    if "store" in st.session_state:
        try:
            _init_chats_table()
            for cid, data in chats.items():
                st.session_state.store.conn.execute(
                    "INSERT OR REPLACE INTO chat_sessions (chat_id, user_id, created, messages) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, data["user_id"], data["created"],
                     json.dumps(data["messages"], default=str))
                )
            # Clean up deleted chats
            existing_ids = set(chats.keys())
            db_rows = st.session_state.store.conn.execute(
                "SELECT chat_id FROM chat_sessions"
            ).fetchall()
            for row in db_rows:
                if row[0] not in existing_ids:
                    st.session_state.store.conn.execute(
                        "DELETE FROM chat_sessions WHERE chat_id = ?", (row[0],)
                    )
            st.session_state.store.conn.commit()
        except Exception:
            pass

    # Also save to JSON as backup
    try:
        with open(CHATS_FILE, "w") as f:
            json.dump(chats, f, indent=2, default=str)
    except Exception:
        pass


def get_chat_title(messages: list) -> str:
    """Generate a title from the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"][:40]
            return text + ("..." if len(msg["content"]) > 40 else "")
    return "New chat"


def create_new_chat(user_id: str) -> str:
    """Create a new chat session and return its ID."""
    chat_id = str(uuid.uuid4())[:8]
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = load_all_chats()

    st.session_state.all_chats[chat_id] = {
        "user_id": user_id,
        "created": datetime.now().isoformat(),
        "messages": [],
    }
    save_all_chats(st.session_state.all_chats)
    return chat_id


def init_session():
    if "store" not in st.session_state:
        db_path = str(PROJECT_ROOT / "memories_ui.db")
        # Try Turso cloud DB first (from st.secrets or env), fallback to local SQLite
        turso_url = None
        turso_token = None
        try:
            turso_url = st.secrets.get("TURSO_DATABASE_URL")
            turso_token = st.secrets.get("TURSO_AUTH_TOKEN")
        except Exception:
            pass
        st.session_state.store = MemoryStore(
            db_path=db_path, turso_url=turso_url, turso_token=turso_token
        )
        st.session_state.db_type = st.session_state.store.db_type

    if "manager" not in st.session_state:
        llm_fn = _build_llm_fn()
        st.session_state.manager = ConversationManager(
            store=st.session_state.store, llm_fn=llm_fn,
        )

    if "current_user" not in st.session_state:
        st.session_state.current_user = "user_1"

    # Load persisted chats
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = load_all_chats()

    # Create a default chat if none exist for current user
    if "active_chat_id" not in st.session_state:
        user_chats = _get_user_chats(st.session_state.current_user)
        if user_chats:
            st.session_state.active_chat_id = user_chats[0][0]
        else:
            st.session_state.active_chat_id = create_new_chat(st.session_state.current_user)


def _build_ollama_fn():
    """Build Ollama LLM function as local fallback."""
    try:
        import ollama as ollama_lib
        # Test connection
        ollama_lib.chat(model="llama3.2", messages=[{"role": "user", "content": "hi"}])
        st.session_state._ollama_model = "llama3.2"
        return True
    except Exception:
        pass
    try:
        import ollama as ollama_lib
        models = ollama_lib.list()
        if models and models.get("models"):
            model_name = models["models"][0]["name"]
            st.session_state._ollama_model = model_name
            return True
    except Exception:
        pass
    return False


def _call_ollama(messages: list[dict]) -> str:
    """Call Ollama for generation."""
    import ollama as ollama_lib
    model = st.session_state.get("_ollama_model", "llama3.2")
    resp = ollama_lib.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()


def _build_gemini_caller():
    """Build Gemini API caller. Returns (caller_fn, api_key_found)."""
    try:
        # Load .env locally
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

        from google import genai
        from google.genai import types

        # Try st.secrets first (Streamlit Cloud), then env var
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None, False

        client = genai.Client(api_key=api_key)
        MODEL_CHAIN = [
            "gemini-2.5-flash", "gemini-2.5-flash-lite",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
        ]

        def call_gemini(messages: list[dict]) -> str:
            system_prompt = ""
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    contents.append(types.Content(
                        role="user", parts=[types.Part(text=msg["content"])]
                    ))
                elif msg["role"] == "assistant":
                    contents.append(types.Content(
                        role="model", parts=[types.Part(text=msg["content"])]
                    ))
            config = types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                temperature=0.7, max_output_tokens=512,
            )
            for model in MODEL_CHAIN:
                for attempt in range(2):
                    try:
                        resp = client.models.generate_content(
                            model=model, contents=contents, config=config,
                        )
                        return resp.text.strip() if resp.text else None
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "quota" in err.lower() or "403" in err:
                            if attempt == 0:
                                time.sleep(2)
                                continue
                            break
                        elif "404" in err:
                            break
                        else:
                            raise
            return None  # All models rate-limited

        return call_gemini, True
    except Exception:
        return None, False


def _build_llm_fn():
    """
    Build LLM function with fallback chain:
    Gemini API → Ollama (local) → None (rule-based fallback in ConversationManager)
    """
    gemini_fn, has_gemini = _build_gemini_caller()
    has_ollama = _build_ollama_fn()

    if not has_gemini and not has_ollama:
        st.session_state.llm_status = "No LLM (add GEMINI_API_KEY or run Ollama)"
        st.session_state.llm_ok = False
        return None

    # Status label
    parts = []
    if has_gemini:
        parts.append("Gemini")
    if has_ollama:
        parts.append(f"Ollama ({st.session_state.get('_ollama_model', '?')})")
    st.session_state.llm_status = " + ".join(parts)
    st.session_state.llm_ok = True

    def llm_fn(messages: list[dict]) -> str:
        # Try Gemini first
        if gemini_fn:
            try:
                result = gemini_fn(messages)
                if result:
                    return result
            except Exception:
                pass  # Fall through to Ollama

        # Try Ollama as fallback
        if has_ollama:
            try:
                return _call_ollama(messages)
            except Exception:
                pass  # Fall through to None

        # Both failed — return None so ConversationManager uses rule-based fallback
        return None

    # Health check (non-blocking — don't fail init if rate-limited)
    try:
        test = llm_fn([{"role": "user", "content": "Say ok"}])
        if not test:
            st.session_state.llm_status += " (warming up)"
    except Exception:
        st.session_state.llm_status += " (warming up)"

    return llm_fn


def _get_user_chats(user_id: str) -> list[tuple[str, dict]]:
    """Get all chats for a user, sorted newest first."""
    chats = st.session_state.get("all_chats", {})
    user_chats = [
        (cid, data) for cid, data in chats.items()
        if data.get("user_id") == user_id
    ]
    user_chats.sort(key=lambda x: x[1].get("created", ""), reverse=True)
    return user_chats


def _get_active_messages() -> list:
    """Get messages for the active chat."""
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id:
        return []
    chats = st.session_state.get("all_chats", {})
    chat = chats.get(chat_id, {})
    return chat.get("messages", [])


def _save_message(role: str, content: str, debug: dict = None):
    """Append a message to the active chat and persist."""
    chat_id = st.session_state.active_chat_id
    chats = st.session_state.all_chats
    if chat_id not in chats:
        return
    msg = {"role": role, "content": content}
    if debug:
        msg["debug"] = debug
    chats[chat_id]["messages"].append(msg)
    save_all_chats(chats)


init_session()


# ============================================================
#  SIDEBAR
# ============================================================

with st.sidebar:
    # --- New Chat button ---
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        new_id = create_new_chat(st.session_state.current_user)
        st.session_state.active_chat_id = new_id
        # Clear conversation manager history for fresh context
        st.session_state.manager.clear_history(st.session_state.current_user)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Chat history list ---
    user_chats = _get_user_chats(st.session_state.current_user)
    active_id = st.session_state.get("active_chat_id", "")

    if user_chats:
        # Group by date
        today = datetime.now().date()
        groups = {}
        for cid, data in user_chats:
            try:
                created = datetime.fromisoformat(data["created"]).date()
            except Exception:
                created = today
            if created == today:
                label = "Today"
            elif (today - created).days == 1:
                label = "Yesterday"
            elif (today - created).days < 7:
                label = "This week"
            else:
                label = created.strftime("%B %Y")
            groups.setdefault(label, []).append((cid, data))

        for group_label, chats_in_group in groups.items():
            st.markdown(f'<div class="chat-date">{group_label}</div>', unsafe_allow_html=True)
            for cid, data in chats_in_group:
                title = get_chat_title(data.get("messages", []))
                is_active = cid == active_id
                icon = "💬" if not is_active else "▶"
                btn_type = "primary" if is_active else "secondary"

                if st.button(
                    f"{icon} {title}",
                    key=f"chat_{cid}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.active_chat_id = cid
                    # Reload conversation history into manager
                    st.session_state.manager.clear_history(st.session_state.current_user)
                    for msg in data.get("messages", []):
                        st.session_state.manager._add_to_history(
                            st.session_state.current_user, msg["role"], msg["content"]
                        )
                    st.rerun()

    st.markdown("---")

    # --- User switcher ---
    user_options = ["user_1", "user_2", "user_3", "user_4", "user_5"]
    selected_user = st.selectbox("Switch User", user_options,
        index=user_options.index(st.session_state.current_user))
    if selected_user != st.session_state.current_user:
        st.session_state.current_user = selected_user
        # Switch to the latest chat for this user or create one
        user_chats = _get_user_chats(selected_user)
        if user_chats:
            st.session_state.active_chat_id = user_chats[0][0]
        else:
            st.session_state.active_chat_id = create_new_chat(selected_user)
        st.session_state.manager.clear_history(selected_user)
        st.rerun()

    # --- Status ---
    llm_ok = st.session_state.get("llm_ok", False)
    st.markdown(f"{'🟢' if llm_ok else '🔴'} **{st.session_state.get('llm_status', 'No LLM')}**")

    mem_count = st.session_state.store.count(st.session_state.current_user)
    db_label = "Turso Cloud" if st.session_state.get("db_type") == "turso" else "Local SQLite"
    st.caption(f"Memories: {mem_count} | DB: {db_label} | User: {st.session_state.current_user}")

    st.markdown("---")

    # Navigation for non-chat pages
    page = st.radio("Pages", ["Chat", "Memory Inspector", "Eval Suite", "Benchmarks"],
                     label_visibility="collapsed")

    # --- Danger zone ---
    with st.expander("Manage"):
        if st.button("Delete this chat", use_container_width=True):
            cid = st.session_state.active_chat_id
            if cid in st.session_state.all_chats:
                del st.session_state.all_chats[cid]
                save_all_chats(st.session_state.all_chats)
            user_chats = _get_user_chats(st.session_state.current_user)
            if user_chats:
                st.session_state.active_chat_id = user_chats[0][0]
            else:
                st.session_state.active_chat_id = create_new_chat(st.session_state.current_user)
            st.session_state.manager.clear_history(st.session_state.current_user)
            st.rerun()

        if st.button("Clear all chats", use_container_width=True):
            uid = st.session_state.current_user
            st.session_state.all_chats = {
                k: v for k, v in st.session_state.all_chats.items()
                if v.get("user_id") != uid
            }
            save_all_chats(st.session_state.all_chats)
            st.session_state.active_chat_id = create_new_chat(uid)
            st.session_state.manager.clear_history(uid)
            st.rerun()

        if st.button("Reset memories", use_container_width=True):
            st.session_state.store.delete_user_memories(st.session_state.current_user)
            st.rerun()


# ============================================================
#  CHAT PAGE
# ============================================================

def render_memory_pills(debug_info):
    pills = ""
    for mem in debug_info.get("memories_extracted", []):
        mt = mem.get("type", "")
        css = "memory-pill-correction" if "corrected" in mt else "memory-pill-extracted"
        lbl = "corrected" if "corrected" in mt else "learned"
        pills += f'<span class="memory-pill {css}">{lbl}: {mem["entity"]}.{mem["attribute"]} = {mem["value"]}</span> '

    for mem in debug_info.get("memories_used", []):
        pills += f'<span class="memory-pill">{mem["entity"]}.{mem["attribute"]} = {mem["value"]}</span> '

    strategy = debug_info.get("strategy", "")
    label = {"recall": "from memory", "history_recall": "from conversation",
             "honest_missing": "no memory", "general": "general chat",
             "ask_confirm": "sensitive"}.get(strategy, strategy)
    pills += f'<span class="strategy-badge strategy-{strategy}">{label}</span>'
    return pills


def chat_page():
    user_id = st.session_state.current_user
    msgs = _get_active_messages()

    # Welcome screen
    if not msgs:
        st.markdown("""
        <div class="welcome-card">
            <h2>Hey there! I'm your AI Companion</h2>
            <p>I remember things you tell me across conversations.<br>
            Try telling me your name, pets, hobbies — I'll remember!</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Try saying:**")
        suggestions = [
            "Hi, my name is Arjun!",
            "I have a dog named Buddy",
            "I live in Mumbai and I like biryani",
            "My favorite color is blue",
        ]
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state._pending_msg = s
                st.rerun()

    # Render existing messages
    for msg in msgs:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("debug"):
                pills = render_memory_pills(msg["debug"])
                if pills:
                    st.markdown(pills, unsafe_allow_html=True)
                with st.expander("Debug", expanded=False):
                    st.json(msg["debug"])

    # Pending message from suggestion chip
    pending = st.session_state.pop("_pending_msg", None)

    # Chat input
    prompt = st.chat_input(f"Message as {user_id}...")
    if pending:
        prompt = pending

    if prompt:
        # Save & display user message
        _save_message("user", prompt)
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.manager.chat(user_id, prompt)
                    response = result["response"]
                except Exception as e:
                    response = f"Sorry, I ran into an issue: {str(e)[:100]}. Try again!"
                    result = {"response": response, "strategy": "error",
                              "memories_used": [], "memories_extracted": []}

            st.markdown(response)

            debug_info = {
                "strategy": result["strategy"],
                "memories_used": result["memories_used"],
                "memories_extracted": result["memories_extracted"],
            }
            pills = render_memory_pills(debug_info)
            if pills:
                st.markdown(pills, unsafe_allow_html=True)
            with st.expander("Debug", expanded=False):
                st.json(debug_info)

        _save_message("assistant", response, debug_info)
        st.rerun()


# ============================================================
#  MEMORY INSPECTOR
# ============================================================

def memory_inspector_page():
    st.title("Memory Inspector")
    user_id = st.session_state.current_user

    c1, c2 = st.columns(2)
    with c1:
        show_status = st.selectbox("Status", ["active", "superseded", "all"])
    with c2:
        entity_filter = st.text_input("Filter entity", "")

    if show_status == "all":
        memories = st.session_state.store.get_by_user(user_id, status=None)
    else:
        memories = st.session_state.store.get_by_user(user_id, status=MemoryStatus(show_status))

    if entity_filter:
        memories = [m for m in memories if entity_filter.lower() in m.entity.lower()]

    if not memories:
        st.info("No memories found.")
        return

    data = []
    for m in memories:
        icon = {"active": "🟢", "superseded": "🔴", "stale": "🟡"}.get(m.status.value, "⚪")
        data.append({
            "": f"{icon}",
            "Entity": m.entity,
            "Attribute": m.attribute,
            "Value": m.value,
            "Type": m.memory_type.value,
            "Confidence": f"{m.confidence:.0%}",
            "Sensitivity": m.sensitivity.value,
            "Time": m.timestamp.strftime("%m/%d %H:%M"),
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ============================================================
#  EVAL SUITE
# ============================================================

def eval_page():
    st.title("Evaluation Suite")

    if st.button("Run 77 eval cases", type="primary"):
        bar = st.progress(0)
        runner = EvalRunner()
        cases = runner.load_cases()
        results = []
        for i, case in enumerate(cases):
            results.append(runner.run_case(case))
            bar.progress((i + 1) / len(cases))
        st.session_state.eval_results = runner._aggregate(results)
        bar.empty()

    if "eval_results" not in st.session_state:
        st.info("Click above to run evaluations.")
        return

    r = st.session_state.eval_results
    cols = st.columns(5)
    cols[0].metric("Pass Rate", f"{r.pass_rate:.0%}")
    cols[1].metric("Hallucination", f"{r.hallucination_rate:.0%}")
    cols[2].metric("Recall", f"{r.memory_recall_rate:.0%}")
    cols[3].metric("Corrections", f"{r.correction_success_rate:.0%}")
    cols[4].metric("Isolation", f"{r.multi_user_isolation:.0%}")

    st.markdown("---")
    for cat, d in sorted(r.results_by_category.items()):
        bar = "🟩" * d["passed"] + "🟥" * (d["total"] - d["passed"])
        st.markdown(f"**{cat.replace('_',' ').title()}** {bar} {d['passed']}/{d['total']}")

    st.markdown("---")
    show = st.selectbox("Filter", ["All", "Failed Only", "Passed Only"])
    for res in r.individual_results:
        if show == "Failed Only" and res.passed:
            continue
        if show == "Passed Only" and not res.passed:
            continue
        icon = "✅" if res.passed else "❌"
        with st.expander(f"{icon} {res.case_id} — {res.category} (score: {res.score:.2f})"):
            st.markdown(f"**Response:** {res.response}")
            st.markdown(f"**Expected:** {res.expected_behavior}")
            if res.failure_reason:
                st.error(res.failure_reason)


# ============================================================
#  BENCHMARKS
# ============================================================

def benchmarks_page():
    st.title("Benchmarks")
    f = PROJECT_ROOT / "benchmarks" / "benchmark_results.json"
    if not f.exists():
        st.info("Run Eval Suite first.")
        return
    data = json.loads(f.read_text())
    c = st.columns(3)
    c[0].metric("Pass Rate", f"{data['pass_rate']:.0%}")
    c[1].metric("Hallucination", f"{data['hallucination_rate']:.0%}")
    c[2].metric("Recall", f"{data['memory_recall_rate']:.0%}")

    if data.get("results_by_category"):
        st.bar_chart(pd.DataFrame({
            "Category": [c.replace("_", " ").title() for c in data["results_by_category"]],
            "Pass Rate": [data["results_by_category"][c]["pass_rate"] for c in data["results_by_category"]],
        }).set_index("Category"))


# ============================================================
#  ROUTER
# ============================================================

if page == "Chat":
    chat_page()
elif page == "Memory Inspector":
    memory_inspector_page()
elif page == "Eval Suite":
    eval_page()
elif page == "Benchmarks":
    benchmarks_page()
