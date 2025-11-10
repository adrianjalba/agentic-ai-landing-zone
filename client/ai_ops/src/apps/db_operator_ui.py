import os
import asyncio
import threading
import queue
import time
from contextlib import AsyncExitStack
import concurrent.futures

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentFinish
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import StructuredTool
from mcp import StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

from src.llm.oci_genai import initialize_llm
from src.prompt_engineering.topics.db_operator import promt_oracle_db_operator
from src.tools.rag_agent import _rag_agent_service
from src.tools.python_scratchpad import run_python
from src.common.config import *
from src.utils.mcp_helper_connection import safe_connect

from langchain.callbacks.base import BaseCallbackHandler
import html, json, re, hashlib


# ─────────────────────────────────────────────
# SAFE RENDER + STABILITY HELPERS
# ─────────────────────────────────────────────
# Strip ASCII control chars except \t, \n, \r
_CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

def _safe_str(x) -> str:
    """Best-effort stringify + strip control chars."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    return _CONTROL_CHARS_RE.sub('', s)

def _safe_html_text(text: str) -> str:
    """Escape HTML and convert newlines to <br> safely."""
    s = _safe_str(text)
    return html.escape(s).replace("\n", "<br>")

def _render_html(container, html_str: str, fallback_text: str = ""):
    """Render HTML safely; on failure, show plain text instead (prevents UI crashes)."""
    try:
        container.markdown(html_str, unsafe_allow_html=True)
    except Exception as e:
        container.warning(f"Render fallback (invalid HTML): {e}")
        container.code(_safe_str(fallback_text or html_str), language="text")

def _hash_chat(chat_history: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for role, content in chat_history:
        h.update(role.encode("utf-8", errors="ignore")); h.update(b"\x00")
        h.update(str(content).encode("utf-8", errors="ignore")); h.update(b"\x00")
    return h.hexdigest()

def rerun_throttled(min_interval_sec: float = 1.0, key: str = "_last_rerun_ts"):
    """Prevent back-to-back reruns causing flicker."""
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last >= min_interval_sec:
        st.session_state[key] = now
        st.rerun()


class ReactLogHandler(BaseCallbackHandler):
    """Stream real-time logs to a queue; resilient to None/variant payloads."""
    def __init__(self, log_q):
        self.log_q = log_q

    def _ts(self):
        return time.strftime('%H:%M:%S')

    def _emit(self, emoji: str, prefix: str, msg: str):
        try:
            safe = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False, default=str)
        except Exception:
            safe = str(msg)
        safe = _safe_str(safe)  # sanitize at source
        self.log_q.put(f"[{self._ts()}] {emoji} {prefix}: {safe}")

    def _name_from_serialized(self, serialized):
        try:
            if isinstance(serialized, dict):
                if "name" in serialized and isinstance(serialized["name"], str):
                    return serialized["name"]
                if "id" in serialized:
                    sid = serialized["id"]
                    if isinstance(sid, (list, tuple)) and sid:
                        return ".".join(map(str, sid))
                    return str(sid)
                return json.dumps(serialized, ensure_ascii=False)[:120]
            if isinstance(serialized, (list, tuple)):
                return ".".join(map(str, serialized))
            if serialized is None:
                return "Unknown"
            return str(serialized)
        except Exception:
            return "Unknown"

    # callbacks
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = self._name_from_serialized(serialized)
        self._emit("🔗", "Chain Start", {"name": name, "inputs": inputs})

    def on_chain_end(self, outputs, **kwargs):
        self._emit("✅", "Chain End", outputs)

    def on_error(self, error, **kwargs):
        self._emit("💥", "Error", str(error))

    def on_llm_start(self, serialized, prompts, **kwargs):
        name = self._name_from_serialized(serialized)
        if isinstance(prompts, (list, tuple)):
            for p in prompts:
                self._emit("🧠", f"LLM Prompt ({name})", p)
        else:
            self._emit("🧠", f"LLM Prompt ({name})", prompts)

    def on_llm_end(self, response, **kwargs):
        text = None
        try:
            gens = getattr(response, "generations", None)
            if gens and len(gens) and len(gens[0]) and getattr(gens[0][0], "text", None):
                text = gens[0][0].text
        except Exception:
            pass
        self._emit("💡", "LLM Response", text or "<no generations>")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = self._name_from_serialized(serialized)
        self._emit("🧩", f"Action: {name}", input_str)

    def on_tool_end(self, output, **kwargs):
        self._emit("👁️", "Observation", output)

    def on_agent_action(self, action, **kwargs):
        log_text = getattr(action, "log", None)
        self._emit("⚙️", "Agent Action", log_text or str(action))

    def on_agent_finish(self, finish, **kwargs):
        try:
            val = getattr(finish, "return_values", {}) or {}
        except Exception:
            val = {}
        self._emit("🎯", "Final Answer", val.get("output") or val)


# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
class GlobalState:
    def __init__(self):
        self.prompt_q = queue.Queue()
        self.response_q = queue.Queue()
        self.log_q = queue.Queue()
        self.stop_flag = {"stop": False}
        self.trace_logs = []
        self.threads = {}
        self.pending_approvals = {}  # sql -> {"original_tool": tool, "payload": payload, "future": future}

if "global_state" not in st.session_state:
    st.session_state.global_state = GlobalState()
_global_state = st.session_state.global_state

st.set_page_config(page_title="Oracle DB Operator", layout="wide")
load_dotenv()


# ─────────────────────────────────────────────
# HELPERS (app logic)
# ─────────────────────────────────────────────
def is_sql_tool(tool) -> bool:
    nm = getattr(tool, "name", "") or ""
    return any(kw in nm.lower() for kw in ("adb", "sql", "oracle"))

class RunSQLInput(BaseModel):
    sql: str = Field(default="select sysdate from dual")
    model: str = Field(default="oci/generativeai-chat:2024-05-01")
    sqlcl: str = Field(default="sqlcl")

# Approval inside UI; execution stays in agent loop
async def user_confirmed_tool(original_tool, auto_approve):
    async def wrapper(*args, **kwargs):
        sql_query = kwargs.get("sql", args[0] if args else "select sysdate from dual")
        payload = {"sql": sql_query}
        if not auto_approve:
            _global_state.log_q.put(f"[DEBUG] Queuing SQL for approval: {sql_query}")
            future = concurrent.futures.Future()
            _global_state.pending_approvals[sql_query] = {
                "original_tool": original_tool,
                "payload": payload,
                "future": future,
            }
            try:
                decision = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
            except asyncio.TimeoutError:
                _global_state.pending_approvals.pop(sql_query, None)
                return "⚠️ SQL approval timed out after 5 minutes. Execution denied."
            except concurrent.futures.CancelledError:
                return "⚠️ SQL execution denied by user."

            approved = bool(decision is True or (isinstance(decision, dict) and decision.get("approved")))
            if not approved:
                # Abort chain immediately to stop agent continuation
                raise RuntimeError("SQL execution denied by user.")

            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(payload)
            elif hasattr(original_tool, "invoke"):
                return original_tool.invoke(payload)
            return original_tool.run(**payload)
        else:
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(payload)
            elif hasattr(original_tool, "invoke"):
                return original_tool.invoke(payload)
            return original_tool.run(**payload)

    return StructuredTool(
        name=original_tool.name,
        description=original_tool.description or "SQL tool",
        args_schema=RunSQLInput,
        coroutine=wrapper,
    )

def normalize_output(ai_response, history):
    if isinstance(ai_response, AgentFinish):
        msg = ai_response.return_values.get("output")
    elif isinstance(ai_response, dict):
        msg = ai_response.get("output")
    else:
        msg = ai_response
    if isinstance(msg, AIMessage):
        history.append(msg); return msg.content
    elif isinstance(msg, str):
        history.append(AIMessage(content=msg)); return msg
    elif isinstance(msg, dict) and "content" in msg:
        history.append(AIMessage(content=msg["content"])); return msg["content"]
    return "<<no response>>"

# Hide internal traces from AI panel
def _clean_ai_text(text: str) -> str:
    if not text: return ""
    keep = []
    for line in text.splitlines():
        if re.match(r'^\s*(Thought|Action|Observation|Agent Action|Tool|Reasoning)\b', line, re.I):
            continue
        keep.append(line)
    out = "\n".join(keep).strip()
    m = re.search(r'(?is)Final Answer:\s*(.+)$', out)
    return m.group(1).strip() if m else out

# Minimal markdown→HTML safe-ish rendering
def _md_to_html(md_text: str) -> str:
    if not md_text:
        return ""
    esc = html.escape(md_text)
    def repl(m):
        code = m.group(2)
        return (
            "<pre style='margin:8px 0;padding:10px;background:#0d0d0d;border-radius:6px;overflow:auto;'>"
            "<code>" + html.escape(code) + "</code></pre>"
        )
    esc = re.sub(r"```(\w+)?\n(.*?)```", repl, esc, flags=re.S)
    esc = re.sub(r"`([^`]+)`", r"<code style='background:#0d0d0d;padding:2px 4px;border-radius:4px;'>\1</code>", esc)
    esc = re.sub(r"(?m)^- (.+)$", r"• \1", esc)
    esc = esc.replace("\n", "<br>")
    return esc


# ─────────────────────────────────────────────
# ASYNC AGENT LOOP
# ─────────────────────────────────────────────
async def agent_loop(auto_approve):
    log = lambda m: _global_state.log_q.put(f"[{time.strftime('%H:%M:%S')}] {m}")
    try:
        log("🟡 Initializing LLM + MCP servers...")
        model = initialize_llm()

        adb_server = StdioServerParameters(command=SQLCLI_MCP_PROFILE, args=["-mcp"])
        tavily_server = StdioServerParameters(command="npx", args=["-y", "mcp-remote", TAVILY_MCP_SERVER])
        file_server = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", FILE_SYSTEM_ACCESS_KEY])
        url_redis = f"{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp"
        url_dbtools = f"{MCP_SSE_HOST_DBTOOLS}:{MCP_SSE_PORT_DBTOOLS}/mcp"

        async with AsyncExitStack() as stack:
            adb_sess = await safe_connect("Oracle SQLcl", adb_server, stack)
            tav_sess = await safe_connect("Tavily", tavily_server, stack)
            file_sess = await safe_connect("File Server", file_server, stack)
            red_sess = await safe_connect("Redis", url_redis, stack, http_streamable=True)
            dbt_sess = await safe_connect("DBTools", url_dbtools, stack, http_streamable=True)

            all_tools = []
            for name, sess in [
                ("SQLcl", adb_sess),
                ("Tavily", tav_sess),
                ("File Server", file_sess),
                ("Redis", red_sess),
                ("DBTools", dbt_sess),
            ]:
                if sess:
                    try:
                        tools = await load_mcp_tools(sess)
                        if tools:
                            log(f"🟢 [{name}] Connected ({len(tools)} tools)")
                        else:
                            log(f"🟡 [{name}] Connected but no tools found.")
                        all_tools += tools
                    except Exception as e:
                        log(f"🔴 [{name}] Failed to list tools: {e}")
                else:
                    log(f"🔴 [{name}] Connection failed.")

            tools_final = [await user_confirmed_tool(t, auto_approve) if is_sql_tool(t) else t for t in all_tools]
            tools_final += [run_python, _rag_agent_service]

            agent = initialize_agent(
                tools=tools_final,
                llm=model,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                agent_kwargs={"prefix": promt_oracle_db_operator},
                callbacks=[ReactLogHandler(_global_state.log_q)],
            )

            history = []
            log("🚀 Agent loop started.")

            while not _global_state.stop_flag["stop"]:
                try:
                    prompt = _global_state.prompt_q.get(timeout=0.5)
                    log(f"🟡 Received: {prompt}")
                    # >>> ADD THIS BLOCK <<<
                    if str(prompt).strip().lower() in {
                        "clear memory", "clear history", "reset", "reset memory", "/clear", "/reset"
                    }:
                        history.clear()
                        _global_state.response_q.put("🧠 Memory cleared. Conversation context reset.")
                        log("🧽 Cleared agent conversation history on user request.")
                        continue
                    # <<< END ADD >>>
                    history.append(HumanMessage(content=prompt))
                    ai_resp = await agent.ainvoke({"input": history})
                    msg = normalize_output(ai_resp, history)
                    _global_state.response_q.put(msg)
                    log("🟢 Response generated.")
                except queue.Empty:
                    await asyncio.sleep(0.1)
                except Exception as e:
                    log(f"🔴 Error: {e}")
                    # Ensure UI clears spinner on error
                    _global_state.response_q.put(f"❌ {e}")

            log("🟠 Agent shutting down.")
    except Exception as e:
        log(f"💥 Crash: {e}")
        _global_state.response_q.put(f"❌ {e}")


# ─────────────────────────────────────────────
# THREADING
# ─────────────────────────────────────────────
def start_agent_thread(auto_approve):
    if "agent" in _global_state.threads and _global_state.threads["agent"].is_alive():
        return
    _global_state.stop_flag["stop"] = False
    t = threading.Thread(target=lambda: asyncio.run(agent_loop(auto_approve)), daemon=True)
    t.start()
    _global_state.threads["agent"] = t

def start_log_stream():
    """Ultra-low latency log pump: blocks on .get() and never sleeps."""
    if "stream" in _global_state.threads and _global_state.threads["stream"].is_alive():
        return

    MAX_LOGS = 5000

    def pump():
        while not _global_state.stop_flag["stop"]:
            try:
                msg = _global_state.log_q.get(timeout=0.2)
                _global_state.trace_logs.append(_safe_str(msg))
                while True:
                    try:
                        _global_state.trace_logs.append(_safe_str(_global_state.log_q.get_nowait()))
                    except queue.Empty:
                        break
                if len(_global_state.trace_logs) > MAX_LOGS:
                    del _global_state.trace_logs[: len(_global_state.trace_logs) - MAX_LOGS]
            except queue.Empty:
                continue

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    _global_state.threads["stream"] = t


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.title("Oracle DB Operator 💬")
    st.caption("Conversational Agent for Oracle MCP Servers")

    # ---------- Session ----------
    ss = st.session_state
    ss.setdefault("busy", False)
    ss.setdefault("suppress_refresh", False)
    ss.setdefault("chat_history", [])
    ss.setdefault("chat_input", "")
    ss.setdefault("pending_response", False)
    ss.setdefault("_chat_hash", None)
    ss.setdefault("_chat_rendered_once", False)

    if ss.get("_clear_chat_input", False):
        ss._clear_chat_input = False
        ss.pop("chat_input", None)

    auto_approve = st.sidebar.checkbox("Auto-approve SQL executions", value=False)
    agent_running = ("agent" in _global_state.threads) and _global_state.threads["agent"].is_alive()
    st.sidebar.markdown(f"**Status:** {'🟢 Running' if agent_running else '🔴 Stopped'}")

    # Optional live refresh (slow)
    live_refresh = st.sidebar.toggle(
        "Live refresh logs",
        value=False,
        help="When ON, page re-runs every 4 seconds to pull new logs."
    )

    # Test button to add a fake pending SQL
    if st.sidebar.button("🧪 Test Add Pending SQL"):
        from langchain_core.tools import Tool
        fake_tool = Tool(name="test_sql", description="Fake SQL tool", func=lambda x: "Fake result")
        future = concurrent.futures.Future()
        _global_state.pending_approvals["SELECT 1 FROM DUAL;"] = {
            "original_tool": fake_tool,
            "payload": {"sql": "SELECT 1 FROM DUAL;"},
            "future": future
        }
        st.info("Added test SQL for approval.")
        rerun_throttled(0.5)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        _global_state.trace_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting MCP Agent…")
        with st.spinner("Starting MCP Agent..."):
            start_agent_thread(auto_approve)
            start_log_stream()
            st.success("✅ Agent started")
        # brief drain
        t_end = time.time() + 1.0
        while time.time() < t_end:
            drained = False
            try:
                while not _global_state.log_q.empty():
                    _global_state.trace_logs.append(_safe_str(_global_state.log_q.get_nowait()))
                    drained = True
            except queue.Empty:
                pass
            if not drained:
                time.sleep(0.05)
        rerun_throttled(0.5)

    if c2.button("🛑 Stop Agent"):
        _global_state.stop_flag["stop"] = True
        st.warning("Stopping agent...")

    # ---------- Logs (CRASH-PROOF) ----------
    st.subheader("📜 Agent Logs (Live)")
    if st.button("🧹 Clear Logs"):
        _global_state.trace_logs.clear()

    try:
        while not _global_state.log_q.empty():
            _global_state.trace_logs.append(_safe_str(_global_state.log_q.get_nowait()))
    except queue.Empty:
        pass

    logs_box = st.empty()
    log_text_raw = "\n".join(_global_state.trace_logs[-1500:]) or "No logs yet..."
    log_html = (
        "<div id='logsbox' style='background-color:#111;color:#EEE;padding:10px;"
        "border-radius:8px;height:300px;overflow-y:auto;"
        "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
        "font-size:13px;line-height:1.35;'>"
        + _safe_html_text(log_text_raw) +
        f"</div><span id='logsmarker' data-n='{len(_global_state.trace_logs)}' style='display:none;'></span>"
        "<script>(function(){"
        "  var el=document.getElementById('logsbox'); if(!el) return;"
        "  function tail(){"
        "    el.scrollTop = el.scrollHeight;"
        "    el.scrollTop = el.scrollHeight - el.clientHeight + 1;"  # ensure bottom even if layout shifts
        "  }"
        "  tail();"
        "  requestAnimationFrame(tail);"
        "  setTimeout(tail,0);"
        "})();</script>"
    )



    _render_html(logs_box, log_html, fallback_text=log_text_raw)

    # ---------- Conversation (STABLE, NO FLICKER) ----------
    st.markdown("#### 💬 Conversation")
    chat_ph = st.empty()

    def render_chat(always: bool = False):
        try:
            new_hash = _hash_chat(ss.chat_history)
        except Exception:
            new_hash = str(time.time())

        if not (always or not ss._chat_rendered_once or new_hash != ss._chat_hash):
            return

        ss._chat_hash = new_hash
        ss._chat_rendered_once = True

        bubbles = []
        for role, content in ss.chat_history:
            if role == "user":
                safe = _md_to_html(str(content))  # escapes internally
                bubbles.append(
                    "<div style='background:#2a2a2a;color:#fff;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                    "<b>👤 You:</b><br>" + safe + "</div>"
                )
            else:
                cleaned = _clean_ai_text(str(content))
                safe = _md_to_html(cleaned)
                bubbles.append(
                    "<div style='background:#1e1e1e;color:#9CDCFE;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                    "<b>🤖 AI:</b><br>" + safe + "</div>"
                )

        html_block = (
            "<div id='chatbox' style='background:#111;color:#EEE;padding:10px;border-radius:8px;"
            "height:300px;overflow-y:auto;'>"
            + ("".join(bubbles) if bubbles else "<i>No conversation yet...</i>")
            + f"</div><span id='chatmarker' data-n='{len(ss.chat_history)}' style='display:none;'></span>"
            "<script>(function(){"
            "  var cb=document.getElementById('chatbox'); if(!cb) return;"
            "  function tail(){"
            "    cb.scrollTop = cb.scrollHeight;"
            "    cb.scrollTop = cb.scrollHeight - cb.clientHeight + 1;"
            "  }"
            "  tail();"
            "  requestAnimationFrame(tail);"
            "  setTimeout(tail,0);"
            "})();</script>"
        )


        _render_html(chat_ph, html_block, fallback_text="(chat render fallback)")

    # initial render
    render_chat(always=True)

    # ---------- Pending SQL Approvals (readable monospace cards) ----------
    pending_sqls = list(_global_state.pending_approvals.keys())
    if pending_sqls:
        st.markdown("### 📝 Pending SQL Approvals")

        for sql in pending_sqls:
            item = _global_state.pending_approvals[sql]

            # Monospace, wrapped, high-contrast SQL card
            sql_card_html = (
                "<div style='border:1px solid #333; background:#0d0d0d; border-radius:10px; padding:12px; margin:8px 0;'>"
                "<div style='color:#bbb; font-size:12px; margin-bottom:6px;'>SQL Query</div>"
                "<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
                "            font-size:13px; line-height:1.45; color:#E8E8E8; white-space:pre-wrap; word-break:break-word;'>"
                + _safe_html_text(sql) +
                "</div>"
                "</div>"
            )
            _render_html(st, sql_card_html, fallback_text=sql)

            col1, col2 = st.columns(2)
            if col1.button("✅ Approve and Execute", key=f"approve_{hash(sql)}"):
                _global_state.log_q.put(f"[DEBUG] Approving SQL: {sql}")
                try:
                    item['future'].set_result({"approved": True})
                    st.success("✅ Approved. Execution will proceed.")
                    _global_state.log_q.put(f"[DEBUG] Future approval set for {sql}")
                except Exception as e:
                    _global_state.log_q.put(f"[DEBUG] Approval exception: {e}")
                    st.error(f"❌ Error: {e}")
                finally:
                    _global_state.pending_approvals.pop(sql, None)
                rerun_throttled(0.5)

            if col2.button("❌ Deny", key=f"deny_{hash(sql)}"):
                _global_state.log_q.put(f"[DEBUG] Denying SQL: {sql}")
                try:
                    item['future'].set_result({"approved": False})
                except Exception:
                    try:
                        item['future'].cancel()
                    except Exception:
                        pass
                finally:
                    _global_state.pending_approvals.pop(sql, None)
                rerun_throttled(0.5)
    else:
        st.info(f"No pending SQL approvals. Pending count: {len(_global_state.pending_approvals)}")


    # ---------- Input ----------
    status_ph = st.empty()
    if ss.pending_response:
        _render_html(
            status_ph,
            "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
            fallback_text="Thinking…"
        )

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "💬 Ask your question:",
            placeholder="Type your SQL or question here...",
            key="chat_input",
            height=80,
        )
        submitted = st.form_submit_button("🚀 Run", use_container_width=True)

    if submitted:
        q = (user_query or "").strip()
        agent_running_now = ("agent" in _global_state.threads) and _global_state.threads["agent"].is_alive()

        if not q:
            st.warning("Please enter a question.")
        elif not agent_running_now:
            st.error("Agent not running. Please start it first.")
        else:
            _global_state.prompt_q.put(q)
            ss.chat_history.append(("user", q))
            ss.pending_response = True
            _render_html(
                status_ph,
                "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
                fallback_text="Thinking…"
            )
            ss._clear_chat_input = True
            rerun_throttled(0.5)

    # ---------- Check for AI response (NO extra rerun here) ----------
    if ss.pending_response:
        try:
            reply = _global_state.response_q.get_nowait()
            ss.chat_history.append(("ai", reply))
            ss.pending_response = False
            status_ph.empty()
            render_chat(always=True)  # update chat, no st.rerun() → no flicker
        except queue.Empty:
            pass

    # ---------- Optional auto-refresh logs only ----------
    try:
        if live_refresh:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=4000, key="live_log_refresh")
    except Exception:
        pass


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
