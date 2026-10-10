"""Streamlit chat interface for the NexaChain Supply Operations agent."""
import os
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

st.set_page_config(page_title="NexaChain Supply Operations Assistant", page_icon="📦",layout="wide")

from agent import  build_graph, build_knowledge_base, get_pending_escalations, list_accounts  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MODEL = "openai/gpt-oss-120b"
EXAMPLES = [
    "What is the cancellation fee if a PO is cancelled 36 hours before the scheduled ship date?",
    "A supplier delivery is four days late due to a customs delay. Does BrightPoint's contract entitle them to a service credit, and how much?",
    "What service credit does Vertex Robotics get for PO-6301?",
    "How is stock for SKU-EC-4410 and SKU-MC-1050? How many days until we stock out of SKU-MC-1050?",
    "Please escalate case CASE-9003 to supply operations. The customer disputes the fee.",
    "What cancellation fee applies to PO-2044?",
]


# ---------------------------------------------------------------- resources --
@st.cache_resource(show_spinner="Indexing NexaChain documents (first run takes a minute)…")
def get_kb():
    return build_knowledge_base(DATA_DIR)


@st.cache_resource(show_spinner=False)
def get_graph(model_name: str, api_key: str):
    from langchain_groq import ChatGroq

    return build_graph(get_kb(), ChatGroq(model=model_name, api_key=api_key, temperature=0))


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")

TOOL_META = {
    "search_documents": ("🔍", "Document search"),
    "lookup_supply_data": ("🗄️", "Supply data lookup"),
    "calculate": ("🧮", "Calculator"),
    "request_escalation": ("🚨", "Escalation (needs confirmation)"),
}


if __name__ == "__main__":
    # app code
    ...

# ------------------------------------------------------------------ helpers --
def badge(name: str) -> str:
    icon, label = TOOL_META.get(name, ("🔧", name))
    return f":blue-badge[{icon} {label}]"


def render_tools(tools: list):
    if not tools:
        st.caption("No tools used, answered from the conversation alone.")
        return
    st.markdown(" ".join(badge(t["name"]) for t in tools))
    with st.expander(f"Tool details ({len(tools)})"):
        for t in tools:
            st.markdown(f"**{TOOL_META.get(t['name'], ('🔧',))[0]} `{t['name']}`**")
            st.json(t["args"], expanded=False)
            if t.get("result") is not None:
                st.code(t["result"][:1500] + ("…" if len(t["result"]) > 1500 else ""), language="text")


def init_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("pending", None)  # interrupt payload awaiting confirmation
    ss.setdefault("turn_tools", [])  # tool events for the turn in progress


def run_agent(graph, payload, permitted):
    """Stream one agent run, showing tool calls live. Handles the confirmation interrupt."""
    config = {"configurable": {"thread_id": st.session_state.thread_id, "permitted_accounts": permitted}}
    tools = st.session_state.turn_tools
    answer, interrupted = "", None

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        try:
            for update in graph.stream(payload, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    interrupted = update["__interrupt__"][0].value
                    continue
                for _, data in update.items():
                    for m in (data or {}).get("messages", []):
                        if isinstance(m, AIMessage):
                            if m.tool_calls:
                                for c in m.tool_calls:
                                    tools.append({"id": c["id"], "name": c["name"], "args": c["args"], "result": None})
                                    status.write(f"{badge(c['name'])} `{c['args']}`")
                            elif m.content:
                                answer = m.content
                        elif isinstance(m, ToolMessage):
                            for t in tools:
                                if t["id"] == m.tool_call_id:
                                    t["result"] = str(m.content)
        except Exception as e:  # rate limits, malformed tool calls, etc.
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.session_state.turn_tools = []
            return

        if interrupted:
            status.update(label="Waiting for your confirmation", state="running", expanded=True)
            st.session_state.pending = interrupted
            return

        status.update(
            label=f"Used {len(tools)} tool call(s)" if tools else "Answered without tools",
            state="complete",
            expanded=False,
        )
        st.markdown(answer)
        render_tools(tools)

    st.session_state.messages.append({"role": "assistant", "content": answer, "tools": list(tools)})
    st.session_state.turn_tools = []


# --------------------------------------------------------------------- main --
init_state()
st.title("📦 NexaChain Supply Operations Assistant")


st.caption("Ask about policies, contracts, purchase orders, inventory and escalations. Tool usage is shown on every answer.")

if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    st.error("Put the NexaChain PDFs and `NexaChain_Assessment_Data.xlsx` in the `data/` folder and reload.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    api_key = secret("GROQ_API_KEY") or st.text_input("Groq API key", type="password", help="Or set GROQ_API_KEY in secrets.")
    model_name = st.text_input("Model", DEFAULT_MODEL)

    kb = get_kb()
    accounts = list_accounts(kb.tables)  # [(Account_ID, Account_Name)]
    permitted = None
    if accounts:
        INTERNAL = "Internal staff (all accounts)"
        labels = {f"{i} · {n}": i for i, n in accounts}
        choice = st.selectbox(
            "Signed in as",
            [INTERNAL, *labels],
            help="Access control is enforced in the tools: only rows for the signed-in Account_ID are returned. Switch to test it.",
            on_change=lambda: st.session_state.update(messages=[], thread_id=str(uuid.uuid4()), pending=None, turn_tools=[]),
        )
        permitted = None if choice == INTERNAL else [labels[choice]]
    if kb.snapshot:
        st.caption(f"🕒 Dataset time (\"now\"): {kb.snapshot}")

    st.divider()
    st.subheader("Tools")
    for name, (icon, label) in TOOL_META.items():
        st.markdown(f"{icon} {label}")
    st.caption(f"{len(kb.files)} files · {kb.n_chunks} text chunks · {len(kb.tables)} data sheets")

    st.divider()
    st.subheader("Try an example")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", use_container_width=True, disabled=bool(st.session_state.pending)):
            st.session_state.queued = ex

    st.divider()
    pending_log = get_pending_escalations()
    st.subheader(f"Pending escalations ({len(pending_log)})")
    for e in pending_log[-5:]:
        st.caption(f"**{e['priority']}** · {e.get('reference') or 'n/a'} · {e['summary'][:80]}")

    if st.button("🗑️ New conversation", use_container_width=True):
        for k in ("messages", "thread_id", "pending", "turn_tools"):
            st.session_state.pop(k, None)
        st.rerun()

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()

graph = get_graph(model_name, api_key)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_tools(msg.get("tools", []))

# Human-in-the-loop confirmation card
if st.session_state.pending:
    slot = st.empty()
    with slot.container(border=True):
        st.warning("**Confirmation required.** The agent proposes this state-changing action:")
        st.json(st.session_state.pending["proposed_action"])
        c1, c2 = st.columns(2)
        confirm = c1.button("✅ Confirm escalation", type="primary", use_container_width=True)
        cancel = c2.button("❌ Cancel", use_container_width=True)
    if confirm or cancel:
        slot.empty()
        st.session_state.pending = None
        run_agent(graph, Command(resume={"confirmed": bool(confirm)}), permitted)
        st.rerun()

prompt = st.chat_input("Ask about a policy, PO, SKU or contract…", disabled=bool(st.session_state.pending))
prompt = prompt or st.session_state.pop("queued", None)
if prompt:
    st.session_state.turn_tools = []
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    run_agent(graph, {"messages": [("user", prompt)]}, permitted)
    if st.session_state.pending:
        st.rerun()
