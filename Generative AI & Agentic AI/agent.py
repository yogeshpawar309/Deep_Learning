"""NexaChain Supply Operations agent"""

# Standard library
import ast
import json
import operator
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

# libraries
import pandas as pd
from docling.document_converter import DocumentConverter
from fastembed import SparseTextEmbedding

# LangChain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

import streamlit as st

ESCALATION_LOG = os.environ.get("ESCALATION_LOG", "escalation_log.json")

SYSTEM_PROMPT = """You are NexaChain's Supply Operations AI Assistant.

GROUNDING
- Use ONLY the supplied NexaChain data pack and the structured operational data exposed through tools.
- Do not invent policy, PO, account, inventory, contract, or case information.
- The current Procurement Policy is Version 5, effective June 1, 2026.
- Version 4 is deprecated and must not be used to resolve current requests.
  Search results flagged DEPRECATED must be ignored for current answers.
- Account-specific Master Supply Agreements or Vendor Agreements override the standard policy
  only where they explicitly state an override. For any PO fee/credit question: (1) look up the PO,
  (2) look up its account (Governing_Agreement, Override_Notes), (3) search the agreement/policy text,
  (4) apply the override if it covers the situation, otherwise standard Policy v5.
- If sources conflict or the available evidence is insufficient, explain the conflict/uncertainty
  and escalate rather than guessing.
- Cite the source document name for every policy or contract claim.

CALCULATIONS
- Use the calculate tool for every number. Show the working (rate, days/hours, PO value, cap).
- "Per full day late" means whole days only: use the _full_days_late_floor field, not the fractional Days_Late.
- Service credits require the delay to be carrier-fault (per the policy) and to exceed the threshold; check
  Delay_Reason and the threshold before computing anything.
- Use Hours_Until_Ship_At_Snapshot for cancellation timing and pick the correct band's boundaries.
- Never compute days-until-stockout without a daily usage figure. If the data has none, say so, and report
  on-hand vs reorder point and safety stock instead. Apply known-issue caveats (e.g. KI-089) and mention them.
- The Cases sheet Historical_Resolution_Notes are context only and may be wrong (some cite deprecated v4).
  Never treat them as ground truth.

TOOLS
- search_documents: policies, SOPs, agreements, known issues.
- lookup_supply_data: structured records - purchase_order, inventory, case, account (call once per identifier).
- calculate: arithmetic.
- request_escalation: propose an escalation to a human.

ACCESS CONTROL
- Never bypass tool-layer access controls. If a tool reports access denied, tell the user you cannot
  show that record under the current account, and do not reveal, confirm or infer anything about it.

WHEN TO ANSWER
- Answer directly when the evidence supports a confident answer.
- Escalate when the request needs human judgment, an unsupported exception, unclear policy eligibility
  (e.g. carrier-vs-buyer fault unclear), quality claims over $10,000, or an action outside your capabilities.

STATE-CHANGING ACTIONS
- Never claim an escalation exists unless the request_escalation tool result says it was created.
- Calling request_escalation only PROPOSES the action; the user is asked to confirm in the UI.
- Call request_escalation at most once per turn, on its own (not alongside other tools).
"""

@dataclass
class KnowledgeBase:
    store: object  # QdrantVectorStore (or anything with similarity_search)
    tables: dict  # sheet name -> DataFrame (structured data, reachable ONLY through the access-controlled tool)
    files: list
    n_chunks: int
    readme: str = ""
    snapshot: str = ""


converter = DocumentConverter()

def _read_pdf(path: Path) -> str:
    """Extract PDF text + tables as markdown using Docling."""
    result = converter.convert(str(path))
    return result.document.export_to_markdown()

def load_files(data_dir: Path):
    docs, tables = [], {}

    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            docs.append(
                Document(
                    page_content=_read_pdf(path),
                    metadata={"source": path.name}
                )
            )

        elif path.suffix.lower() in {".xlsx", ".xls"}:
            for sheet, df in pd.read_excel(path, sheet_name=None).items():
                tables[sheet] = df.dropna(how="all").dropna(axis=1, how="all")

    return docs, tables


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path="./qdrant/hybrid1")

def build_knowledge_base(data_dir: Path) -> KnowledgeBase:
     # 1. Load files
    docs, tables = load_files(data_dir)

    if not docs:
        raise FileNotFoundError(
            f"No PDF/XLSX files found in {data_dir}"
        )
    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=150)

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No document chunks were created.")

    # 3. Configure dense embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Configure sparse BM25 embeddings
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Create a Qdrant client for local storage
    client = get_qdrant_client()

    # 6.Create a collection with both dense and sparse vectors
    client.create_collection(
        collection_name="NexaChain_Supply_docs",
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )

    # 7. Create hybrid Qdrant vector store
    store = QdrantVectorStore(
        client=client,
        collection_name="NexaChain_Supply_docs",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    # 8. Add chunks to Qdrant
    docs = store.add_documents(documents = chunks)

    # 9. Return knowledge base
    return KnowledgeBase(
        store=store,
        tables=tables,
        files=list(data_dir.iterdir()),
        n_chunks=len(chunks),
        readme=None,
        snapshot=None,
    )

# --------------------------------------------------------------------------- #
# Structured-data lookup (with Account_ID access control)
# --------------------------------------------------------------------------- #
SHEET_FOR = {
    "purchase_order": "purchase",
    "inventory": "inventory",
    "case": "case",
    "account": "account",
}
LABELS = {"purchase_order": "Purchase order", "inventory": "SKU", "case": "Case", "account": "Account"}
CONTEXT_ONLY_COL = "Historical_Resolution_Notes"


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def _acct_col(df: pd.DataFrame):
    return next((c for c in df.columns if _norm(c) == "accountid"), None)

def list_accounts(tables: dict) -> list:
    """[(Account_ID, Account_Name)] from the Accounts sheet."""
    for df in tables.values():
        columns = df.columns
        id_col = "accountid" if "accountid" in columns else None
        name_col = "accountname" if "accountname" in columns else None
        if id_col is not None and name_col is not None:
            return [
                (str(i), str(n))
                for i, n in zip(df[id_col], df[name_col])
            ]
    return []


def _visible(df, mask, permitted):
    """Row-level access control. `permitted=None` = internal, unrestricted."""
    col = _acct_col(df)
    if permitted is None or col is None:
        return mask
    return mask & df[col].astype(str).isin(set(permitted))


def _derive(query_type: str, rec: dict) -> dict:
    """Add computed helper fields so the model does not have to guess."""
# Purchase order
    if query_type == "purchase_order":
        late = pd.to_numeric(rec.get("Days_Late"), errors="coerce")
        if pd.notna(late):
            rec["_full_days_late_floor"] = int(late // 1)
# Inventory
    if query_type == "inventory":
        qty = pd.to_numeric(rec.get("On_Hand_Qty"), errors="coerce")
        rp = pd.to_numeric(rec.get("Reorder_Point"), errors="coerce")
        ss = pd.to_numeric(rec.get("Safety_Stock"), errors="coerce")

# Reorder point calculations
        if pd.notna(qty) and pd.notna(rp):
            rec["_below_reorder_point"] = bool(qty < rp)
            rec["_shortfall_vs_reorder_point"] = int(rp - qty) if qty < rp else 0
       
# Safety stock calculation
        if pd.notna(qty) and pd.notna(ss):
            rec["_below_safety_stock"] = bool(qty < ss)

# KI-089 adjustment KI-089: reduce stock by 5%
        if "KI-089" in str(rec.get("Known_Issue_Flag", "")) and pd.notna(qty):
            adj = round(float(qty) * 0.95)
            rec["_adjusted_stock"] = adj
            
            if pd.notna(rp):
                rec["_below_reorder_point_after_KI-089_adjustment"] = bool(adj < rp)
        
# Estimate stockout days
        usage = next((k for k in rec if any(h in k.lower() for h in ("daily_usage", "usage", "consumption"))), None)
        
        if usage and pd.notna(pd.to_numeric(rec[usage], errors="coerce")) and float(rec[usage]) > 0 and pd.notna(qty):
            rec["_estimated_days_until_stockout"] = round(float(qty) / float(rec[usage]), 2)
        else:
            rec["_days_until_stockout"] = "CANNOT BE COMPUTED: the workbook has no daily-usage figure for this SKU"
# Mark context-only data
    if query_type == "case" and CONTEXT_ONLY_COL in rec:
        rec[CONTEXT_ONLY_COL + " (CONTEXT ONLY, may be wrong; not ground truth)"] = rec.pop(CONTEXT_ONLY_COL)
    return rec


def lookup(tables: dict, query_type: str, identifier: str, permitted=None) -> str:
# Check query type
    if query_type not in SHEET_FOR:
        return f"Unsupported query type. Use one of: {', '.join(SHEET_FOR)}."
    
# Normalize the identifier
    target = _norm(identifier)
    if not target:
        return "Empty identifier."
    
# Select preferred sheet
    preferred = [n for n in tables if SHEET_FOR[query_type] in n.lower()]
    ordered = preferred + [n for n in tables if n not in preferred]
    denied = False
# Search each sheet
    for name in ordered:
        df = tables[name]
        normed = df.map(_norm)

    # Exact match
        mask = (normed == target).any(axis=1)
    # Partial match if needed
        if not mask.any() and len(target) >= 4:  # e.g. account names: "Vertex Robotics"
            mask = normed.apply(lambda s: s.str.contains(target, regex=False)).any(axis=1)
        if not mask.any():
            continue
    # Check user permissions   
        visible = _visible(df, mask, permitted)
        if not visible.any():
            denied = True
            continue
        kind = next((q for q, frag in SHEET_FOR.items() if frag in name.lower()), query_type)
    # Get matching rows
        rows = [
            _derive(kind, {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.to_dict().items() if pd.notna(v)})
            for _, r in df.loc[visible].head(15).iterrows()
        ]
    # Return results as JSON    
        return json.dumps({"sheet": name, "matches": rows}, default=str, indent=2)

    if denied:
        return (
            f"ACCESS DENIED: no {LABELS[query_type].lower()} matching '{identifier}' is available under the "
            "current account's permissions. Do not reveal or infer anything about it."
        )
    # Nothing found
    return f"{LABELS[query_type]} '{identifier}' not found. Sheets searched: {', '.join(tables)}."


# --------------------------------------------------------------------------- #
#  calculator
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}
def _eval(node):
    # 1. Get the main expression
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    # 2. If it is a number, return it
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # 3. Calculate +, -, *, /, etc.
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left),_eval(node.right)        )

    # 4. Handle negative numbers
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))

    # 5. Reject anything else
    raise ValueError("Only basic math is allowed.")


# --------------------------------------------------------------------------- #
# Escalation (state-changing: only executed after explicit confirmation)
# --------------------------------------------------------------------------- #
def prepare_escalation(summary: str, reference: str = "", priority: str = "MEDIUM", requested_by: str = "internal") -> dict:
    """Build the proposal only. Nothing is persisted."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "reference": reference,
        "priority": priority,
        "requested_by": requested_by,
        "status": "awaiting_confirmation",
    }


def _read_log() -> list:
    if os.path.exists(ESCALATION_LOG):
        with open(ESCALATION_LOG) as f:
            return json.load(f)
    return []


def create_escalation(escalation: dict, confirmed: bool) -> dict:
    if not confirmed:
        raise ValueError("Explicit user confirmation is required.")
    record = {**escalation, "status": "pending"}
    log = _read_log()
    log.append(record)
    with open(ESCALATION_LOG, "w") as f:
        json.dump(log, f, indent=2)
    return record


def get_pending_escalations() -> list:
    return [e for e in _read_log() if e.get("status") == "pending"]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
# State — stores the conversation
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


TOOL_META = {
    "search_documents": ("🔍", "Document search"),
    "lookup_supply_data": ("🗄️", "Supply data lookup"),
    "calculate": ("🧮", "Calculator"),
    "request_escalation": ("🚨", "Escalation (needs confirmation)"),
}


def build_graph(kb: KnowledgeBase, llm):
    """Compile the LangGraph agent. `llm` is any chat model that supports bind_tools."""

    @tool
    def search_documents(query: str, config: RunnableConfig) -> str:
        """Search procurement policies (v5 current, v4 deprecated), the Returns SOP, the Warehouse guide
        and known issues, and account supply/vendor agreements. Use for any policy, contract, fee,
        credit, claim-window, eligibility or process question."""
        
        # get permitted accounts
        permitted = config.get("configurable", {}).get("permitted_accounts")
        
        # Search the similar Document
        results = kb.store.similarity_search(query, k=14)
       
        # Filter account-specific documents
        results = [
            d for d in results
            if permitted is None or not d.metadata.get("account_id") or d.metadata["account_id"] in permitted
        ][:6]

        # Handle no results
        if not results:
            return "No relevant passages found (some agreements may be restricted to their own account)."
        # Prepare the results
        out = []
        # Number each result
        for i, doc in enumerate(results, 1):
            flag = "  [DEPRECATED - superseded, do not use for current answers]" if doc.metadata.get("deprecated") else ""
            out.append(f"[{i}] Source: {doc.metadata.get('source', 'unknown')}{flag}\n{doc.page_content}")
        return "\n\n".join(out)

    @tool
    def lookup_supply_data(
        query_type: Literal["purchase_order", "inventory", "case", "account"],
        identifier: str,
        config: RunnableConfig,
    ) -> str:
        """Look up ONE structured record: purchase_order (e.g. PO-6301), inventory (a SKU such as
        SKU-MC-1050; returns on-hand vs reorder point and known-issue adjustments), case (e.g.
        CASE-9003) or account (an Account_ID such as ACC-1005, or an account name; returns the
        governing agreement and override notes). Call once per identifier. Results are limited to
        the signed-in account."""
       
        # Get the user's permitted accounts
        permitted = config.get("configurable", {}).get("permitted_accounts")
        
        # Call the actual lookup function
        return lookup(kb.tables, query_type, identifier, permitted)

    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression exactly, e.g. '48000 * 0.015 * 4'."""
        try:
            return str(_eval(ast.parse(expression, mode="eval")))
        except Exception as e:
            return f"Calculation error: {e}"

    @tool
    def request_escalation(summary: str, reference: str = "", priority: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM") -> str:
        """Propose escalating an issue to human supply operations (a case, PO or dispute).
        This only PROPOSES the action; the user must confirm before anything is created."""
        return "handled by graph"  # never called directly; see run_tools

    # Create a list of tools
    tools = [search_documents, lookup_supply_data, calculate, request_escalation]
    # Create a tool map
    tool_map = {t.name: t for t in tools}
    # Tool calling
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: State, config: RunnableConfig):
        # Get permitted accounts
        permitted = config.get("configurable", {}).get("permitted_accounts")
        # Determine the account scope
        scope = "internal staff (all accounts)" if permitted is None else f"signed in as account(s): {', '.join(permitted)} only"
        # Build the data context
        context = (
            "\n\nDATA CONTEXT\n"
            f"- Dataset snapshot time = 'now' for ALL time-based questions: {kb.snapshot or 'not provided'} "
            "(do not use the real-world date).\n"
            f"- Structured sheets available via lookup_supply_data: {', '.join(kb.tables) or 'none'}.\n"
            f"- Session scope: {scope}.\n"
            f"- Workbook README notes:\n{kb.readme}"
        )
        # call the LLM
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT + context)] + state["messages"])
        # Return the model response
        return {"messages": [response]}

    def route(state: State):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    def run_tools(state: State, config: RunnableConfig):
        results = []
        # Get the tool calls from the last message
        for call in state["messages"][-1].tool_calls:
        # Check whether it is an escalation request
            if call["name"] == "request_escalation":
            # Get permitted accounts
                permitted = config.get("configurable", {}).get("permitted_accounts")
            # Prepare the escalation proposal
                proposal = prepare_escalation(**call["args"], requested_by="internal" if permitted is None else ", ".join(permitted))
            # Pause and ask for confirmation
                decision = interrupt({"type": "confirmation_required", "proposed_action": proposal})
            # Check whether the user confirmed
                if isinstance(decision, dict) and decision.get("confirmed") is True:
                    rec = create_escalation(proposal, confirmed=True)
                    content = f"Escalation CREATED (id {rec['id']}, status {rec['status']})."
                else:
                    content = "Escalation NOT created: the user declined to confirm."
            # Handle normal tools
            elif call["name"] in tool_map:
                try:
                    content = str(tool_map[call["name"]].invoke(call["args"], config))
                except Exception as e:
                    content = f"Tool error: {e}"
            else:
                content = f"Unknown tool {call['name']}."
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

# Create the graph
    graph = StateGraph(State)
# Add the model node
    graph.add_node("model", call_model)
# Add the tools node
    graph.add_node("tools", run_tools)
# Connect START to the model
    graph.add_edge(START, "model")
# Add conditional routing from the model    
    graph.add_conditional_edges("model", route, 
                                {"tools": "tools", 
                                 END: END})
# Connect tools back to the model
    graph.add_edge("tools", "model")
# Compile the graph
    return graph.compile(checkpointer=MemorySaver())
