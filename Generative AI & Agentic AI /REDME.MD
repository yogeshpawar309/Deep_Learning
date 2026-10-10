# 📦 NexaChain Supply Operations AI Assistant

An AI-powered supply-chain assistant built with **LangGraph, LangChain, Groq, Qdrant, Streamlit, Docling, and Pandas**.

The system helps authorised users answer questions about **procurement policies, contracts, purchase orders, inventory, and supply-chain cases** using the provided NexaChain data pack.

## 🚀 Features

- 🔍 **Document Search** — Searches policies, SOPs, contracts, and warehouse documents.
- 🗄️ **Supply Data Lookup** — Retrieves authorised PO, inventory, case, and account information.
- 🧮 **Calculator** — Performs controlled calculations for fees, credits, and delays.
- 🔐 **Access Control** — Restricts account-specific data at the tool/data layer.
- 🤖 **LangGraph Agent** — Uses an LLM + tools workflow.
- 👤 **Human-in-the-Loop** — Requires explicit confirmation before creating an escalation.
- 📊 **Streamlit UI** — Interactive chat interface with visible tool usage.

## 🏗️ Architecture

```text
PDF / Excel Data
      ↓
Docling + Pandas
      ↓
Qdrant Hybrid Search + Structured Tables
      ↓
LangGraph Agent
      ↓
LLM ↔ Tools
      ↓
Answer / Escalation
      ↓
Human Confirmation
```

## 🛠️ Tools

| Tool | Purpose |
|---|---|
| `search_documents` | Search policies, SOPs, contracts |
| `lookup_supply_data` | Lookup PO, inventory, case, account |
| `calculate` | Perform arithmetic calculations |
| `request_escalation` | Propose escalation to human team |

## 👤 Human-in-the-Loop

State-changing actions are **not executed immediately**.

```text
User Request
     ↓
Agent prepares escalation
     ↓
Confirmation required
     ↓
 ┌───────────┐
 │ Confirm?  │
 └─────┬─────┘
    Yes │ No
       ↓   ↓
    Create  Cancel
```

The escalation is persisted only after the user explicitly confirms it.

## 🔐 Reliability

The assistant follows these rules:

- Uses only the supplied NexaChain data.
- Current **Procurement Policy v5** takes precedence over deprecated v4.
- Account-specific agreements can override standard policy when explicitly applicable.
- Restricted records are not revealed.
- Conflicting or insufficient evidence can result in escalation rather than guessing.
- Historical case resolutions are treated as context, not ground truth.

## 📁 Project Structure

```text
NexaChain/
├── agent.py
├── app.py
├── README.md
├── data/
│   ├── *.pdf
│   └── NexaChain_Assessment_Data.xlsx
└── escalation_log.json
```

## ⚙️ Installation

```bash
git clone <your-repository-url>
cd <repository-name>

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Set your Groq API key:

```text
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

## 💬 Example Queries

```text
What cancellation fee applies to PO-2044?
```

```text
What service credit does Vertex Robotics get for PO-6301?
```

```text
How is stock for SKU-MC-1050?
```

```text
Please escalate case CASE-9003 to supply operations.
```

## 🧰 Technology Stack

**Python · Streamlit · LangGraph · LangChain · Groq · Qdrant · Docling · Pandas · Hugging Face Embeddings · BM25**

## ⚠️ Note

This project is designed for the **NexaChain AI Agent assessment/data pack** and is a demonstration system rather than a production supply-chain platform.
