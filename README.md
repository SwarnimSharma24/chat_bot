# AI Chatbot (RAG with Next.js & FastAPI)

A hybrid-retrieval Retrieval-Augmented Generation (RAG) chatbot combining:
- **Frontend**: Next.js 16 (App Router, Tailwind CSS, Lucide Icons, Axios)
- **Backend**: FastAPI, FAISS, Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Local Ollama (`llama3.2:1b`)

---

## Prerequisites

1. **Python 3.10+**
2. **Node.js 18+ & npm**
3. **Ollama** installed: [ollama.com](https://ollama.com)
   - Pull the model:
     ```bash
     ollama pull llama3.2:1b
     ```

---

## Quick Setup & Start on Personal Laptop

### 1. Clone the Repository

```bash
git clone https://github.com/SwarnimSharma24/chatbot.git
cd chatbot
```

---

### 2. Start the Backend (FastAPI)

Open a new terminal window:

```bash
# 1. Navigate to backend directory
cd project

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# macOS / Linux:
# source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server (runs on port 3001)
uvicorn app:app --port 3001 --reload
```

> **Note:** The FAISS index (`documents.index`) and chunks (`chunks.json`) are pre-built and included. If you ever change `documents.json`, you can regenerate them by running `python rebuild_index.py`.

---

### 3. Start the Frontend (Next.js)

Open a second terminal window:

```bash
# 1. Navigate to frontend directory
cd frontend/chatbot

# 2. Install Node dependencies
npm install

# 3. Start the Next.js development server
npm run dev
```

The frontend will run at **`http://localhost:3000`**.

---

### 4. Ensure Ollama is Running

Make sure Ollama is active in the background or run:

```bash
ollama run llama3.2:1b
```

---

## Project Structure

```text
├── frontend/
│   └── chatbot/               # Next.js frontend application
│       ├── src/
│       │   ├── app/           # Next.js app routes & layouts
│       │   ├── components/    # Reusable UI components (Header, etc.)
│       │   ├── layout/        # Main chat interface
│       │   └── services/      # Axios API configuration & endpoints
│       ├── package.json       # Frontend dependencies
│       └── .env.example       # Example environment variables
│
├── project/                   # FastAPI backend application
│   ├── app.py                 # FastAPI endpoints & CORS configuration
│   ├── rag_context.py         # RAG pipeline orchestrator
│   ├── faiss_text.py          # Hybrid search & FAISS vector retrieval
│   ├── llm_service.py         # Ollama LLM integration & concurrency lock
│   ├── rebuild_index.py       # Rebuilds FAISS index from documents.json
│   ├── chunks.json            # Processed document chunks with metadata
│   ├── documents.json         # Raw knowledge base documents
│   ├── documents.index        # Precomputed FAISS index
│   └── requirements.txt       # Backend Python dependencies
│
├── .gitignore                 # Optimized root gitignore
├── requirements.txt           # Root requirements shortcut
└── README.md                  # This setup guide
```

