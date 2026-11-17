# DocuChat 🤖

### AI-Powered PDF Question Answering System using RAG

DocuChat is an AI-powered document assistant that allows users to upload PDF files and ask questions about their content using natural language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate contextual answers using a Large Language Model.

## ✨ Features

- 📄 PDF document upload
- 🔍 Automatic text extraction
- 🧩 Intelligent text chunking
- 🧠 Text embeddings and semantic retrieval
- 💬 Natural-language question answering
- ⚡ LLM integration using Groq API
- 🎨 Modern chat-based user interface
- 🌙 Light and dark theme support

## 🛠️ Tech Stack

**Frontend:** HTML, CSS, JavaScript

**Backend:** Python, FastAPI, Uvicorn

**AI/ML:** Retrieval-Augmented Generation (RAG), Text Embeddings, Semantic Search

**PDF Processing:** PyMuPDF

**LLM API:** Groq

## 🏗️ Architecture

1. User uploads a PDF document.
2. The backend extracts text using PyMuPDF.
3. Extracted text is divided into smaller chunks.
4. Embeddings are generated for the chunks.
5. Relevant chunks are retrieved based on the user's question.
6. Retrieved context is passed to the LLM through Groq.
7. The generated answer is returned and displayed in the chat interface.

## 📁 Project Structure

```text
DocuChat/
├── backend/
│   ├── main.py
│   ├── embeddings.py
│   ├── requirements.txt
│   └── data/
│       └── uploads/
│
├── frontend/
│   ├── index.html
│   └── index_old.html
│
├── .gitignore
└── README.md
```

*Note: Project structure may vary as development continues.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Karan-malhotra1926/DocuChat.git
cd DocuChat
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file inside the `backend` folder:

```env
GROQ_API_KEY=your_groq_api_key
```

Never share or commit your API key.

### 6. Run the backend

```bash
python -m uvicorn main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

### 7. Run the frontend

Serve the `frontend` folder locally on port 5500 and open:

`http://localhost:5500`

## 🚀 Future Enhancements

- Multiple PDF document support
- Source citations with page numbers
- Chat history
- Hybrid search (BM25 + Vector Search)
- Reranking for improved retrieval accuracy
- RAG evaluation system
- Cloud deployment

## 👨‍💻 Author

**Karan Malhotra**

B.Tech Computer Science Engineering (AI & ML)

GitHub: [Karan-malhotra1926](https://github.com/Karan-malhotra1926)

---

⭐ If you find this project interesting, consider giving it a star!