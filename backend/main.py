import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import shutil
import pdfplumber
import docx
import faiss
import numpy as np
import openai
from tempfile import NamedTemporaryFile

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "uploaded_docs"
VECTOR_DB_PATH = "faiss.index"
CHUNK_SIZE = 500  # characters
EMBED_DIM = 1536  # for OpenAI ada-002

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper: Parse document

def parse_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def parse_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Helper: Embedding

def get_embedding(text: str) -> np.ndarray:
    resp = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002",
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# Helper: FAISS

def load_faiss_index():
    if os.path.exists(VECTOR_DB_PATH):
        return faiss.read_index(VECTOR_DB_PATH)
    else:
        return faiss.IndexFlatL2(EMBED_DIM)


def save_faiss_index(index):
    faiss.write_index(index, VECTOR_DB_PATH)

# Upload endpoint
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Parse
    if ext == "pdf":
        text = parse_pdf(file_path)
    elif ext == "docx":
        text = parse_docx(file_path)
    elif ext == "txt":
        text = parse_txt(file_path)
    else:
        return JSONResponse({"error": "Unsupported file type."}, status_code=400)
    # Chunk
    chunks = chunk_text(text)
    # Embedding and FAISS
    index = load_faiss_index()
    meta = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        index.add(np.expand_dims(emb, 0))
        meta.append({"file": file.filename, "chunk_id": i, "text": chunk})
    save_faiss_index(index)
    # Save meta
    with open(os.path.join(UPLOAD_DIR, file.filename + ".meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"message": "File uploaded and processed.", "chunks": len(chunks)}

# Ask endpoint
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    # Embed question
    q_emb = get_embedding(question)
    # Load index
    index = load_faiss_index()
    if index.ntotal == 0:
        return JSONResponse({"error": "No documents uploaded yet."}, status_code=400)
    D, I = index.search(np.expand_dims(q_emb, 0), 5)
    # Gather context
    context_chunks = []
    citations = []
    for idx in I[0]:
        # Find which file/chunk this is
        for meta_file in os.listdir(UPLOAD_DIR):
            if meta_file.endswith(".meta.json"):
                with open(os.path.join(UPLOAD_DIR, meta_file), "r", encoding="utf-8") as f:
                    import json
                    meta = json.load(f)
                    if idx < len(meta):
                        chunk = meta[idx]
                        context_chunks.append(chunk["text"])
                        citations.append({"file": chunk["file"], "chunk_id": chunk["chunk_id"], "text": chunk["text"]})
                        break
    # Compose prompt
    prompt = f"Context:\n" + "\n---\n".join(context_chunks) + f"\n\nQuestion: {question}\nAnswer:"
    # Call LLM
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = resp.choices[0].message.content
    return {"answer": answer, "citations": citations}
