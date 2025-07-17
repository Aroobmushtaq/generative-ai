
# Install required packages
import gradio as gr
import fitz
import docx
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import traceback

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(file.name)
            text = "\n".join([page.get_text() for page in doc])
            return text, "PDF"
        elif file.name.endswith(".docx"):
            d = docx.Document(file.name)
            text = "\n".join([para.text for para in d.paragraphs])
            return text, "DOCX"
        else:
            raise Exception("Unsupported file type.")
    except Exception as e:
        raise Exception(f"File reading error: {e}")

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def build_index(chunks):
    emb = model.encode(chunks)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb).astype("float32"))
    return index, emb

def retrieve_top_k(query, chunks, index, embeddings, k=3):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def ask_groq(api_key, prompt, model_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Groq Error {response.status_code}: {response.text}")
    return response.json()['choices'][0]['message']['content']

def process_file(file):
    try:
        text, file_type = extract_text(file)
        if not text.strip():
            return None, None, "❌ No text found in uploaded file.", ""
        chunks = chunk_text(text)
        index, emb = build_index(chunks)
        state = {"chunks": chunks, "index": index, "embeddings": emb, "file": file.name, "type": file_type}
        log = f"✅ File: {file.name} ({file_type})\n✅ Chunks: {len(chunks)}\n✅ Ready to answer questions."
        return state, log, "", ""
    except Exception as e:
        return None, "", f"❌ Error: {str(e)}", traceback.format_exc()

def answer_question(state, question, api_key, model_choice):
    try:
        if state is None:
            return "❌ Please upload a file first.", "No memory found."
        if not api_key.strip():
            return "❌ Please enter your Groq API key.", "Missing API key."
        chunks = state["chunks"]
        index = state["index"]
        emb = state["embeddings"]
        top_chunks = retrieve_top_k(question, chunks, index, emb)
        context = "\n\n".join(top_chunks)
        prompt = f"Use this context to answer:\n\n{context}\n\nQ: {question}"
        answer = ask_groq(api_key, prompt, model_choice)
        return f"### ✅ Answer:\n{answer}", "✅ Answer generated using previous file."
    except Exception as e:
        return f"❌ Error: {str(e)}", traceback.format_exc()

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 📚 Ask Multiple Questions on PDF/DOCX using RAG + Groq")
    state = gr.State()
    with gr.Row():
        file = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"])
        api_key = gr.Textbox(label="🔐 Groq API Key", type="password")
        load_btn = gr.Button("📂 Load File")
    logs = gr.Textbox(label="🛠️ File Load Logs", lines=8)
    with gr.Row():
        question = gr.Textbox(label="❓ Ask a Question")
        model_choice = gr.Dropdown(choices=["llama3-70b-8192", "mixtral-8x7b-32768"], value="llama3-70b-8192")
        ask_btn = gr.Button("🚀 Get Answer")
    output = gr.Markdown()
    debug = gr.Textbox(label="🧪 Debug Info", lines=8)
    load_btn.click(process_file, inputs=file, outputs=[state, logs, output, debug])
    ask_btn.click(answer_question, inputs=[state, question, api_key, model_choice], outputs=[output, debug])

demo.launch()
