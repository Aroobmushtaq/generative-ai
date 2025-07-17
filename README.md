# 📄 RAG PDF/DOCX QA App with Groq API + Gradio UI
<br>
This is a simple Retrieval-Augmented Generation (RAG) application built in Google Colab. It allows users to:

Upload PDF or DOCX documents

Ask questions about the document content

Get accurate answers using Groq's LLaMA 3 models

Use Gradio for a clean and interactive interface

## 💻 Features
✅ Upload .pdf or .docx files
<br>
✅ Automatically extract text content
<br>
✅ Embed document with FAISS vector search
<br>
✅ Ask questions in natural language
<br>
✅ Powered by Groq's LLaMA3-8B/70B
<br>
✅ Simple and secure UI with API key input

## 🚀 How to Run
<br>
🧠 Note: This project runs in Google Colab, no local installation needed.
<br>
Open this notebook in Google Colab

Run the first cell to install dependencies:
```bash
!pip install -q gradio faiss-cpu sentence-transformers openai pymupdf python-docx
```

Upload your PDF or DOCX file in the UI.

Paste your Groq API key (get it from https://console.groq.com/keys)

Select a model (llama3-8b-8192 or llama3-70b-8192)

Ask a question about the uploaded document.

Get the answer instantly in the chat area.

