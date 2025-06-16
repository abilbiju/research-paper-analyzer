
# 📄 Research Paper Analyzer

Welcome to the **Research Paper Analyzer** — a user-friendly Streamlit web app designed to simplify reading and understanding academic research papers. Upload a research paper in PDF format, and get instant summaries, keyword extraction, Q&A, and even translation of key insights.

🔗 **Live App**: [Click here to try it out](https://research-paper-analyzer-jcdseen7wpnneoa5pvu5tk.streamlit.app/)  
🔗 **GitHub Repo**: [abilbiju/research-paper-analyzer](https://github.com/abilbiju/research-paper-analyzer)

---

## 🚀 Features

- 📥 **Upload PDF**: Drag and drop a research paper in PDF format.
- 🧠 **Automatic Summarization**: Extracts the core ideas using advanced LLMs.
- ❓ **Ask Questions**: Interact with the paper using natural language queries.
- 🔍 **Keyword Extraction**: Highlights key terms and concepts.
- 🌐 **Multilingual Support**: Translate content into multiple languages.
- 📊 **Insights Generator**: Generates structured insights for quick reference.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLMs**: LangChain + OpenAI / Gemini
- **Document Parsing**: PyMuPDF / PDFPlumber
- **Embeddings and Retrieval**: FAISS / ChromaDB
- **Translation (optional)**: Whisper / Google Translate API

---

## 📦 Installation (For Local Development)

```bash
git clone https://github.com/abilbiju/research-paper-analyzer.git
cd research-paper-analyzer
pip install -r requirements.txt
cd research-paper-analyzer/api
streamlit run og.py
```

---

## 🧪 How to Use

1. Open the [web app](https://research-paper-analyzer-jcdseen7wpnneoa5pvu5tk.streamlit.app/)
2. Upload a PDF research paper.
3. Wait for the content to load and be processed.
4. Ask questions, view summaries, or translate findings.

---

## 🧠 Use Cases

- Students summarizing lengthy academic papers
- Researchers cross-checking multiple sources
- Professionals quickly understanding technical documents

---

## 🤝 Contributions

Want to add new features like citation analysis, better question-answering, or more LLMs? PRs and ideas are always welcome!

---

## 📜 License

MIT License. Feel free to use, modify, and share.
