# 📚 Study Assistant RAG Chatbot

A simple AI chatbot that answers questions from custom study notes using **Python, LangChain, OpenAI, and Pinecone**.

## ✨ Features
- Reads notes from `dbms.pdf`
- Splits text into chunks
- Stores embeddings in Pinecone
- Retrieves relevant context
- Answers only from provided notes

## 🛠️ Tech Stack
Python, LangChain, OpenAI, Pinecone, dotenv

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## 🔑 .env File

```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
```

## 💬 Example
`you: What is DBMS?`  
`bot: DBMS is a software system used to manage data...`

## 👨‍💻 Author
Ajit Gaud
