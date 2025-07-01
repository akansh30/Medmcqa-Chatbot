# MedMCQA Chatbot

The chatbot is designed to answer **medical multiple-choice questions (MCQs)** with clear explanations that is **grounded only in the dataset**. It uses **LangGraph** to define the flow, **FAISS** for fast semantic retrieval, and **LLM explanation cleanup** via **Groq’s llama3**

### Working Chatbot Interface
![Screenshot 2025-07-01 202649](https://github.com/user-attachments/assets/0e9decdf-3a37-4d10-9c42-f24aecb22cd1)

### Chatbot Declining Out-of-Dataset Queries
![Screenshot 2025-07-01 200059](https://github.com/user-attachments/assets/bdb6d227-21bb-40af-a0a4-7b188c1d4d0a)

## Project Structure
```bash
Medmcqa-Chatbot/
│
├── chatbot/
│   ├── flow.py             # LangGraph conversational flow
│   ├── utils.py            # FAISS semantic search logic
│   ├── groq_llm.py         # LLM prompt for explanation refinement
│   └── cli_chatbot.py      # CLI mode for testing
│
├── data/                   # Save the Raw MedMCQA data here
├── faiss_index/            # Generated FAISS index files
│
├── chat_ui.py              # Streamlit frontend UI
├── faiss_ingest.py         # FAISS index creation from raw data
├── verify_faiss.py         # Test script to verify retrieval logic
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup Instructions

### 1.Clone the Repository

```bash
git clone https://github.com/akansh30/Medmcqa-Chatbot.git
cd Medmcqa-Chatbot
```
### 2.Create & Activate venv
```bash
python -m venv venv
source venv\Scripts\activate
```
### 3.Install Dependencies
```bash
pip install -r requirements.txt
```
### 4.Add Your Dataset
Download the [MedMCQA dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa) and place it in data folder:
```bash
data/medmcqa_raw.json
```
### 5.Build FAISS Index
This script converts your dataset into a searchable format.
```bash
python faiss_ingest.py
```
It will create: `faiss_index/index.faiss` and `faiss_index/id_map.pkl`

### 6.LLM Setup
Sign up at <https://console.groq.com> and create an API key.
Create a `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
```
### 7.Run the Chatbot (Streamlit Interface)
```bash
streamlit run chat_ui.py
```
## Justification of Design Choices:

### LangGraph structure:
                       `START → retrieve → route → (respond or fallback) → END`
The LangGraph structure in this chatbot is designed to keep the conversation grounded strictly in the MedMCQA dataset. It follows a clear three-step flow: the `retrieve` node uses FAISS to search for the most relevant question, then a `route` function checks if a result exists and decides whether to answer (respond) or gracefully decline (`fallback`). This ensures the bot only responds when confident and avoids hallucinating answers outside the dataset.

The `respond` node formats the output with the question, options, correct answer, and a cleaned explanation using the Llama3 model via Groq. Explanation cleanup is offloaded to the LLM, while all factual content remains dataset-based. This structure ensures safety, readability and easy extensibility.

### LLM choice:
The LLM used in this project is `llama3-70b-8192`, accessed via Groq’s high-speed API. I chose this model because it offers a **strong balance between performance and cost**, and Groq provides **fast inference** for free, making it ideal for real-time chatbot applications. In this chatbot, the LLM is used only to refine the explanation field from the Medmcqa dataset. It cleans up noisy text like "Ref." or "Reference" improving readability while preserving the medical accuracy already present in the dataset. This keeps the chatbot grounded and accurate.

### FAISS Vector database:
I chose FAISS as the vector database because it's lightweight, fast and works well for local setups. While I initially explored using Weaviate for its advanced features but it consumed too much memory for the large MedMCQA dataset. FAISS allowed me to efficiently store and search over 180,000 medical question embeddings using minimal resources. It pairs well with a simple metadata map, making retrieval both accurate and fast without the overhead of a full database server.

### Hallucination Prevention Strategy
To make sure the chatbot only answers questions grounded in the MedMCQA dataset, I implemented multiple layers of filtering in the search logic. First, I use exact `string normalization and matching` to catch any direct question hits. If there's no exact match, the system moves to `semantic search` using sentence embeddings from the `BAAI/bge-small-en model`. But even then, I added a `score threshold` of **0.88** to ensure only highly similar vector matches are considered. On top of that, I included a `Jaccard similarity `check based on word overlap between the query and candidate questions, requiring at least** 25 percent** overlap.This makes my chatbot hallucination-free and ensures it responds strictly based on the MedMCQA dataset.




