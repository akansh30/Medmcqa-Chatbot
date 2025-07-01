# MedMCQA Chatbot

The chatbot is designed to answer **medical multiple-choice questions (MCQs)** with clear explanations i.e **grounded only in the dataset**. It uses **LangGraph** to define the flow, **FAISS** for fast semantic retrieval, and **LLM-powered explanation cleanup** via **Groq’s llama3**

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
4.Add Your Dataset
Download the [MedMCQA dataset](https://huggingface.co/datasets/openlifescienceai/medmcqa) and place it in data folder:
```bash
data/medmcqa_raw.json
```
5. Build FAISS Index
This script converts your dataset into a searchable format.
```bash
python faiss_ingest.py
```
It will create: `faiss_index/index.faiss` and `faiss_index/id_map.pkl`

6. LLM Setup
Sign up at <https://console.groq.com> and create an API key.
Create a `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
```
7. Run the Chatbot ( Streamlit Interface)
```bash
streamlit run chat_ui.py
```




