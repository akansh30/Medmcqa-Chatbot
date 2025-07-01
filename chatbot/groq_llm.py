import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def refine_explanation(explanation: str, question: str) -> str:
    prompt = f"""
You are a helpful and precise medical assistant.

Your task is to clean and clarify the following medical explanation. Follow these rules:

1. Remove any initial references like "Ref.", "Refer", or "Reference" at the beginning of the text.
2. Remove any inline or parenthetical references like "(Ref. Cummings...)", "(Ref)", "(Reference XYZ)".
3. Do NOT remove any medically relevant information or concepts.
4. Return only the cleaned explanation — don't add extra info or formatting.
5. If the explanation is already clean, return it as-is.

Question: {question}

Original Explanation:
{explanation}

Cleaned Explanation:
""".strip()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(GROQ_URL, headers=headers, json=data)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"].strip()
    return explanation 
