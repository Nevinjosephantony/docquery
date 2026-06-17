# llm.py

import requests
from config import GROQ_API_KEY, LLM_MODEL, GROQ_ENDPOINT

def call_llm(prompt, temperature=0.3):
    response = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]