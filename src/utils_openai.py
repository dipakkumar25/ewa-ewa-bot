# src/utils_openai.py

import os
from dotenv import load_dotenv
from openai import OpenAI

from src.config import COLOR_MAP

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=API_KEY)


def gpt_answer(question: str, context: str):
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion:{question}"}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def gpt_risk_summary(context: str):
    prompt = f"""
Summarize SAP EWA risk based on the following context:

{context}

Include:
- Worst KPI changes (RED first)
- Why issues are important
- Recommended SAP fixes
- Keep < 200 words
"""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
