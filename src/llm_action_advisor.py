from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_action_recommendation(kpi, prev, curr, change, root_cause):
    prompt = f"""
KPI: {kpi}
Previous Status: {prev}
Current Status: {curr}
Change: {change}
Root Cause: {root_cause}

Suggest corrective actions in 2-3 bullet points.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an SAP technical advisor. Be concise and factual."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
