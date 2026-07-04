import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(system_prompt, user_prompt, temperature=0, max_tokens=512):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(URL, headers=headers, json=payload)

    print(response.status_code)
    print(response.text)

    if response.status_code != 200:
        return None

    return response.json()["choices"][0]["message"]["content"]
if __name__ == "__main__":
    result = call_llm(
        "You are a helpful AI assistant.",
        "Say hello in one short sentence."
    )
    print(result)