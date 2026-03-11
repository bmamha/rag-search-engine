import os
import json
from dotenv import load_dotenv
from google import genai
from .prompts import (
    SPELL_PROMPT,
    REWRITE_PROMPT,
    EXPAND_PROMPT,
    individual_rerank_prompt,
    batch_rerank_prompt,
)

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=API_KEY)
MODEL = "gemma-3-27b-it"


def enhance(query: str, method: str | None) -> str:
    prompt = enhance_prompt_selector(query, method)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def enhance_prompt_selector(query, method: str | None) -> str:
    match method:
        case "spell":
            return f'{SPELL_PROMPT} "{query}"'
        case "rewrite":
            return f'{REWRITE_PROMPT} "{query}"'
        case "expand":
            return f'{EXPAND_PROMPT} "{query}"'
        case _:
            return query


def individual_rerank_score(query: str, doc: dict) -> float:
    prompt = individual_rerank_prompt(query, doc)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    try:
        score = float(response.text.strip())
        return score
    except ValueError:
        return 0.0


def batch_rerank(query: str, doc: dict) -> list:
    prompt = batch_rerank_prompt(query, doc)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    ranked_score_list = json.loads(response.text.strip())
    return ranked_score_list
