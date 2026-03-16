import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Callable
from .prompts import (
    SPELL_PROMPT,
    REWRITE_PROMPT,
    EXPAND_PROMPT,
    evaluate_prompt,
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


def llm_response_generator(
    query: str, docs: dict, prompt_generator_func: Callable[[str, dict], str]
) -> str:
    prompt = prompt_generator_func(query, docs)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return (response.text or "").strip().strip('"')


def llm_image_response_generator(
    img: bytes, system_prompt: str, mime: str, query: str
) -> tuple[str, int | None]:
    parts = [system_prompt, types.Part.from_bytes(data=img, mime_type=mime), query]
    response = client.models.generate_content(model=MODEL, contents=parts)
    content = (response.text or "").strip()
    total_token_count = (
        response.usage_metadata.total_token_count if response.usage_metadata else None
    )
    return content, total_token_count


def individual_rerank_score(query: str, doc: dict) -> float:
    text = llm_response_generator(query, doc, individual_rerank_prompt)
    try:
        score = float(text)
        return score
    except ValueError:
        return 0.0


def batch_rerank(query: str, docs: dict) -> list:
    text = llm_response_generator(query, docs, batch_rerank_prompt)
    ranked_score_list = json.loads(text)
    return ranked_score_list


def evaluate(query: str, docs: dict) -> list:
    text = llm_response_generator(query, docs, evaluate_prompt)
    scores_list = json.loads(text)
    return scores_list
