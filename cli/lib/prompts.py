SPELL_PROMPT = """Fix any spelling errors in the user-provided movie search query below. Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words. 
                Preserve punctuation and capitalization unless a change is required for a typo fix. If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else. User query: 
                """
REWRITE_PROMPT = """Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: 
"""

EXPAND_PROMPT = """Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: 
"""

IMAGE_PROMPT = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def individual_rerank_prompt(query: str, doc: dict) -> str:
    title = doc.get("title", "")
    description = doc.get("description", "")

    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def batch_rerank_prompt(query: str, doc: dict) -> str:
    doc_list = []
    for movie in doc.values():
        doc_list.append(
            f"{movie.get("id", "")},{movie.get("title", "")} - {movie.get("description", "")[:100]}..."
        )

    doc_list_str = "\n".join(doc_list)
    return f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


def evaluate_prompt(query: str, docs: dict) -> str:
    formatted_results = doc_summary(docs)
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{formatted_results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def augmented_generation_prompt(query: str, docs: dict) -> str:
    film_titles = doc_summary(docs)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{film_titles}

Provide a comprehensive answer that addresses the query:"""
    return prompt


def summarize_prompt(query: str, docs: dict) -> str:
    results = doc_summary(docs)
    return f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{results}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""


def citations_prompt(query: str, docs: dict) -> str:
    documents = doc_summary(docs)
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return prompt


def question_prompt(question: str, docs: dict) -> str:
    context = doc_summary(docs)
    return f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""


def doc_summary(docs: dict) -> str:
    formatted_results = []
    for film in docs.values():
        formatted_results.append(
            f"{film.get("title", "")} - {film.get("description", "")}..."
        )
    return chr(10).join(formatted_results)
