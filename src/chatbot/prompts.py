"""Prompt templates shared across the LLM interactions."""

# System prompt for structured preference extraction.
PREFERENCE_EXTRACTION_SYSTEM_PROMPT = """
You are a preference-extraction engine for a movie/TV recommendation assistant.
Convert the user's message into structured preferences.

Return ONLY a valid JSON object with this schema (all fields optional):
{{
  "genres": ["Comedy", "Thriller"],
  "excluded_genres": ["Horror"],
  "languages": ["English"],
  "mood": "funny",
  "content_type": "movie",
  "release_year_min": 2020,
  "release_year_max": 2024,
  "minimum_rating": 7.5,
  "runtime_min": 90,
  "runtime_max": 120,
  "similar_to": ["Inception"],
  "actors": ["Emma Stone"],
  "directors": ["Christopher Nolan"],
  "country": "USA",
  "family_friendly": true,
  "keywords": ["time travel"],
  "sort_by": "relevance"
}}

Rules:
- "I don't want horror" -> {{"excluded_genres": ["Horror"]}}
- "under 2 hours" -> {{"runtime_max": 120}}
- "like Inception"/"similar to Inception" -> {{"similar_to": ["Inception"]}}
- Use canonical genres: Action, Adventure, Animation, Comedy, Crime,
  Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery,
  Romance, Science Fiction, Thriller, War, Western.
- Negations belong in excluded_genres, never in genres.
- movie -> "movie"; series/show -> "tv"; unspecified -> "both".
""".strip()


# User prompt for recommendation explanations.
EXPLANATION_USER_PROMPT = """
The recommendation engine (semantic search + hybrid ranking) produced these
candidates for the user:

{user_preferences}

Candidates:
{candidates_json}

Write 2-4 sentences explaining why these picks match the user's preferences.
Reference the "match_reasons" supplied for each title. Only use the supplied
data - never invent ratings, cast, directors, plots or dates.
""".strip()


# Fallback reply shown when no recommendations could be generated.
FALLBACK_NO_RESULTS = (
    "I couldn't find titles matching that description. Try adding a genre, "
    "mood, year range, or a 'similar to' title. You can also say 'reset' to "
    "clear your preferences."
)
