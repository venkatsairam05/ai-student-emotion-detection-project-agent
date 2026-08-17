"""Rule-based preference extraction.

Used as the primary fallback when Claude is unavailable, and as a
complement/recovery layer when the LLM returns invalid JSON. Keeps the
system functional in offline/demo mode.
"""

from __future__ import annotations

import re
from typing import Optional

from src.models.preferences import KNOWN_GENRES, CONTENT_TYPES, UserPreferences

# Genre aliases -> canonical genre names.
_GENRE_ALIASES: dict[str, str] = {
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "sci fi": "Science Fiction",
    "sci-fi movie": "Science Fiction",
    "sci-fi series": "Science Fiction",
    "space": "Science Fiction",
    "rom-com": "Romance",
    "romantic comedy": "Romance",
    "romance": "Romance",
    "horror": "Horror",
    "thriller": "Thriller",
    "mystery": "Mystery",
    "comedy": "Comedy",
    "funny": "Comedy",
    "humorous": "Comedy",
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "animated": "Animation",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "family": "Family",
    "fantasy": "Fantasy",
    "history": "History",
    "war": "War",
    "western": "Western",
    "music": "Music",
    "musical": "Music",
}

# Keyword/entity patterns mapped onto preferences.
_MOOD_PATTERNS: list[tuple[str, str]] = [
    ("mind-bending", "mind-bending"),
    ("mind bending", "mind-bending"),
    ("dark", "dark"),
    ("gritty", "dark"),
    ("uplifting", "uplifting"),
    ("feel-good", "feel-good"),
    ("feel good", "feel-good"),
    ("heartwarming", "heartwarming"),
    ("funny", "funny"),
    ("hilarious", "funny"),
    ("scary", "scary"),
    ("less serious", "light"),
    ("light-hearted", "light"),
    ("lighthearted", "light"),
    ("serious", "serious"),
    ("inspiring", "inspiring"),
    ("thrilling", "thrilling"),
    ("suspenseful", "suspenseful"),
    ("emotional", "emotional"),
    ("binge-worthy", "binge-worthy"),
]

# Explicit negation markers.
_NEGATION_RE = re.compile(
    r"\b(?:not?|no|don'?t|do not|doesn'?t|does not|avoid|skip|never|without|"
    r"can'?t stand|hate|hates|dislike|dislikes)\b",
    re.IGNORECASE,
)

_SIMILAR_RE = re.compile(
    r"\b(?:similar to|something like|anything like|kind of like|same as|"
    r"comparable to|recommend.*?like|movies? like|shows? like|series? like|"
    r"such as|like)\b\s*[:\-]?\s*['\"]?"
    r"([^,.;!?]+?)"
    r"(?=\b(?:but|and|under|after|before|less|more|with|without|while|though|"
    r"that|or|yet|released|from|no|not|for|in|at|by|too|very|so|than|it)\b|$)",
    re.IGNORECASE,
)

# Words to strip off the end of a captured similar-title.
_TRAILING_MEDIA_WORDS = (
    " movies", " movie", " shows", " show", " films", " film",
    " series", " dramas", " drama", " anime", " something",
)

_POSITIVE_WISH_RE = re.compile(
    r"\b(?:i (?:want|wanna|need|would like|like|love|enjoy|prefer)|"
    r"recommend|suggest|looking for|searching for|gimme|give me|"
    r"looking to watch|want to watch)\b",
    re.IGNORECASE,
)


def _token_is_negated(text: str, span_start: int, span_end: int) -> bool:
    """True when a token span is preceded by a negation marker."""
    before = text[max(0, span_start - 80):span_start]
    last = before.split()[-4:] if before.split() else []
    return any(_NEGATION_RE.fullmatch(tok.strip(".,;:!?")) for tok in last)


def _extract_genres(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    for alias, canonical in _GENRE_ALIASES.items():
        for match in re.finditer(rf"\b{re.escape(alias)}\b", lowered):
            span = match.span()
            if _token_is_negated(lowered, span[0], span[1]):
                if canonical not in prefs.excluded_genres:
                    prefs.excluded_genres.append(canonical)
            elif canonical not in prefs.genres:
                prefs.genres.append(canonical)


def _extract_content_type(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    has_movie = bool(re.search(r"\b(movie|film|movies|films)\b", lowered))
    has_tv = bool(
        re.search(r"\b(tv|series|show|shows|drama|dramas|k-drama|korean drama|anime)\b", lowered)
    )
    if has_movie and not has_tv:
        prefs.content_type = "movie"
    elif has_tv and not has_movie:
        prefs.content_type = "tv"
    else:
        prefs.content_type = "both"


def _extract_language(prefs: UserPreferences, text: str) -> None:
    mapping = {
        "english": "English",
        "english-language": "English",
        "hollywood": "English",
        "korean": "Korean",
        "k-drama": "Korean",
        "spanish": "Spanish",
        "french": "French",
        "japanese": "Japanese",
        "japanese anime": "Japanese",
        "anime": "Japanese",
        "hindi": "Hindi",
        "bollywood": "Hindi",
        "tamil": "Tamil",
        "telugu": "Telugu",
    }
    lowered = text.lower()
    for token, canonical in mapping.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            if not _token_is_negated(
                lowered, *re.search(rf"\b{re.escape(token)}\b", lowered).span()
            ):
                if canonical not in prefs.languages:
                    prefs.languages.append(canonical)


def _extract_year_range(prefs: UserPreferences, text: str) -> None:
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    years = [int(y) for y in years]
    if years:
        prefs.release_year_min = min(years)
        prefs.release_year_max = max(years)
    after = re.search(r"\bafter\s+(\d{4})\b", text, re.IGNORECASE)
    if after:
        prefs.release_year_min = int(after.group(1))
        if prefs.release_year_max is None or prefs.release_year_max < prefs.release_year_min:
            prefs.release_year_max = None
    before = re.search(r"\bbefore\s+(\d{4})\b", text, re.IGNORECASE)
    if before:
        prefs.release_year_max = int(before.group(1))


def _extract_runtime(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    if _token_is_negated(lowered, *((len(lowered) - 5, len(lowered)))) and "long" in lowered:
        pass
    under_match = re.search(r"\bunder\s+([\d.]+)\s*(hours?|hrs?|h)\b", lowered)
    if under_match:
        value = float(under_match.group(1))
        unit = under_match.group(2)
        minutes = value * 60 if unit.startswith(("h", "hour")) else value
        prefs.runtime_max = int(minutes)
    less_than = re.search(r"\bless\s+than\s+([\d.]+)\s*(hours?|hrs?|h|minutes?|min)\b", lowered)
    if less_than:
        value = float(less_than.group(1))
        unit = less_than.group(2)
        minutes = value * 60 if unit.startswith("h") else value
        prefs.runtime_max = int(minutes)
    over_match = re.search(r"\b(?:over|more\s+than|above)\s+([\d.]+)\s*(hours?|hrs?|h|minutes?|min)\b", lowered)
    if over_match:
        value = float(over_match.group(1))
        unit = over_match.group(2)
        minutes = value * 60 if unit.startswith("h") else value
        prefs.runtime_min = int(minutes)


def _extract_rating(prefs: UserPreferences, text: str) -> None:
    matches = re.findall(
        r"\b(?:rating|rated|rating of|imdb)\s*[>=]?\s*([\d.]+)\b",
        text,
        re.IGNORECASE,
    )
    values = [float(v) for v in matches if 0 < float(v) <= 10]
    if values:
        prefs.minimum_rating = max(prefs.minimum_rating, max(values))


def _extract_similar(prefs: UserPreferences, text: str) -> None:
    for match in _SIMILAR_RE.finditer(text):
        title = match.group(1).strip(" '\".!?,")
        if not title:
            continue
        lowered_title = title.lower()
        if lowered_title.startswith("i would like") or lowered_title.startswith("i like"):
            continue
        for suffix in _TRAILING_MEDIA_WORDS:
            if lowered_title.endswith(suffix):
                title = title[: -len(suffix)].strip()
                lowered_title = title.lower()
                break
        if title and lowered_title not in {
            "it", "this", "that", "those", "the", "that show", "that movie",
            "that one", "something", "anything", "a movie", "a show", "this movie",
            "this show", "these", "them", "anime", "movies", "shows", "it but",
            "it and",
        } and len(title) > 1:
            if title not in prefs.similar_to:
                prefs.similar_to.append(title)


def _extract_people(prefs: UserPreferences, text: str) -> None:
    actor_match = re.search(
        r"\b(?:starring|starring |featuring|with|acted by|played by|"
        r"starring\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        text,
    )
    if actor_match and actor_match.group(1) not in prefs.actors:
        prefs.actors.append(actor_match.group(1))
    director_match = re.search(
        r"\b(?:by|directed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        text,
    )
    if director_match and "directed by" in text.lower():
        if director_match.group(1) not in prefs.directors:
            prefs.directors.append(director_match.group(1))


def _extract_mood(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    for pattern, mood in _MOOD_PATTERNS:
        if pattern in lowered and not prefs.mood:
            prefs.mood = mood
            break


def _extract_family_friendly(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    if re.search(r"\b(?:family|kids|children|family-friendly|all ages)\b", lowered):
        prefs.family_friendly = True


def _extract_country(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    mapping = {
        "korean": "South Korea",
        "k-drama": "South Korea",
        "british": "United Kingdom",
        "uk": "United Kingdom",
        "india": "India",
        "bollywood": "India",
        "japanese": "Japan",
        "french": "France",
        "spanish": "Spain",
        "american": "United States",
        "hollywood": "United States",
    }
    for token, country in mapping.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            if not _token_is_negated(
                lowered, *re.search(rf"\b{re.escape(token)}\b", lowered).span()
            ):
                prefs.country = country
                return


def _extract_keywords(prefs: UserPreferences, text: str) -> None:
    lowered = text.lower()
    keywords = [
        "time travel", "heist", "zombie", "superhero", "apocalypse", "space",
        "detective", "courtroom", "high school", "travel", "survival",
        "murder mystery", "biopic", "noir", "coming of age",
    ]
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            if keyword not in prefs.keywords:
                prefs.keywords.append(keyword)


def extract_preferences_rule_based(text: str) -> UserPreferences:
    """Extract preferences from ``text`` using regex/heuristic rules."""
    prefs = UserPreferences()
    _extract_genres(prefs, text)
    _extract_content_type(prefs, text)
    _extract_language(prefs, text)
    _extract_year_range(prefs, text)
    _extract_runtime(prefs, text)
    _extract_rating(prefs, text)
    _extract_similar(prefs, text)
    _extract_people(prefs, text)
    _extract_mood(prefs, text)
    _extract_family_friendly(prefs, text)
    _extract_country(prefs, text)
    _extract_keywords(prefs, text)
    return prefs


def to_canonical_genre(name: str) -> Optional[str]:
    """Map an arbitrary genre label to the canonical vocabulary."""
    cleaned = name.strip().title()
    if cleaned in KNOWN_GENRES:
        return cleaned
    return _GENRE_ALIASES.get(name.strip().lower())
