"""Reusable PRIYA UI components.

Pure presentation helpers (no chat/recommendation logic lives here). They
render branded HTML, chip buttons and movie cards, and return simple
"interaction" values (which chip/button was pressed) so callers can react.
"""

from __future__ import annotations

from html import escape
from typing import Any, Optional

import streamlit as st

from src.ui import theme
from src.utils.logging_config import get_logger

logger = get_logger("ui.components")

# --------------------------------------------------------------------- #
# Navigation / footer
# --------------------------------------------------------------------- #
PAGES = {
    "home": "Home",
    "discover": "Discover",
    "picks": "My Picks",
    "about": "About",
}


def render_navbar(current_page: str, llm_ready: bool = False, tmdb_ready: bool = False) -> None:
    """Sticky glass navigation bar (pure HTML, links switch pages)."""
    avatar = theme.priya_avatar_svg(40)
    links = "".join(
        f'<a class="priya-navlink{" active" if page == current_page else ""}" '
        f'href="?page={page}">{label}</a>'
        for page, label in PAGES.items()
    )
    status = "AI online" if (llm_ready or tmdb_ready) else "local mode"
    st.markdown(
        f"""
        <div class="priya-navbar">
          <div class="priya-nav-brand">
            <img src="{avatar}" width="40" height="40" alt="Priya avatar"/>
            <div class="priya-nav-title">
              <div class="name">🎬 PRIYA</div>
              <div class="tag">AI Movie Assistant</div>
            </div>
          </div>
          <nav class="priya-nav-links">{links}</nav>
          <div class="priya-status"><span class="dot"></span>Priya is online · {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="priya-footer">
          <div class="name">🎬 Priya AI</div>
          <div style="margin:.3rem 0">Your personal AI companion for discovering movies and TV shows.</div>
          <div style="margin-bottom:.5rem">Powered by AI • TMDB • Semantic Search</div>
          <div>
            <a href="?page=about">About</a> · <a href="?page=picks">My Picks</a> · <a href="?page=home">Home</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="priya-section-title">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Hero + chips
# --------------------------------------------------------------------- #
def render_hero() -> None:
    avatar = theme.priya_avatar_svg(118)
    st.markdown(
        f"""
        <div class="priya-hero">
          <div class="priya-hero-avatar"><img src="{avatar}" width="118" height="118" alt="Priya"/></div>
          <div class="priya-hero-eyebrow">✨ Your Personal AI</div>
          <div class="priya-hero-title">What are we watching tonight?</div>
          <div class="priya-hero-sub">
            Tell Priya your mood, favorite movies, or what you're in the mood for.<br/>
            She'll find something you'll love.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip_buttons(labels: list[str], key_prefix: str, per_row: int = 4) -> Optional[str]:
    """Render pill buttons; returns the label that was clicked (or None)."""
    pressed: Optional[str] = None
    for start in range(0, len(labels), per_row):
        row = labels[start:start + per_row]
        cols = st.columns(len(row))
        for col, label in zip(cols, row):
            if col.button(label, key=f"{key_prefix}_{start}_{label}", use_container_width=True):
                pressed = label
    return pressed


# --------------------------------------------------------------------- #
# Movie card data normalization
# --------------------------------------------------------------------- #
def _tmdb_href(content_type: str, tmdb_id: Any) -> str:
    """Official TMDB page URL for a card ('' when we have no id)."""
    if not tmdb_id:
        return ""
    section = "tv" if content_type == "tv" else "movie"
    return f"https://www.themoviedb.org/{section}/{tmdb_id}"


def movie_to_card(movie: Any, tmdb) -> dict:
    """Convert an engine :class:`Movie` into a UI card dict."""
    poster = tmdb.poster_url(movie.poster_path) if movie.poster_path and tmdb else ""
    backdrop = tmdb.backdrop_url(movie.backdrop_path) if movie.backdrop_path and tmdb else ""
    match_pct = round(movie.final_score * 100) if movie.final_score else None
    return {
        "title": movie.title,
        "overview": movie.overview,
        "rating": movie.rating,
        "year": movie.display_year,
        "genres": movie.genres,
        "poster": poster,
        "backdrop": backdrop,
        "runtime": movie.runtime,
        "content_type": movie.content_type,
        "director": movie.director,
        "cast": movie.cast,
        "trailer_key": movie.trailer_key,
        "tmdb_id": movie.tmdb_id,
        "tmdb_url": _tmdb_href(movie.content_type, movie.tmdb_id),
        "media": "tv" if movie.content_type == "tv" else "movie",
        "match_pct": match_pct,
        "reasons": movie.match_reasons,
    }


def tmdb_item_to_card(item: dict, tmdb, media: str = "movie") -> dict:
    """Convert a TMDB result dict into a UI card dict."""
    poster_path = item.get("poster_path") or ""
    backdrop_path = item.get("backdrop_path") or ""
    title = item.get("title") or item.get("name") or "Untitled"
    genres = tmdb.genre_names(item.get("genre_ids"), media) if tmdb and hasattr(tmdb, "genre_names") else []
    return {
        "title": title,
        "overview": item.get("overview", "") or "",
        "rating": float(item.get("vote_average") or 0.0),
        "year": (item.get("release_date") or item.get("first_air_date") or "")[:4] or "N/A",
        "genres": genres,
        "poster": tmdb.poster_url(poster_path) if tmdb else "",
        "backdrop": tmdb.backdrop_url(backdrop_path) if tmdb else "",
        "runtime": None,
        "content_type": media,
        "director": "",
        "cast": [],
        "trailer_key": "",
        "tmdb_id": item.get("id"),
        "tmdb_url": _tmdb_href(media, item.get("id")),
        "media": media,
        "match_pct": None,
        "reasons": [],
    }


def _poster_html(card: dict) -> str:
    poster = card.get("poster") or ""
    title = escape(card["title"])
    if poster:
        inner = f'<img src="{poster}" alt="{title} poster" loading="lazy"/>'
    else:
        inner = f'<div class="ph">{title}</div>'
    badge = ""
    if card.get("match_pct") is not None:
        badge = f'<span class="priya-match-badge"><span class="spark">✨</span> {card["match_pct"]}% AI Match</span>'
    elif card.get("rating"):
        badge = f'<span class="priya-match-badge">⭐ {card["rating"]:.1f}</span>'
    type_label = "📺" if card.get("content_type") == "tv" else "🎬"
    return (
        f'<div class="priya-mcard-poster">{inner}{badge}'
        f'<span class="priya-mcard-type">{type_label}</span></div>'
    )


def movie_card_html(card: dict) -> str:
    """Premium poster-card HTML (pure markup, no widgets).

    Shows the real TMDB poster, rating, genres, year, runtime (or a
    ``📺 TV Series`` label for shows without one), an overview snippet, the AI
    match score and reasons for recommendations, plus official Trailer / TMDB
    links — only when that data actually exists.
    """
    title = escape(card["title"])

    meta: list[str] = []
    if card.get("rating"):
        meta.append(f'<span class="m">⭐ {card["rating"]:.1f}</span>')
    genres = [escape(g) for g in (card.get("genres") or [])]
    if genres:
        meta.append(f'<span class="m">🎭 {", ".join(genres)}</span>')
    year = card.get("year") or ""
    if year and year != "N/A":
        meta.append(f'<span class="m">📅 {year}</span>')
    runtime = card.get("runtime")
    if runtime:
        hours, minutes = divmod(int(runtime), 60)
        runtime_label = f"{hours}h {minutes}m" if hours else f"{minutes}m"
        meta.append(f'<span class="m">⏱ {runtime_label}</span>')
    elif card.get("content_type") == "tv":
        meta.append('<span class="m">📺 TV Series</span>')
    meta_html = f'<div class="priya-mcard-meta">{" • ".join(meta)}</div>' if meta else ""

    overview = escape((card.get("overview") or "").strip())
    overview_html = f'<div class="priya-mcard-overview">{overview}</div>' if overview else ""

    match_html = ""
    if card.get("match_pct") is not None:
        match_html = (
            f'<div class="priya-mcard-match">✨ <b>{card["match_pct"]}% AI Match</b>'
            f" from Priya's engine.</div>"
        )

    reasons_html = ""
    reasons = (card.get("reasons") or [])[:3]
    if reasons:
        items = "".join(
            f'<div class="priya-mcard-reason">✓ {escape(r)}</div>' for r in reasons
        )
        reasons_html = (
            '<div class="priya-mcard-reasons"><div class="label">💜 Why Priya picked this</div>'
            f"{items}</div>"
        )

    links: list[str] = []
    if card.get("trailer_key"):
        links.append(
            f'<a class="priya-card-link" href="https://www.youtube.com/watch?v={card["trailer_key"]}" '
            'target="_blank" rel="noopener">▶ Trailer</a>'
        )
    if card.get("tmdb_url"):
        links.append(
            f'<a class="priya-card-link tmdb" href="{card["tmdb_url"]}" '
            'target="_blank" rel="noopener">🔗 TMDB</a>'
        )
    links_html = f'<div class="priya-mcard-links">{"".join(links)}</div>' if links else ""

    return (
        f'<div class="priya-mcard">'
        f"{_poster_html(card)}"
        f'<div class="priya-mcard-body">'
        f'<div class="priya-mcard-title">{title}</div>'
        f"{meta_html}{overview_html}{match_html}{reasons_html}{links_html}"
        f"</div></div>"
    )


def movie_card(card: dict, tmdb, namespace: str, index: int) -> str:
    """Render one card + its action buttons; returns pressed action or "". """
    st.markdown(movie_card_html(card), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    action = ""
    if c1.button("✨ View", key=f"{namespace}_{index}_view", use_container_width=True):
        action = "view"
    if c2.button("❤️", key=f"{namespace}_{index}_save", use_container_width=True):
        action = "save"
    return action


def card_grid(cards: list[dict], tmdb, namespace: str, per_row: int = 4) -> Optional[dict]:
    """Render a responsive-ish grid of cards.

    Returns the card the user asked to view or save, or ``None``.
    """
    result = None
    for index, card in enumerate(cards):
        if index % per_row == 0:
            cols = st.columns(per_row)
        action = movie_card(card, tmdb, namespace, index)
        if action:
            result = {"action": action, "card": card}
    return result


def manage_grid(cards: list[dict], tmdb, namespace: str, per_row: int = 4) -> Optional[dict]:
    """Grid for My Picks: each card gets View + Remove buttons."""
    result = None
    for index, card in enumerate(cards):
        if index % per_row == 0:
            cols = st.columns(per_row)
        st.markdown(movie_card_html(card), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("✨ View", key=f"{namespace}_{index}_pview", use_container_width=True):
            result = {"action": "view", "card": card}
        if c2.button("🗑 Remove", key=f"{namespace}_{index}_premove", use_container_width=True):
            result = {"action": "remove", "card": card}
    return result


def genre_badges(genres: list[str]) -> str:
    return " ".join(f'<span class="priya-genre">{g}</span>' for g in genres)


# --------------------------------------------------------------------- #
# Loading / empty / misc
# --------------------------------------------------------------------- #
def typing_indicator() -> None:
    st.markdown(
        """
        <div class="priya-typing" style="margin:.4rem 0">
          <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def loading_message(msg: str) -> None:
    st.markdown(
        f'<div class="priya-loading"><img src="{theme.priya_avatar_svg(64)}" width="64" height="64" '
        f'alt="Priya"/><div class="msg" style="margin-top:.5rem">{msg}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(emoji: str, title: str, hint: str = "") -> None:
    st.markdown(
        f'<div class="priya-empty"><div class="big">{emoji}</div>'
        f'<div style="font-size:1.05rem;color:var(--text-secondary);margin-bottom:.4rem">{title}</div>'
        f'<div>{hint}</div></div>',
        unsafe_allow_html=True,
    )
