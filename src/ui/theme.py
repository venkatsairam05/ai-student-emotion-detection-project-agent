"""Global cinematic theme for PRIYA.

Injected as a single ``<style>`` block so the whole app shares one design
language: dark cinematic background, purple/pink gradient accents, glass
panels, premium cards and a chat UI that does not look like "default
Streamlit".

Everything lives behind CSS variables (see ``:root``) so it can be re-themed
in one place.
"""

from __future__ import annotations

import base64

import streamlit as st

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');

:root {
    --bg-primary: #0B0714;
    --bg-secondary: #120D20;
    --card-bg: #181127;
    --card-bg-hover: #1E1630;
    --purple: #A855F7;
    --pink: #EC4899;
    --green: #22C55E;
    --text-primary: #FFFFFF;
    --text-secondary: #B8B1C9;
    --muted: #777080;
    --gradient: linear-gradient(135deg, #A855F7, #EC4899);
    --radius: 18px;
    --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-display: 'Poppins', var(--font-ui);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    background-image:
        radial-gradient(1000px 600px at 85% -10%, rgba(168, 85, 247, 0.16), transparent 60%),
        radial-gradient(800px 500px at -10% 20%, rgba(236, 72, 153, 0.10), transparent 60%),
        radial-gradient(700px 700px at 50% 120%, rgba(168, 85, 247, 0.08), transparent 55%);
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: var(--font-ui);
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], footer { visibility: hidden; height: 0; }
[data-testid="stAppViewContainer"] > .main { padding-top: 1.2rem; }

/* Streamlit chrome */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(18, 13, 32, 0.92), rgba(11, 7, 20, 0.97)) !important;
    border-right: 1px solid rgba(168, 85, 247, 0.15);
    backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] hr { border-color: rgba(168, 85, 247, 0.15); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: var(--font-display);
    color: var(--text-primary);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: var(--text-secondary);
}

/* ---------- Navigation bar ---------- */
.priya-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.7rem 1.4rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(18, 13, 32, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
    margin-bottom: 0.6rem;
}
.priya-nav-brand { display: flex; align-items: center; gap: 0.6rem; }
.priya-nav-brand img { border-radius: 50%; box-shadow: 0 0 18px rgba(168, 85, 247, 0.55); }
.priya-nav-title { line-height: 1.05; }
.priya-nav-title .name {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: 0.5px;
    background: var(--gradient);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.priya-nav-title .tag { font-size: 0.72rem; color: var(--muted); }
.priya-nav-links { display: flex; align-items: center; gap: 0.3rem; flex-wrap: wrap; }
.priya-navlink {
    color: var(--text-secondary); text-decoration: none;
    font-weight: 600; font-size: 0.9rem;
    padding: 0.42rem 1rem; border-radius: 999px;
    border: 1px solid transparent; transition: all .18s ease;
}
.priya-navlink:hover {
    color: #fff; border-color: rgba(168, 85, 247, 0.4);
    background: rgba(168, 85, 247, 0.12);
}
.priya-navlink.active {
    color: #fff; border-color: rgba(168, 85, 247, 0.5);
    background: rgba(168, 85, 247, 0.16);
}
.priya-status {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.78rem; color: var(--text-secondary);
    border: 1px solid rgba(34, 197, 94, 0.35);
    background: rgba(34, 197, 94, 0.08);
    border-radius: 999px; padding: 0.28rem 0.8rem; white-space: nowrap;
}
.priya-status .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ---------- Hero ---------- */
.priya-hero { text-align: center; padding: 2.4rem 1rem 1.6rem; }
.priya-hero-avatar { margin-bottom: 0.9rem; }
.priya-hero-avatar img {
    border-radius: 50%;
    box-shadow: 0 0 0 6px rgba(168, 85, 247, 0.12), 0 0 60px rgba(168, 85, 247, 0.5);
    animation: floaty 5s ease-in-out infinite;
}
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.priya-hero-eyebrow {
    font-family: var(--font-display); font-weight: 600; font-size: 0.85rem;
    letter-spacing: 3px; text-transform: uppercase;
    background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.priya-hero-title {
    font-family: var(--font-display);
    font-weight: 800; font-size: clamp(1.9rem, 4.5vw, 3.2rem); line-height: 1.12;
    margin: 0.6rem 0 0.5rem;
    background: linear-gradient(180deg, #ffffff, #cfc3ec);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.priya-hero-sub { color: var(--text-secondary); font-size: 1.02rem; max-width: 640px; margin: 0 auto 1.4rem; }

/* ---------- Generic cards / sections ---------- */
.priya-section-title {
    font-family: var(--font-display); font-weight: 700; font-size: 1.35rem;
    margin: 1.6rem 0 0.9rem; display: flex; align-items: center; gap: 0.5rem;
}
.priya-subtext { color: var(--muted); font-size: 0.9rem; margin-top: -0.6rem; margin-bottom: 1rem; }

/* ---------- Movie cards ---------- */
.priya-card-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 1rem; }
.priya-mcard {
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--card-bg);
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.priya-mcard:hover { transform: translateY(-4px) scale(1.015); border-color: rgba(168, 85, 247, 0.45); box-shadow: 0 14px 40px rgba(168, 85, 247, 0.22); }
.priya-mcard-poster { position: relative; aspect-ratio: 2 / 3; background: #0d0916; }
.priya-mcard-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.priya-mcard-poster .ph {
    width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(160deg, #221936, #120d20);
    color: var(--text-secondary); font-weight: 600; text-align: center; padding: 0.6rem; font-size: 0.85rem;
}
.priya-match-badge {
    position: absolute; top: 8px; right: 8px;
    background: rgba(11, 7, 20, 0.72);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(236, 72, 153, 0.5);
    color: #fff; font-weight: 700; font-size: 0.75rem;
    padding: 0.22rem 0.55rem; border-radius: 999px;
}
.priya-match-badge .spark { background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent; }
.priya-mcard-type {
    position: absolute; top: 8px; left: 8px;
    background: rgba(11, 7, 20, 0.72); backdrop-filter: blur(6px);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #fff; font-size: 0.68rem; font-weight: 600;
    padding: 0.2rem 0.5rem; border-radius: 999px;
}
.priya-mcard-body { padding: 0.65rem 0.75rem 0.75rem; }
.priya-mcard-title { font-family: var(--font-display); font-weight: 700; font-size: 0.95rem; line-height: 1.25; margin-bottom: 0.2rem; }
.priya-mcard-meta { color: var(--text-secondary); font-size: 0.78rem; }
.priya-mcard-rating { color: #ffd76e; font-weight: 700; font-size: 0.8rem; }
.priya-mcard-actions { display: flex; gap: 0.5rem; margin-top: 0.55rem; }
.priya-mcard-actions div[data-testid="stButton"] button {
    border-radius: 999px !important; padding: 0.3rem 0.7rem !important;
    font-size: 0.78rem !important; font-weight: 600;
    background: rgba(168, 85, 247, 0.10) !important; border: 1px solid rgba(168, 85, 247, 0.35) !important; color: #fff !important;
}
.priya-mcard-actions div[data-testid="stButton"] button:hover { background: rgba(168, 85, 247, 0.25) !important; }
.priya-mcard-actions div[data-testid="stButton"] button:focus { box-shadow: none !important; }

/* card meta / overview / match / reasons / links */
.priya-mcard-body { display: flex; flex-direction: column; gap: 0.3rem; }
.priya-mcard-meta { color: var(--text-secondary); font-size: 0.78rem; line-height: 1.5; }
.priya-mcard-meta .m { white-space: normal; }
.priya-mcard-overview {
    color: var(--text-secondary); font-size: 0.8rem; line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; margin-top: 0.15rem;
}
.priya-mcard-match {
    font-size: 0.8rem; color: #fff; margin-top: 0.15rem;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(236, 72, 153, 0.35);
    border-radius: 10px; padding: 0.28rem 0.55rem;
}
.priya-mcard-match b {
    background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.priya-mcard-reasons { margin-top: 0.15rem; }
.priya-mcard-reasons .label { font-size: 0.7rem; color: var(--muted); font-weight: 600; margin-bottom: 0.1rem; }
.priya-mcard-reason { font-size: 0.72rem; color: var(--text-secondary); line-height: 1.35; }
.priya-mcard-links { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.35rem; }
.priya-card-link {
    display: inline-flex; align-items: center;
    padding: 0.28rem 0.62rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; color: #fff; text-decoration: none;
    border: 1px solid rgba(168, 85, 247, 0.4);
    background: rgba(168, 85, 247, 0.12);
    transition: all .18s ease;
}
.priya-card-link:hover {
    border-color: var(--pink); background: rgba(168, 85, 247, 0.24);
    box-shadow: 0 0 14px rgba(168, 85, 247, 0.35); color: #fff;
}
.priya-card-link.tmdb { border-color: rgba(34, 197, 94, 0.4); background: rgba(34, 197, 94, 0.10); }
.priya-card-link.tmdb:hover { border-color: var(--green); background: rgba(34, 197, 94, 0.20); box-shadow: 0 0 14px rgba(34, 197, 94, 0.3); }

/* Responsive card grid: 4-up on wide screens, wraps 2-3 on tablets, 1-2 on phones */
@media (max-width: 1150px) {
  [data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
  }
  [data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 300px !important;
    min-width: 280px !important;
  }
}

/* genre badges (details view) */
.priya-genre {
    display: inline-block; margin: 0.15rem 0.2rem 0.15rem 0;
    padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem;
    color: var(--text-secondary);
    border: 1px solid rgba(168, 85, 247, 0.35);
    background: rgba(168, 85, 247, 0.08);
}

/* ---------- Chips (moods / quick prompts) ---------- */
.priya-chip-row { display: flex; flex-wrap: wrap; gap: 0.55rem; justify-content: center; margin: 0.9rem 0; }
.priya-chip-row div[data-testid="stButton"] button {
    border-radius: 999px !important;
    padding: 0.45rem 1.05rem !important;
    font-size: 0.85rem !important; font-weight: 600;
    color: var(--text-secondary) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    transition: all .18s ease;
}
.priya-chip-row div[data-testid="stButton"] button:hover {
    color: #fff !important;
    border-color: rgba(168, 85, 247, 0.6) !important;
    background: rgba(168, 85, 247, 0.16) !important;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.25);
}
.priya-chip-row div[data-testid="stButton"] button:focus { box-shadow: none !important; }

/* ---------- Buttons ---------- */
div[data-testid="stButton"] button {
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid rgba(168, 85, 247, 0.4);
    background: rgba(168, 85, 247, 0.12);
    color: #fff;
    transition: all .18s ease;
}
div[data-testid="stButton"] button:hover { border-color: var(--pink); background: rgba(168, 85, 247, 0.22); box-shadow: 0 0 18px rgba(168, 85, 247, 0.3); }
div[data-testid="stButton"] button:focus { box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.4); }
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--gradient) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 8px 24px rgba(168, 85, 247, 0.35);
}
div[data-testid="stButton"] button[kind="primary"]:hover { box-shadow: 0 10px 30px rgba(236, 72, 153, 0.5); filter: brightness(1.08); }

/* ---------- Text inputs / chat input ---------- */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] > div > div {
    background: rgba(18, 13, 32, 0.8) !important; color: #fff !important;
    border-color: rgba(168, 85, 247, 0.25) !important; border-radius: 12px !important;
}
[data-testid="stChatInput"] { border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 999px; background: rgba(18, 13, 32, 0.8); backdrop-filter: blur(8px); }
[data-testid="stChatInput"] textarea { color: #fff !important; }
[data-testid="stChatInput"] button { background: var(--gradient) !important; border: none !important; border-radius: 50% !important; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0.15rem 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { border: none !important; }
[data-testid="stChatMessageAvatarUser"] { border-radius: 50%; box-shadow: 0 0 12px rgba(236, 72, 153, 0.5); }

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] > div { border-radius: 18px; padding: 0.75rem 1rem; font-size: 0.95rem; line-height: 1.5; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div {
    background: var(--gradient); color: #fff;
    border-top-right-radius: 6px;
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.3);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] > div {
    background: rgba(24, 17, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-top-left-radius: 6px;
    backdrop-filter: blur(8px);
}

/* typing indicator */
.priya-typing {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(24, 17, 39, 0.85); border: 1px solid rgba(255,255,255,0.07);
    padding: 0.8rem 1.1rem; border-radius: 18px; border-top-left-radius: 6px;
}
.priya-typing span { width: 8px; height: 8px; border-radius: 50%; background: var(--purple); animation: blink 1.2s infinite; }
.priya-typing span:nth-child(2) { animation-delay: .2s; background: #c084fc; }
.priya-typing span:nth-child(3) { animation-delay: .4s; background: var(--pink); }
@keyframes blink { 0%,80%,100% { transform: scale(0.6); opacity: .4; } 40% { transform: scale(1); opacity: 1; } }

/* ---------- Loading ---------- */
.priya-loading { text-align: center; padding: 1.2rem 0; }
.priya-loading .msg { font-family: var(--font-display); font-weight: 600; color: var(--text-secondary); }

/* ---------- Progress (match bar) ---------- */
[data-testid="stProgress"] > div > div > div { background: var(--gradient) !important; }

/* ---------- Footer ---------- */
.priya-footer {
    margin-top: 3rem; padding: 1.6rem 1rem; text-align: center;
    border-top: 1px solid rgba(168, 85, 247, 0.15);
    color: var(--muted); font-size: 0.85rem;
}
.priya-footer .name { font-family: var(--font-display); font-weight: 700; color: var(--text-secondary); }
.priya-footer a { color: var(--purple); text-decoration: none; }
.priya-footer a:hover { color: var(--pink); }

/* ---------- Empty state ---------- */
.priya-empty { text-align: center; padding: 3rem 1rem; color: var(--muted); }
.priya-empty .big { font-size: 2.4rem; margin-bottom: 0.6rem; }

/* ---------- Dialogs / details ---------- */
[data-testid="stDialog"] [data-testid="stDialogBody"] { background: var(--bg-secondary); border-radius: 20px; }
.priya-detail-backdrop { border-radius: 16px; overflow: hidden; margin-bottom: 1rem; }
.priya-detail-backdrop img { width: 100%; border-radius: 16px; }
.priya-detail-title { font-family: var(--font-display); font-weight: 800; font-size: 1.7rem; }
.priya-detail-meta { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 0.6rem; }
.priya-reason {
    border-left: 3px solid var(--purple); background: rgba(168, 85, 247, 0.08);
    padding: 0.5rem 0.8rem; border-radius: 0 12px 12px 0; margin: 0.35rem 0;
}

/* misc */
.stMarkdown a { color: var(--purple); }
[data-testid="stExpander"] { border: 1px solid rgba(168,85,247,0.15); border-radius: 14px; background: rgba(24,17,39,0.5); }
[data-testid="stRadio"] label { color: var(--text-secondary) !important; }
"""


def inject() -> None:
    """Inject the global PRIYA theme into the running app."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def priya_avatar_svg(size: int = 96) -> str:
    """Base64 data-URI of the Priya orb avatar (usable in img src / st.chat_message)."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="{size}" height="{size}">
  <defs>
    <radialGradient id="pg" cx="35%" cy="28%" r="80%">
      <stop offset="0%" stop-color="#D8B4FE"/>
      <stop offset="45%" stop-color="#A855F7"/>
      <stop offset="100%" stop-color="#7C3AED"/>
    </radialGradient>
    <linearGradient id="pr" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#A855F7"/>
      <stop offset="100%" stop-color="#EC4899"/>
    </linearGradient>
    <filter id="pglow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="7" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <circle cx="60" cy="60" r="50" fill="url(#pg)" opacity="0.55" filter="url(#pglow)"/>
  <circle cx="60" cy="60" r="46" fill="url(#pg)"/>
  <circle cx="60" cy="60" r="46" fill="none" stroke="url(#pr)" stroke-width="2.5" opacity="0.9"/>
  <circle cx="60" cy="60" r="30" fill="none" stroke="#ffffff" stroke-opacity="0.35" stroke-width="1.4" stroke-dasharray="2.5 6"/>
  <path d="M60 16 l2.6 7.6 7.6 2.6 -7.6 2.6 -2.6 7.6 -2.6 -7.6 -7.6 -2.6 7.6 -2.6 Z" fill="#ffffff" opacity="0.95"/>
  <circle cx="84" cy="82" r="3.2" fill="#ffffff" opacity="0.7"/>
  <circle cx="30" cy="30" r="2" fill="#ffffff" opacity="0.5"/>
</svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
