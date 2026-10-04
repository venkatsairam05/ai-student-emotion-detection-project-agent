"""Colour tokens, CSS, and HTML component builders for the dashboard.

Design goals: vibrant, animated, and readable. Three rules keep it from turning
into a clown suit:

1. Colour is reinforcement, never the only cue. Every swatch is paired with an
   emoji and a text label, so the UI still works for colour-vision deficiency,
   in greyscale, and on a projector.
2. Animation is decorative and interruptible. Everything respects
   ``prefers-reduced-motion``, and no animation carries information on its own.
3. Contrast is checked against the dark base. Neon gradients live on large
   shapes; small text stays on high-contrast neutrals.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.emotions import EMOTION_EMOJI, EMOTIONS

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

#: Emotion name -> hex colour. Hues are spread around the wheel so adjacent
#: classes stay separable; lightness varies too, so greyscale remains readable.
EMOTION_COLORS: Dict[str, str] = {
    "angry": "#ff4d6d",     # rose red
    "disgust": "#b06bff",   # violet
    "fear": "#4f7dff",      # indigo blue
    "happy": "#34e5a2",     # mint
    "sad": "#38bdf8",       # sky
    "surprise": "#ffb020",  # amber
    "neutral": "#94a3b8",   # slate
}

#: Engagement band -> hex colour.
BAND_COLORS: Dict[str, str] = {
    "High": "#34e5a2",
    "Moderate": "#fbbf24",
    "Low": "#fb923c",
    "Very Low": "#ff4d6d",
}

#: Brand gradient stops used for headings, borders, and progress fills.
BRAND_GRADIENT = "linear-gradient(120deg, #7c3aed 0%, #ec4899 32%, #f59e0b 62%, #22d3ee 100%)"

#: Cooler variant for surfaces that sit next to warm content.
ACCENT_GRADIENT = "linear-gradient(120deg, #22d3ee 0%, #7c3aed 60%, #ec4899 100%)"

FALLBACK_COLOR = "#94a3b8"


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #


def emotion_color(label: str) -> str:
    """Hex colour for an emotion, falling back to slate."""

    return EMOTION_COLORS.get(label, FALLBACK_COLOR)


def band_color(band: str) -> str:
    """Hex colour for an engagement band, falling back to slate."""

    return BAND_COLORS.get(band, FALLBACK_COLOR)


def color_scale(label: str, alpha: float = 1.0) -> str:
    """Emotion colour with an alpha channel for translucent overlays."""

    base = emotion_color(label).lstrip("#")
    red, green, blue = (int(base[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {max(0.0, min(1.0, alpha))})"


def mix(hex_a: str, hex_b: str, t: float = 0.5) -> str:
    """Blend two hex colours; ``t=0`` returns ``hex_a``, ``t=1`` returns ``hex_b``."""

    def channels(value: str) -> Tuple[int, int, int]:
        raw = value.lstrip("#")
        return tuple(int(raw[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    t = max(0.0, min(1.0, t))
    first, second = channels(hex_a), channels(hex_b)
    blended = tuple(round(a + (b - a) * t) for a, b in zip(first, second))
    return "#{:02x}{:02x}{:02x}".format(*blended)


def emotion_gradient(label: str) -> str:
    """Vertical gradient in an emotion's hue, for bars and card borders."""

    base = emotion_color(label)
    return f"linear-gradient(180deg, {mix(base, '#ffffff', 0.35)} 0%, {base} 55%, {mix(base, '#0b1020', 0.35)} 100%)"


# --------------------------------------------------------------------------- #
# Core stylesheet
# --------------------------------------------------------------------------- #

PAGE_CSS = """
<style>
  /* ---------- base ---------- */
  .stApp, [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(1100px 620px at 8% -8%, rgba(124, 58, 237, 0.30), transparent 60%),
        radial-gradient(900px 560px at 96% 4%, rgba(236, 72, 153, 0.24), transparent 62%),
        radial-gradient(1000px 700px at 50% 108%, rgba(34, 211, 238, 0.22), transparent 60%),
        linear-gradient(180deg, #0b1020 0%, #101733 45%, #0a0f1f 100%);
      background-attachment: fixed;
      color: #e8ecf8;
  }

  .block-container {
      padding-top: 2.4rem;
      padding-bottom: 3rem;
      max-width: 1280px;
      position: relative;
      z-index: 1;
  }

  h1, h2, h3, h4 { letter-spacing: -0.02em; }

  /* Drifting aurora blobs behind the content. Purely decorative. */
  .stApp::before, .stApp::after {
      content: "";
      position: fixed;
      width: 460px; height: 460px;
      border-radius: 50%;
      filter: blur(90px);
      opacity: 0.30;
      pointer-events: none;
      z-index: 0;
      animation: drift 26s ease-in-out infinite alternate;
  }
  .stApp::before {
      top: -140px; left: -110px;
      background: radial-gradient(circle, #7c3aed, transparent 68%);
  }
  .stApp::after {
      bottom: -180px; right: -120px;
      background: radial-gradient(circle, #06b6d4, transparent 68%);
      animation-duration: 34s;
      animation-delay: -8s;
  }
  @keyframes drift {
      0%   { transform: translate3d(0, 0, 0) scale(1); }
      50%  { transform: translate3d(60px, 44px, 0) scale(1.14); }
      100% { transform: translate3d(-40px, 70px, 0) scale(0.94); }
  }

  /* ---------- hero ---------- */
  .es-hero {
      position: relative;
      margin: 0 0 1.4rem 0;
      padding: 1.9rem 2.1rem;
      border-radius: 26px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background:
        linear-gradient(135deg, rgba(124, 58, 237, 0.32), rgba(236, 72, 153, 0.20) 45%, rgba(34, 211, 238, 0.24));
      box-shadow: 0 22px 60px rgba(3, 7, 18, 0.55);
      overflow: hidden;
      animation: riseIn 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .es-hero::before {
      content: "";
      position: absolute; inset: -60% -20%;
      background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.16), transparent 42%);
      animation: spin 14s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .es-hero > * { position: relative; z-index: 1; }

  .es-hero-title {
      margin: 0 0 0.35rem 0;
      font-size: clamp(1.7rem, 3.1vw, 2.7rem);
      font-weight: 800;
      line-height: 1.1;
      background: linear-gradient(100deg, #ffffff 0%, #f5d0fe 24%, #fde68a 52%, #a5f3fc 78%, #ffffff 100%);
      background-size: 220% 100%;
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      animation: sheen 7s ease-in-out infinite;
  }
  @keyframes sheen {
      0%, 100% { background-position: 0% 50%; }
      50%      { background-position: 100% 50%; }
  }
  .es-hero-sub {
      margin: 0;
      max-width: 62ch;
      font-size: 1.02rem;
      line-height: 1.6;
      color: rgba(232, 236, 248, 0.86);
  }

  /* Live status pill */
  .es-pill {
      display: inline-flex; align-items: center; gap: 0.5rem;
      margin-top: 0.95rem;
      padding: 0.34rem 0.9rem 0.34rem 0.7rem;
      border-radius: 999px;
      font-size: 0.8rem; font-weight: 700; letter-spacing: 0.03em;
      text-transform: uppercase;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(11, 16, 32, 0.55);
      backdrop-filter: blur(8px);
  }
  .es-pill.good { color: #bbf7d0; box-shadow: 0 0 0 1px rgba(52, 229, 162, 0.35), 0 0 22px rgba(52, 229, 162, 0.22); }
  .es-pill.warn { color: #fde68a; box-shadow: 0 0 0 1px rgba(251, 191, 36, 0.35), 0 0 22px rgba(251, 191, 36, 0.22); }
  .es-pill.bad  { color: #fecdd3; box-shadow: 0 0 0 1px rgba(255, 77, 109, 0.35), 0 0 22px rgba(255, 77, 109, 0.22); }
  .es-dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 0 0 currentColor;
      animation: pulse 1.9s ease-out infinite;
  }
  @keyframes pulse {
      0%   { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.55); }
      70%  { box-shadow: 0 0 0 11px rgba(255, 255, 255, 0); }
      100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
  }

  /* ---------- section headers ---------- */
  .es-section {
      display: flex; align-items: center; gap: 0.7rem;
      margin: 1.7rem 0 0.85rem 0;
      font-size: 1.28rem; font-weight: 750;
      color: #f1f5ff;
      animation: riseIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .es-section .es-rule {
      flex: 1; height: 2px; border-radius: 2px;
      background: linear-gradient(90deg, rgba(124,58,237,0.85), rgba(236,72,153,0.55) 40%, transparent 92%);
  }
  @keyframes riseIn {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: none; }
  }

  /* ---------- stat cards ---------- */
  .es-stat {
      position: relative;
      padding: 1.05rem 1.15rem 1.1rem 1.15rem;
      border-radius: 20px;
      margin-bottom: 0.7rem;
      background: linear-gradient(160deg, rgba(23, 31, 58, 0.94), rgba(13, 18, 35, 0.94));
      border: 1px solid rgba(255, 255, 255, 0.09);
      box-shadow: 0 14px 34px rgba(2, 6, 16, 0.45);
      overflow: hidden;
      transition: transform 0.28s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.28s ease, border-color 0.28s ease;
      animation: riseIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .es-stat::after {
      content: "";
      position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
      background: var(--es-accent, #7c3aed);
      box-shadow: 0 0 20px var(--es-accent, #7c3aed);
  }
  .es-stat:hover {
      transform: translateY(-5px) scale(1.015);
      border-color: rgba(255, 255, 255, 0.22);
      box-shadow: 0 22px 48px rgba(2, 6, 16, 0.6), 0 0 30px var(--es-glow, rgba(124,58,237,0.28));
  }
  .es-stat-label {
      font-size: 0.74rem; font-weight: 700; letter-spacing: 0.10em;
      text-transform: uppercase; color: rgba(210, 219, 245, 0.72);
  }
  .es-stat-value {
      display: flex; align-items: baseline; gap: 0.4rem;
      margin-top: 0.3rem;
      font-size: clamp(1.5rem, 2.5vw, 2.05rem); font-weight: 800;
      line-height: 1.15; color: #ffffff;
      text-shadow: 0 0 26px var(--es-glow, rgba(124,58,237,0.45));
  }
  .es-stat-value .es-emoji { font-size: 1.25rem; text-shadow: none; }
  .es-stat-note { margin-top: 0.3rem; font-size: 0.8rem; color: rgba(200, 210, 238, 0.68); }

  /* ---------- engagement gauge ---------- */
  .es-gauge { margin: 0.35rem 0 0.6rem 0; }
  .es-gauge-track {
      position: relative; height: 15px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.1);
      overflow: hidden;
  }
  .es-gauge-fill {
      height: 100%; border-radius: 999px;
      background: var(--es-fill, var(--es-accent));
      background-size: 200% 100%;
      animation: sheen 3.4s ease-in-out infinite;
      transition: width 0.9s cubic-bezier(0.22, 1, 0.36, 1);
      box-shadow: 0 0 18px var(--es-accent);
  }
  .es-gauge-ticks { display: flex; justify-content: space-between; margin-top: 0.35rem;
      font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
      text-transform: uppercase; color: rgba(190, 200, 230, 0.6); }

  /* ---------- probability bars ---------- */
  .es-bars { display: flex; flex-direction: column; gap: 0.42rem; margin: 0.35rem 0 0.9rem 0; }
  .es-bar-row { display: grid; grid-template-columns: 8.6rem 1fr 3.6rem; align-items: center; gap: 0.6rem; }
  .es-bar-name { font-size: 0.83rem; font-weight: 650; color: #dbe3fb; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; }
  .es-bar-track {
      height: 12px; border-radius: 999px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.08);
      overflow: hidden;
  }
  .es-bar-fill {
      height: 100%; border-radius: 999px;
      width: 0;
      animation: grow 0.85s cubic-bezier(0.22, 1, 0.36, 1) forwards;
      box-shadow: 0 0 14px var(--es-accent);
  }
  @keyframes grow { from { width: 0; } to { width: var(--es-width, 0%); } }
  .es-bar-value { font-variant-numeric: tabular-nums; font-size: 0.82rem; font-weight: 750;
      text-align: right; color: #eef2ff; }
  .es-bar-row.top .es-bar-name, .es-bar-row.top .es-bar-value { color: #fff; }
  .es-bar-row.top .es-bar-track { border-color: currentColor; }

  /* ---------- feature cards ---------- */
  .es-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
      gap: 0.85rem; margin: 0.6rem 0 1.1rem 0; }
  .es-card {
      padding: 1.05rem 1.1rem;
      border-radius: 18px;
      background: linear-gradient(165deg, rgba(24, 33, 62, 0.92), rgba(12, 17, 33, 0.92));
      border: 1px solid rgba(255, 255, 255, 0.09);
      border-top: 3px solid var(--es-accent, #7c3aed);
      transition: transform 0.3s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.3s ease;
      animation: riseIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .es-card:hover { transform: translateY(-6px) rotateX(4deg);
      box-shadow: 0 20px 44px rgba(2, 6, 16, 0.6), 0 0 26px var(--es-glow, rgba(124,58,237,0.3)); }
  .es-card-icon { font-size: 1.75rem; line-height: 1; display: block; margin-bottom: 0.5rem; }
  .es-card-title { font-weight: 780; font-size: 1rem; color: #ffffff; margin-bottom: 0.28rem; }
  .es-card-text { font-size: 0.86rem; line-height: 1.55; color: rgba(206, 216, 242, 0.82); }

  /* ---------- emotion chips ---------- */
  .emotion-chip {
      display: inline-flex; align-items: center; gap: 0.35rem;
      padding: 0.28rem 0.8rem; margin: 0.22rem 0.3rem 0.22rem 0;
      border-radius: 999px; font-size: 0.83rem; font-weight: 700;
      color: #fff; letter-spacing: 0.01em; white-space: nowrap;
      border: 1px solid rgba(255, 255, 255, 0.22);
      box-shadow: 0 4px 14px rgba(2, 6, 16, 0.4), inset 0 1px 0 rgba(255,255,255,0.28);
      transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
  }
  .emotion-chip:hover {
      transform: translateY(-3px) scale(1.06);
      filter: saturate(1.25) brightness(1.08);
      box-shadow: 0 10px 26px rgba(2, 6, 16, 0.55), 0 0 20px currentColor;
  }
  .emotion-chip .es-weight {
      font-size: 0.72rem; font-weight: 800; opacity: 0.9;
      background: rgba(0, 0, 0, 0.28); border-radius: 999px; padding: 0 0.42rem;
  }

  /* ---------- notices ---------- */
  .es-notice {
      display: flex; gap: 0.75rem; align-items: flex-start;
      padding: 0.9rem 1.1rem; margin: 0.5rem 0 0.9rem 0;
      border-radius: 16px; font-size: 0.9rem; line-height: 1.55;
      border: 1px solid var(--es-edge, rgba(255,255,255,0.14));
      background: linear-gradient(140deg, var(--es-tint, rgba(124,58,237,0.20)), rgba(12,17,33,0.55));
      animation: riseIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  .es-notice-icon { font-size: 1.25rem; line-height: 1.3; }
  .es-notice strong { color: #ffffff; }

  /* ---------- native widget polish ---------- */
  div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800;
      background: linear-gradient(100deg, #ffffff, #c7d2fe);
      -webkit-background-clip: text; background-clip: text; color: transparent; }
  div[data-testid="stMetric"] {
      border-radius: 18px; padding: 0.85rem 1rem;
      background: linear-gradient(160deg, rgba(23,31,58,0.9), rgba(13,18,35,0.9));
      border: 1px solid rgba(255,255,255,0.09);
      transition: transform 0.25s ease, border-color 0.25s ease;
  }
  div[data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: rgba(167,139,250,0.5); }

  div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
      border-radius: 999px; font-weight: 750; letter-spacing: 0.02em;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: 0 8px 22px rgba(2, 6, 16, 0.42);
      transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
  }
  div.stButton > button:hover, div.stDownloadButton > button:hover {
      transform: translateY(-2px) scale(1.02); filter: brightness(1.1);
      box-shadow: 0 14px 30px rgba(2, 6, 16, 0.5), 0 0 20px rgba(167, 139, 250, 0.45);
  }
  div.stButton > button:active { transform: translateY(0) scale(0.99); }

  [data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 5px rgba(167,139,250,0.22); }

  [data-baseweb="tab-list"] { gap: 0.4rem; border-radius: 14px; padding: 0.3rem;
      background: rgba(255,255,255,0.05); }
  [data-baseweb="tab"] { border-radius: 11px !important; transition: background 0.25s ease, color 0.25s ease; }
  [data-baseweb="tab"]:hover { background: rgba(167, 139, 250, 0.18); }
  [data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(120deg, rgba(124,58,237,0.55), rgba(236,72,153,0.45)) !important;
      box-shadow: 0 6px 18px rgba(2,6,16,0.4);
  }

  [data-testid="stDataFrame"], [data-testid="stExpander"] {
      border-radius: 16px; overflow: hidden;
      border: 1px solid rgba(255,255,255,0.09);
      background: rgba(13, 18, 35, 0.55);
  }
  [data-testid="stExpander"] details summary { transition: color 0.2s ease; }
  [data-testid="stExpander"] details summary:hover { color: #c4b5fd; }

  img { border-radius: 16px; transition: transform 0.3s ease, box-shadow 0.3s ease; }
  img:hover { transform: scale(1.012); box-shadow: 0 18px 40px rgba(2,6,16,0.55); }

  .stCaption { line-height: 1.5; color: rgba(206, 216, 242, 0.72); }
  footer, #MainMenu, [data-testid="stStatusWidget"] { visibility: hidden; height: 0; }

  /* ---------- reduced motion: strip every animation ---------- */
  @media (prefers-reduced-motion: reduce) {
    .stApp::before, .stApp::after, .es-hero::before, .es-gauge-fill,
    .es-hero-title, .es-bar-fill, .es-dot, .es-stat, .es-card, .es-section, .es-notice {
      animation: none !important;
    }
    .es-bar-fill { width: var(--es-width, 0%); }
    .es-stat:hover, .es-card:hover, .emotion-chip:hover, img:hover,
    div.stButton > button:hover { transform: none; }
  }
</style>
"""

SIDEBAR_CSS = """
<style>
  section[data-testid="stSidebar"] {
      padding-top: 1.2rem;
      border-right: 1px solid rgba(167, 139, 250, 0.16);
      background:
        linear-gradient(190deg, rgba(124, 58, 237, 0.16), rgba(13, 18, 35, 0.7) 45%, rgba(6, 211, 238, 0.10));
  }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
      background: linear-gradient(100deg, #ffffff, #f5d0fe 40%, #a5f3fc);
      -webkit-background-clip: text; background-clip: text; color: transparent;
      letter-spacing: -0.01em;
  }
  section[data-testid="stSidebar"] [role="radiogroup"] label {
      border-radius: 12px; padding: 0.3rem 0.5rem; margin-bottom: 0.15rem;
      transition: background 0.22s ease, transform 0.22s ease;
  }
  section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
      background: rgba(167, 139, 250, 0.16); transform: translateX(3px);
  }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: 0.75; }
</style>
"""


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #


def hero_html(
    title: str,
    subtitle: str,
    status: str = "",
    status_tone: str = "good",
) -> str:
    """Animated hero banner with an optional pulsing status pill."""

    pill = ""
    if status:
        pill = (
            f'<div class="es-pill {status_tone}">'
            f'<span class="es-dot"></span>{status}</div>'
        )
    return (
        '<div class="es-hero">'
        f'<h1 class="es-hero-title">{title}</h1>'
        f'<p class="es-hero-sub">{subtitle}</p>'
        f"{pill}</div>"
    )


def section_header_html(text: str, icon: str = "") -> str:
    """Section title with an animated gradient rule."""

    return (
        f'<div class="es-section">{icon}<span>{text}</span>'
        '<span class="es-rule"></span></div>'
    )


def stat_card_html(
    label: str,
    value: str,
    accent: str = "#7c3aed",
    note: str = "",
    icon: str = "",
    delay_ms: int = 0,
    width: Optional[str] = None,
) -> str:
    """One animated metric card.

    ``delay_ms`` staggers the entrance animation so a row of cards cascades
    instead of appearing all at once.
    """

    glow = color_scale(accent, 0.34) if accent.startswith("#") else accent
    style = (
        f"--es-accent:{accent};--es-glow:{glow};"
        f"animation-delay:{delay_ms}ms;"
    )
    if width:
        style += f"width:{width};"
    icon_html = f'<span class="es-emoji">{icon}</span>' if icon else ""
    note_html = f'<div class="es-stat-note">{note}</div>' if note else ""
    return (
        f'<div class="es-stat" style="{style}">'
        f'<div class="es-stat-label">{label}</div>'
        f'<div class="es-stat-value">{icon_html}<span>{value}</span></div>'
        f"{note_html}</div>"
    )


def metric_cards_html(metrics: Dict[str, Any]) -> str:
    """The headline metrics row as a single HTML block.

    Takes a :meth:`SessionTracker.snapshot` payload. Values are also written as
    text (not just colour) so the row stays readable in greyscale.
    """

    from src.emotions import EMOTION_BY_NAME, engagement_band

    engagement = float(metrics.get("mean_engagement", 0.0))
    band = str(metrics.get("engagement_band") or engagement_band(engagement))
    dominant = str(metrics.get("dominant_label", "neutral"))
    emoji = EMOTION_EMOJI.get(dominant, "🙂")
    faces = int(metrics.get("face_count", 0))

    cards = [
        stat_card_html(
            "Engagement", f"{engagement:.0%}", accent=band_color(band),
            note=f"{band} band", icon="🎯", delay_ms=0,
        ),
        stat_card_html(
            "Mood band", band, accent=band_color(band),
            note="weighted emotion score", icon="📊", delay_ms=70,
        ),
        stat_card_html(
            "Faces in frame", str(faces), accent="#4f7dff",
            note="frontal detections", icon="👥", delay_ms=140,
        ),
        stat_card_html(
            "Dominant emotion", dominant.title(), accent=emotion_color(dominant),
            note=emoji, icon=emoji, delay_ms=210,
        ),
    ]
    return "".join(cards)


def band_gauge_html(engagement: float, band: str = "") -> str:
    """Animated engagement gauge with band tick labels."""

    from src.emotions import engagement_band

    value = max(0.0, min(1.0, float(engagement)))
    band = band or engagement_band(value)
    accent = band_color(band)
    percent = round(value * 100)

    return (
        '<div class="es-gauge">'
        f'<div class="es-gauge-track">'
        f'<div class="es-gauge-fill" style="--es-accent:{accent};--es-fill:{emotion_gradient(band) if band == "High" else accent};'
        f'width:{percent}%"></div></div>'
        '<div class="es-gauge-ticks"><span>Very Low</span><span>Low</span>'
        "<span>Moderate</span><span>High</span></div></div>"
    )


def probability_bars_html(probabilities: Dict[str, float], top: int = 1) -> str:
    """Animated per-class probability bars in each emotion's colour."""

    from src.ui.charts import ordered_labels

    labels = ordered_labels(probabilities)
    if not labels:
        return ""

    ranked = sorted(labels, key=lambda name: float(probabilities.get(name, 0.0)), reverse=True)
    top_names = set(ranked[: max(1, top)])

    rows = []
    for index, name in enumerate(labels):
        value = max(0.0, min(1.0, float(probabilities.get(name, 0.0))))
        percent = round(value * 100)
        accent = emotion_color(name)
        highlight = " top" if name in top_names else ""
        name_html = f"{EMOTION_EMOJI.get(name, '🙂')} {name.title()}"
        rows.append(
            f'<div class="es-bar-row{highlight}" style="--es-accent:{accent};color:{accent}">'
            f'<div class="es-bar-name">{name_html}</div>'
            f'<div class="es-bar-track"><div class="es-bar-fill" '
            f'style="--es-width:{percent}%;background:{emotion_gradient(name)};'
            f"animation-delay:{index * 60}ms\"></div></div>"
            f'<div class="es-bar-value">{value:.1%}</div></div>'
        )
    return f'<div class="es-bars">{"".join(rows)}</div>'


def feature_cards_html(items: Sequence[Tuple[str, str, str]]) -> str:
    """Responsive grid of hover-animated feature cards.

    ``items`` is a sequence of ``(icon, title, text)`` triples.
    """

    palette = ["#7c3aed", "#ec4899", "#f59e0b", "#22d3ee", "#34e5a2", "#4f7dff"]
    cards = []
    for index, (icon, title, text) in enumerate(items):
        accent = palette[index % len(palette)]
        cards.append(
            f'<div class="es-card" style="--es-accent:{accent};'
            f'--es-glow:{color_scale(accent, 0.3)};animation-delay:{index * 60}ms">'
            f'<span class="es-card-icon">{icon}</span>'
            f'<div class="es-card-title">{title}</div>'
            f'<div class="es-card-text">{text}</div></div>'
        )
    return f'<div class="es-grid">{"".join(cards)}</div>'


def notice_html(message: str, tone: str = "info", icon: str = "") -> str:
    """Inline callout with a tinted gradient background.

    ``tone`` is one of ``info``, ``success``, ``warn``, ``error``.
    """

    styles = {
        "info": ("rgba(124,58,237,0.20)", "rgba(167,139,250,0.55)", "💡"),
        "success": ("rgba(52,229,162,0.18)", "rgba(52,229,162,0.55)", "✅"),
        "warn": ("rgba(251,191,36,0.18)", "rgba(251,191,36,0.55)", "⚠️"),
        "error": ("rgba(255,77,109,0.20)", "rgba(255,77,109,0.55)", "🛑"),
    }
    tint, edge, default_icon = styles.get(tone, styles["info"])
    return (
        f'<div class="es-notice" style="--es-tint:{tint};--es-edge:{edge}">'
        f'<span class="es-notice-icon">{icon or default_icon}</span>'
        f"<span>{message}</span></div>"
    )


def emotion_legend_markdown() -> str:
    """Markdown legend with colour chips for every emotion."""

    chips = []
    for emotion in EMOTIONS:
        chips.append(
            f'<span class="emotion-chip" style="background:{emotion_gradient(emotion.name)};'
            f'color:{emotion_color(emotion.name)}">'
            f'{emotion.emoji} {emotion.name.title()}'
            f'<span class="es-weight">{emotion.engagement_weight:.2f}</span></span>'
        )
    return " ".join(chips)


def legend_footer_html() -> str:
    """The always-on legend strip shown at the bottom of every page."""

    return (
        '<div style="margin-top:1.4rem">'
        '<div class="es-stat-label" style="margin-bottom:0.45rem">'
        "Engagement weight per emotion &middot; hover a chip</div>"
        f"{emotion_legend_markdown()}</div>"
    )


# --------------------------------------------------------------------------- #
# Streamlit plumbing
# --------------------------------------------------------------------------- #


def apply_theme(st: Any) -> None:
    """Inject the dashboard CSS into a Streamlit page.

    Only touches ``markdown`` on both ``st`` and ``st.sidebar`` so this stays
    safe against minimal Streamlit stand-ins used in tests.
    """

    try:
        st.markdown(PAGE_CSS, unsafe_allow_html=True)
        st.sidebar.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    except Exception:
        pass