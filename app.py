import html

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# same 9 features. app.py is its own file, so it can't see ur notebook variables.
FEATURES = ["danceability", "energy", "valence", "tempo", "acousticness",
            "instrumentalness", "speechiness", "liveness", "loudness"]

# the four that are already 0-1, so they can be shown as meters without scaling
PROFILE = ["danceability", "energy", "valence", "acousticness"]

st.set_page_config(page_title="Song Twin Finder", page_icon="🎧",
                   layout="centered")


# ---------------------------------------------------------------- styling ---
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 780px; }

  /* ---------- hero ---------- */
  .hero { margin-bottom: 1.6rem; }
  .hero h1 {
    font-size: 2.7rem; font-weight: 700; letter-spacing: -.035em;
    margin: 0 0 .45rem; line-height: 1.08;
    background: linear-gradient(92deg, #c4b5fd, #f0abfc 55%, #fda4af);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: #8b8a96; font-size: 1.02rem; margin: 0; }
  .stat {
    display: inline-flex; align-items: center; gap: .45rem;
    font-size: .74rem; color: #64636f; margin-top: .8rem;
  }
  .stat b { color: #9a99a6; font-weight: 600; }
  .dot { width: 3px; height: 3px; border-radius: 50%; background: #3a3945; }

  /* ---------- search ---------- */
  .stTextInput label { color: #8b8a96 !important; font-size: .85rem !important; }
  .stTextInput input {
    background: #16161d !important; border: 1px solid #2a2a35 !important;
    border-radius: 12px !important; padding: .9rem 1rem !important;
    font-size: 1rem !important; color: #ecebf0 !important;
  }
  .stTextInput input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.15) !important;
  }

  /* ---------- empty state ---------- */
  .empty { margin-top: 2.2rem; }
  .empty .lbl {
    font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
    color: #6e6d7a; font-weight: 600; margin-bottom: .75rem;
  }
  .chips { display: flex; flex-wrap: wrap; gap: .45rem; }
  .chip {
    font-size: .82rem; color: #a9a8b5;
    background: #16161d; border: 1px solid #26262f;
    padding: .42rem .8rem; border-radius: 999px;
  }

  /* ---------- seed ---------- */
  .seed {
    background: linear-gradient(97deg, rgba(167,139,250,.13), rgba(240,171,252,.05));
    border: 1px solid rgba(167,139,250,.26);
    border-radius: 16px; padding: 1.15rem 1.25rem; margin: 1.7rem 0 2rem;
  }
  .seed-top { display: flex; align-items: center; gap: .95rem; }
  .seed .lbl {
    font-size: .66rem; letter-spacing: .13em; text-transform: uppercase;
    color: #a78bfa; font-weight: 600; margin-bottom: .2rem;
  }
  .seed .ttl { font-size: 1.06rem; font-weight: 650; color: #f4f3f7; }
  .seed .art { font-size: .86rem; color: #9a99a6; margin-top: .1rem; }

  /* audio profile meters — one hue, magnitude only */
  .profile {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: .55rem 1.5rem;
    margin-top: 1.1rem; padding-top: 1rem;
    border-top: 1px solid rgba(167,139,250,.16);
  }
  .feat { display: flex; align-items: center; gap: .6rem; }
  .feat .k { font-size: .72rem; color: #8b8a96; width: 92px; flex: none; }
  .track {
    flex: 1; height: 4px; border-radius: 2px;
    background: rgba(255,255,255,.07); overflow: hidden;
  }
  .fill {
    display: block; height: 100%; border-radius: 2px; background: #a78bfa;
  }
  .feat .v {
    font-size: .72rem; color: #b9b8c4; width: 26px; flex: none;
    text-align: right; font-variant-numeric: tabular-nums;
  }

  /* ---------- results ---------- */
  .sect {
    font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
    color: #6e6d7a; font-weight: 600; margin: 0 0 .85rem .15rem;
  }
  .row {
    display: flex; align-items: center; gap: .95rem;
    background: #131319; border: 1px solid #22222c;
    border-radius: 14px; padding: .8rem .95rem; margin-bottom: .55rem;
    transition: border-color .18s, background .18s, transform .18s;
    animation: rise .4s cubic-bezier(.4,0,.2,1) both;
  }
  @keyframes rise { from { opacity: 0; transform: translateY(7px); } }
  .row:hover {
    background: #17171f; border-color: #35353f; transform: translateX(3px);
  }
  .art-tile {
    flex: none; width: 46px; height: 46px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: .95rem; font-weight: 700; color: rgba(255,255,255,.92);
    letter-spacing: -.02em;
  }
  .meta { flex: 1; min-width: 0; }
  .name {
    font-size: .96rem; font-weight: 600; color: #f0eff4;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .by {
    font-size: .81rem; color: #85848f; margin-top: .13rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .side { flex: none; text-align: right; }
  .genre {
    display: inline-block; font-size: .67rem; color: #a9a8b5;
    background: #1f1f28; border: 1px solid #2c2c37;
    padding: .2rem .55rem; border-radius: 999px; white-space: nowrap;
  }
  .rank { font-size: .67rem; color: #55545f; margin-top: .32rem; }

  .note {
    color: #5d5c68; font-size: .76rem; line-height: 1.55;
    margin-top: 1.9rem; padding-top: 1rem; border-top: 1px solid #1e1e26;
  }
</style>
""", unsafe_allow_html=True)


# @st.cache_data = "run this once, then remember the answer."
# without it, streamlit reloads 100k rows on every single keystroke. brutal.
@st.cache_data
def load():
    d = pd.read_csv("songs.csv")
    return d, MinMaxScaler().fit_transform(d[FEATURES])


df, X = load()

# stand-in album art: a stable gradient per song, so the same track always gets
# the same tile. purely decorative — no data is encoded in the colour.
PAIRS = [("#7c3aed", "#db2777"), ("#2563eb", "#06b6d4"), ("#ea580c", "#f59e0b"),
         ("#059669", "#84cc16"), ("#db2777", "#f97316"), ("#4f46e5", "#a855f7")]


def tile(name):
    a, b = PAIRS[sum(ord(c) for c in str(name)) % len(PAIRS)]
    return (f'<div class="art-tile" style="background:linear-gradient(135deg,{a},{b})">'
            f'{html.escape(str(name)[:1].upper())}</div>')


def meters(row):
    """Seed song's audio profile. One hue, magnitude only, every bar labelled."""
    out = []
    for f in PROFILE:
        v = float(row[f])
        out.append(
            f'<div class="feat"><span class="k">{f}</span>'
            f'<span class="track"><span class="fill" style="width:{v * 100:.0f}%"></span></span>'
            f'<span class="v">{v:.2f}</span></div>'
        )
    return f'<div class="profile">{"".join(out)}</div>'


st.markdown(
    '<div class="hero"><h1>🎧 Song Twin Finder</h1>'
    '<p>Type a song you love. Get five that feel like it.</p>'
    f'<div class="stat"><b>{len(df):,}</b> songs'
    '<span class="dot"></span><b>9</b> audio features</div></div>',
    unsafe_allow_html=True,
)

# text_input draws a box and hands back whatever they type
query = st.text_input("Song Name", placeholder="blinding lights")

# streamlit reruns this whole file top-to-bottom on every interaction.
# this "if" is what stops it doing anything before they've typed.
if not query:
    st.markdown(
        '<div class="empty"><div class="lbl">Try one of these</div><div class="chips">'
        + "".join(f'<span class="chip">{s}</span>' for s in
                  ["Blinding Lights", "Bohemian Rhapsody", "Clair de Lune",
                   "Enter Sandman", "Weightless", "Take Five"])
        + '</div></div>',
        unsafe_allow_html=True,
    )
else:
    m = df[df["track_name"].str.lower().str.contains(
        query.lower().strip(), na=False, regex=False)]

    if m.empty:
        st.warning("Couldn't find that one — try another spelling?")
    else:
        # exact same logic as the notebook. u already built this part.
        idx = int(m["popularity"].idxmax())
        seed = df.iloc[idx]

        st.markdown(
            f'<div class="seed"><div class="seed-top">{tile(seed["track_name"])}<div>'
            f'<div class="lbl">based on</div>'
            f'<div class="ttl">{html.escape(str(seed["track_name"]))}</div>'
            f'<div class="art">{html.escape(str(seed["artists"]))}</div>'
            f'</div></div>{meters(seed)}</div>',
            unsafe_allow_html=True,
        )

        sims = cosine_similarity(X[idx].reshape(1, -1), X).flatten()
        order = [i for i in np.argsort(sims)[::-1] if i != idx][:5]

        st.markdown('<div class="sect">Your five twins</div>',
                    unsafe_allow_html=True)

        rows = []
        for rank, i in enumerate(order, start=1):
            r = df.iloc[i]
            rows.append(
                f'<div class="row" style="animation-delay:{rank * 45}ms">'
                f'{tile(r["track_name"])}'
                f'<div class="meta">'
                f'<div class="name">{html.escape(str(r["track_name"]))}</div>'
                f'<div class="by">{html.escape(str(r["artists"]))}</div>'
                f'</div>'
                f'<div class="side">'
                f'<span class="genre">{html.escape(str(r["track_genre"]))}</span>'
                f'<div class="rank">#{rank}</div>'
                f'</div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

        st.markdown(
            '<div class="note">Matched on nine audio features, so '
            'similar-sounding songs can span very different genres.</div>',
            unsafe_allow_html=True,
        )
