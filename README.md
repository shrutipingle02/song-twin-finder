# 🎧 Song Twin Finder

Type a song you love, get five that feel like it.

**Live app → [song-twin-finder.streamlit.app](https://song-twin-finder.streamlit.app)**

---

## What it does

Search for any song. The app finds it, then returns the five songs in the dataset
that are closest to it in sound.

It does not look at the title, the artist or the genre. It only looks at how the
song actually sounds.

---

## How it works

Every song in the dataset comes with numbers describing its sound. This project
uses nine of them:

| Feature | What it measures |
|---|---|
| danceability | how easy it is to dance to |
| energy | intensity |
| valence | how happy it sounds |
| tempo | speed in BPM |
| acousticness | acoustic vs electronic |
| instrumentalness | whether it has vocals |
| speechiness | how much spoken word |
| liveness | recorded live or not |
| loudness | volume in decibels |

**Three steps:**

**1. Clean the data.** Drop songs with missing values, and remove duplicates —
the dataset lists the same song once per genre, so without this you get the same
track five times in your results.

**2. Put everything on the same scale.** This is the step that matters most.
Tempo runs to about 200, while danceability runs 0 to 1. Left alone, a 40 BPM gap
would count for forty times more than a complete flip in mood, and every
recommendation would just be "songs at a similar speed". `MinMaxScaler` squashes
all nine features into 0–1 so each one gets an equal vote.

**3. Compare.** Each song is now nine numbers — a point in nine-dimensional
space. Cosine similarity measures how close two points are, and the five closest
come back as your results.

---

## Run it locally

```bash
git clone https://github.com/shrutipingle02/song-twin-finder.git
cd song-twin-finder

pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

---

## Files

```
app.py                    the whole app — recommender and interface
songs.csv                 81,343 songs, cleaned
requirements.txt          four packages
.streamlit/config.toml    dark theme
```

---

## A note on the results

Matches are based on sound, not meaning. Two songs can share almost identical
danceability, energy and tempo and still belong to completely different genres —
so a rock seed can return a drum-and-bass track, and that is the model working as
designed rather than a bug.

Nine numbers capture a lot about how a song feels, but not everything. Treat the
results as a starting point.

---

## Built with

Python · pandas · NumPy · scikit-learn · Streamlit

**Data:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
on Hugging Face — roughly 114,000 songs, reduced to 81,343 after cleaning.

---

## Author

**Shruti Pingle**

[LinkedIn](https://www.linkedin.com/in/shruti-pingle-aa8034196) ·
[shrutipingle02@gmail.com](mailto:shrutipingle02@gmail.com)
