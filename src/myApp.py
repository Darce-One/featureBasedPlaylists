import os.path
import streamlit as st
import pandas
import plotly.graph_objects as go
import matplotlib.pyplot as plt


m3u_filepaths_file = "playlists/streamlit.m3u4"
ESSENTIA_ANALYSIS_PATH = "data/audio_features.pkl"


def load_essentia_analysis():
    return pandas.read_pickle(ESSENTIA_ANALYSIS_PATH)


def plot_genre_distribution(genres):
    plt.figure(figsize=(10, 5))
    plt.hist(
        genres,
        bins=len(set(genres)),
        alpha=0.7,
        color="b",
        label="Primary Genre",
    )
    plt.xticks(rotation=90)
    plt.title("Genre Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.legend()
    st.pyplot(plt)


st.write("# Audio analysis playlists")
st.write(f"Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.")
audio_analysis = load_essentia_analysis()
st.write("Loaded audio analysis for", len(audio_analysis), "tracks.")
st.write("Feature activation statistics:")
st.write(audio_analysis.describe())

bpm_select_range = (63.0, 183.0)
danceability_select_range = (0.0, 1.0)
loudness_select_range = (-20.0, -4.0)
arousal_select_range = (0.0, 9.0)
valence_select_range = (0.0, 9.0)

st.write("## Filter by features")
st.write("And their appropriate ranges")
features_float_types = ["BPM", "Danceability", "Loudness"]
feature_select = st.multiselect("Select by features:", features_float_types)

if "BPM" in feature_select:
    bpm_select_range = st.slider(
        "Select tracks with a bpm within range:",
        value=[115.0, 125.0],
        min_value=63.0,
        max_value=183.0,
    )

if "Danceability" in feature_select:
    danceability_select_range = st.slider(
        "select tracks with a danceability within range:",
        min_value=0.0,
        max_value=1.0,
        value=[0.6, 1.0],
    )

if "Loudness" in feature_select:
    loudness_select_range = st.slider(
        "select tracks with a loudness within range:",
        min_value=-20.0,
        max_value=-4.0,
        value=[-7.0, -6.0],
    )


# Filter the DataFrame based on selections
filtered_audio_analysis = audio_analysis[
    (audio_analysis["BPM"].between(bpm_select_range[0], bpm_select_range[1]))
    & (
        audio_analysis["Danceability"].between(
            danceability_select_range[0], danceability_select_range[1]
        )
    )
    & (
        audio_analysis["Loudness"].between(
            loudness_select_range[0], loudness_select_range[1]
        )
    )
]

st.write("## Filter by genre")
st.write("here is the distribution of genres present after initial filtering")
# Plot genre distribution before filtering
plot_genre_distribution(filtered_audio_analysis["Primary Genre"])

# Add genre filter:
possible_genres = [
    "Hip Hop",
    "Folk, World, & Country",
    "Rock",
    "Electronic",
    "Stage & Screen",
    "Reggae",
    "Non-Music",
    "Classical",
    "Pop",
    "Funk / Soul",
    "Blues",
    "Children's",
    "Jazz",
    "Latin",
    "Brass & Military",
]
genre_select = st.multiselect("Select by genre:", possible_genres)


if genre_select:
    filtered_audio_analysis = filtered_audio_analysis[
        filtered_audio_analysis["Primary Genre"].isin(genre_select)
    ]


if feature_select:
    # Show the distribution of activation values for the selected styles from the filtered data
    st.write("### description of filtered features")
    st.write(filtered_audio_analysis[feature_select].describe())

# Create a Plotly scatter plot for the 2D Valence-Arousal plane
valence_arousal_fig = go.Figure()

# Add a scatter trace for the filtered data
valence_arousal_fig.add_trace(
    go.Scatter(
        x=filtered_audio_analysis["Valence"],
        y=filtered_audio_analysis["Arousal"],
        mode="markers",
        name="Filtered Data Points",
    )
)

# Customize layout
valence_arousal_fig.update_layout(
    title="Valence - Arousal distribution", xaxis_title="Valence", yaxis_title="Arousal"
)

# Display the plot in Streamlit
st.write("## Filter by Emotion")
st.write("Here is a plot of the relevant songs in terms of their emotional dimensions")
st.plotly_chart(valence_arousal_fig)


valence_select_range = st.slider(
    "select tracks with a valence within range:",
    min_value=0.0,
    max_value=9.0,
    value=[4.0, 7.0],
)
arousal_select_range = st.slider(
    "select tracks with an arousal within range:",
    min_value=0.0,
    max_value=9.0,
    value=[4.0, 7.0],
)

filtered_audio_analysis = filtered_audio_analysis[
    (
        audio_analysis["Valence"].between(
            valence_select_range[0], valence_select_range[1]
        )
    )
    & (
        audio_analysis["Arousal"].between(
            arousal_select_range[0], arousal_select_range[1]
        )
    )
]

if st.button("RUN"):
    st.write("## 🔊 Results")

    # Retrieve paths from the filtered DataFrame
    if "Path" in filtered_audio_analysis.columns:
        mp3s = filtered_audio_analysis["Path"].tolist()
    else:
        st.write("Error: No 'Path' column found for audio files.")
        mp3s = []

    # Create the M3U8 playlist
    if mp3s:
        with open(m3u_filepaths_file, "w") as f:
            # Store relative mp3 paths in the playlist.
            mp3_paths = [os.path.join("..", mp3) for mp3 in mp3s]
            f.write("\n".join(mp3_paths))
        st.write(f"Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.")

        # Specify how many songs were added
        st.write(f"{len(mp3s)} songs were added to the playlist.")

        # Display audio previews for the first 10 tracks
        st.write("Audio previews for the first 10 results:")
        for mp3 in mp3s[:10]:
            st.audio(mp3, format="audio/mp3", start_time=0)
    else:
        st.write("No tracks found based on the current filter criteria.")
