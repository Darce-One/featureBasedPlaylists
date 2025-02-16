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
scale_select = ['major', 'minor']
instrumentation_select = ['voice', 'instrumental']

# Initial ranges and session state setup
if "bpm_select_range" not in st.session_state:
    st.session_state.bpm_select_range = (115.0, 125.0)
if "danceability_select_range" not in st.session_state:
    st.session_state.danceability_select_range = (0.6, 1.0)
if "loudness_select_range" not in st.session_state:
    st.session_state.loudness_select_range = (-7.0, -6.0)
if "scale_select" not in st.session_state:
    st.session_state.scale_select = []
if "instrumentation_select" not in st.session_state:
    st.session_state.instrumentation_select = []
if "valence_select_range" not in st.session_state:
    st.session_state.valence_select_range = (4.0, 7.0)
if "arousal_select_range" not in st.session_state:
    st.session_state.arousal_select_range = (4.0, 7.0)
if "genre_select" not in st.session_state:
    st.session_state.genre_select = []
if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False


# Logic to check changes
def check_pre_run_changes():
    has_changes = False

    # Detect changes in sliders
    if st.session_state.bpm_select_range != bpm_select_range:
        st.session_state.bpm_select_range = bpm_select_range
        has_changes = True
    if st.session_state.danceability_select_range != danceability_select_range:
        st.session_state.danceability_select_range = danceability_select_range
        has_changes = True
    if st.session_state.loudness_select_range != loudness_select_range:
        st.session_state.loudness_select_range = loudness_select_range
        has_changes = True
    if st.session_state.valence_select_range != valence_select_range:
        st.session_state.valence_select_range = valence_select_range
        has_changes = True
    if st.session_state.arousal_select_range != arousal_select_range:
        st.session_state.arousal_select_range = arousal_select_range
        has_changes = True

    # Detect changes in selectors
    if st.session_state.genre_select != genre_select:
        st.session_state.genre_select = genre_select
        has_changes = True
    if st.session_state.scale_select != scale_select:
        st.session_state.scale_select = scale_select
        has_changes = True
    if st.session_state.instrumentation_select != instrumentation_select:
        st.session_state.instrumentation_select = instrumentation_select
        has_changes = True

    return has_changes


st.write("## Filter by features")
st.write("And their appropriate ranges")
features_options = ["BPM", "Danceability", "Loudness", "Scale", "Instrumentation"]
feature_select = st.multiselect("Select by features:", features_options)

if "BPM" in feature_select:
    bpm_select_range = st.slider(
        "Select tracks with a bpm within range:",
        value=st.session_state.bpm_select_range,
        min_value=63.0,
        max_value=183.0,
    )

if "Danceability" in feature_select:
    danceability_select_range = st.slider(
        "select tracks with a danceability within range:",
        value=st.session_state.danceability_select_range,
        min_value=0.0,
        max_value=1.0,
    )

if "Loudness" in feature_select:
    loudness_select_range = st.slider(
        "select tracks with a loudness within range:",
        value=st.session_state.loudness_select_range,
        min_value=-20.0,
        max_value=-4.0,
    )

if "Scale" in feature_select:
    scale_select = st.multiselect("select scales",
        ['major', 'minor'],
        default=st.session_state.scale_select)

if "Instrumentation" in feature_select:
    instrumentation_select = st.multiselect("select instrumentation",
        ['voice', 'instrumental'],
        default=st.session_state.instrumentation_select)


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

if scale_select:
    filtered_audio_analysis = filtered_audio_analysis[
        filtered_audio_analysis["Scale"].isin(scale_select)
    ]

if instrumentation_select:
    filtered_audio_analysis = filtered_audio_analysis[
        filtered_audio_analysis["Instrumentation"].isin(instrumentation_select)
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
genre_select = st.multiselect(
    "Select by genre:", possible_genres, default=st.session_state.genre_select
)


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
    value=st.session_state.valence_select_range,
)
arousal_select_range = st.slider(
    "select tracks with an arousal within range:",
    min_value=0.0,
    max_value=9.0,
    value=st.session_state.arousal_select_range,
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

# Reset the RUN button state if any pre-run filters have changed
if check_pre_run_changes():
    st.session_state.button_pressed = False


mp3s = []
# Button part!
if st.button("RUN") or st.session_state.button_pressed:
    st.session_state.button_pressed = True
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
        for idx, mp3 in enumerate(mp3s[:10]):
            st.write(f"\n {idx + 1}:")
            st.audio(mp3, format="audio/mp3", start_time=0)
    else:
        st.write("No tracks found based on the current filter criteria.")

    if mp3s:
        st.write("## Detailed Track Information")
        if "selected_index" not in st.session_state:
            st.session_state.selected_index = 1  # Default to the first track
        # Create a dictionary that maps integer indices to track file paths
        indexed_tracks = {i + 1: mp3 for i, mp3 in enumerate(mp3s[:10])}

        # Allow the user to select a track by its position number
        st.session_state.selected_index = st.selectbox(
            "Select a track number to view detailed features:",
            list(indexed_tracks.keys()),
            index=st.session_state.selected_index - 1,  # Adjust for zero-based index
        )

        # Find the file path for the selected track number
        selected_track = indexed_tracks[st.session_state.selected_index]

        # Find the row in the DataFrame corresponding to the selected track
        track_features = filtered_audio_analysis[
            filtered_audio_analysis["Path"] == selected_track
        ]

        # Display the features of the selected track
        if not track_features.empty:
            st.write(
                f"### Features for the selected track (Track #{st.session_state.selected_index}):"
            )
            st.write(track_features.T)  # Transpose to display features vertically
        else:
            st.write("Could not find the features for the selected track.")
else:
    st.write(
        "No tracks are available for detailed viewing. Please hit run, or adjust the filter criteria."
    )
