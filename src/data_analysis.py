import json
import pandas as pd

from utils import (
    plot_bpm_distribution,
    plot_danceability_distribution,
    plot_emotion_distribution,
    plot_genre_distribution,
    plot_key_distribution,
    plot_loudness_distribution,
)


# Load the JSON data
def load_data(filename="audio_features.json"):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def main():
    data = load_data()

    # Initialize the data structures
    track_names = []
    audio_files = []
    genres_first = []
    genres_concatenated = []
    bpms = []
    danceabilities = []
    keys_temperley = []
    keys_krumhansl = []
    keys_edma = []
    loudness_values = []
    valence_arousal = []
    valence_list = []
    arousal_list = []
    discogs_effnet_embeddings = []
    msd_musiccnn_embeddings = []

    # Populate the data structures
    for track, features in data.items():
        track_names.append(track)
        audio_files.append(features["audio_file"])
        # Extract genres
        if "Genre" in features:
            genres = features["Genre"]
            if genres:
                genres_first.append(genres[0])
                genres_concatenated.append("--".join(genres))

        # Extract BPMs
        if "Bpm" in features:
            bpm_avg = (features["Bpm"]["bpm_re"] + features["Bpm"]["bpm_cnn"]) / 2
            bpms.append(bpm_avg)

        # Extract Danceability
        if "Danceability" in features:
            danceabilities.append(features["Danceability"])

        # Extract Keys
        if "Key" in features:
            keys = features["Key"]
            keys_temperley.append(keys.get("key_temperley", "Unknown"))
            keys_krumhansl.append(keys.get("key_krumhansl", "Unknown"))
            keys_edma.append(keys.get("key_edma", "Unknown"))

        # Extract Loudness
        if "Loudness" in features:
            loudness_values.append(features["Loudness"])

        # Extract Valence/Arousal
        if "Emotion" in features:
            emotion = features["Emotion"]
            valence_arousal.append(
                (emotion.get("valence", 0), emotion.get("arousal", 0))
            )
            valence_list.append(emotion.get("valence", 0))
            arousal_list.append(emotion.get("arousal", 0))

        # Extract Embeddings
        discogs_effnet_embeddings.append(features.get("Discogs-Effnet-Embeddings", []))
        msd_musiccnn_embeddings.append(features.get("MSD-MusicCNN-Embeddings", []))

    df = pd.DataFrame(
        {
            "Track": track_names,
            "Path": audio_files,
            "Primary Genre": genres_first,
            "All Genres": genres_concatenated,
            "BPM": bpms,
            "Danceability": danceabilities,
            "Key Temperley": keys_temperley,
            "Key Krumhansl": keys_krumhansl,
            "Key Edma": keys_edma,
            "Loudness": loudness_values,
            "Valence": valence_list,
            "Arousal": arousal_list,
            "Discogs Effnet Embeddings": discogs_effnet_embeddings,
            "MSD MusicCNN Embeddings": msd_musiccnn_embeddings,
        }
    )

    df.to_pickle("part3/data/audio_features.pkl")

    plot_bpm_distribution(bpms)
    plot_danceability_distribution(danceabilities)
    plot_genre_distribution(genres_first)
    plot_key_distribution(keys_temperley, keys_krumhansl, keys_edma)
    plot_loudness_distribution(loudness_values)
    plot_emotion_distribution(valence_arousal)


if __name__ == "__main__":
    main()
