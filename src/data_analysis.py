import json
import pandas as pd

from utils import (
    plot_bpm_distribution,
    plot_danceability_distribution,
    plot_emotion_distribution,
    plot_genre_distribution,
    plot_key_distribution,
    plot_loudness_distribution,
    plot_vocal_instrumental_distribution,
    plot_major_minor_distribution,
    analyze_key_scale_pairs,
    plot_key_scale_distribution,
)

OUTPUT_PICKLE_PATH = "data/audio_features.pkl"
INPUT_JSON_PATH = "audio_features.json"


# Load the JSON data
def load_data(filename=INPUT_JSON_PATH):
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
    scales_temperley = []
    scales_krumhansl = []
    scales_edma = []
    scales_default = []
    vocal_instrumental = []
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
        effnet_features = features["Effnet-Discogs-features"]
        main_genre = effnet_features["main_genre"]
        sub_genre = effnet_features["sub_genre"]
        genres_first.append(main_genre)
        genres_concatenated.append(f"{main_genre}--{sub_genre}")

        # Extract BPMs
        bpm_avg = (features["Bpm"]["bpm_re"] + features["Bpm"]["bpm_cnn"]) / 2
        bpms.append(bpm_avg)

        # Extract Danceability
        danceabilities.append(effnet_features["danceability"])

        # Extract Keys
        keys = features["Key"]
        keys_temperley.append(keys["key_temperley"])
        keys_krumhansl.append(keys["key_krumhansl"])
        keys_edma.append(keys["key_edma"])
        scales_temperley.append(keys["scale_temperley"])
        scales_krumhansl.append(keys["scale_krumhansl"])
        scales_edma.append(keys["scale_edma"])
        scales_default.append(keys["scale"])

        # Extract Vocal/Instrumental
        vocal_instrumental.append(effnet_features["vocal-instrumental"])

        # Extract Loudness
        loudness_values.append(features["Loudness"])

        # Extract Valence/Arousal
        music_cnn_features = features["MSD-MusicCNN-features"]
        valence = music_cnn_features["valence"]
        arousal = music_cnn_features["arousal"]
        valence_arousal.append((valence, arousal))
        valence_list.append(valence)
        arousal_list.append(arousal)

        # Extract Embeddings
        discogs_effnet_embeddings.append(effnet_features["effnet-discogs-embeddings"])
        msd_musiccnn_embeddings.append(music_cnn_features["music-cnn-embeddings"])

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
            "Scale": scales_default,
            "Instrumentation": vocal_instrumental,
            "Loudness": loudness_values,
            "Valence": valence_list,
            "Arousal": arousal_list,
            "Discogs Effnet Embeddings": discogs_effnet_embeddings,
            "MSD MusicCNN Embeddings": msd_musiccnn_embeddings,
        }
    )

    df.to_pickle(OUTPUT_PICKLE_PATH)

    plot_vocal_instrumental_distribution(vocal_instrumental)
    plot_key_scale_distribution(
        keys_temperley,
        keys_krumhansl,
        keys_edma,
        scales_temperley,
        scales_krumhansl,
        scales_edma,
    )
    plot_bpm_distribution(bpms)
    plot_danceability_distribution(danceabilities)
    plot_genre_distribution(genres_first)
    plot_key_distribution(keys_temperley, keys_krumhansl, keys_edma)
    plot_major_minor_distribution(scales_default)
    plot_loudness_distribution(loudness_values)
    plot_emotion_distribution(valence_arousal)
    analyze_key_scale_pairs(
        keys_temperley,
        keys_krumhansl,
        keys_edma,
        scales_temperley,
        scales_krumhansl,
        scales_edma,
    )


if __name__ == "__main__":
    main()
