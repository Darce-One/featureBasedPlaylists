import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


# Plot a histogram for genre counts
def plot_genre_distribution(genres_first):
    genre_counts = {genre: genres_first.count(genre) for genre in set(genres_first)}

    # Sort genres by count for better display
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    genres, counts = zip(*sorted_genres)

    plt.figure(figsize=(10, 8))
    plt.barh(genres, counts, color="b", alpha=0.7)
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.title("Genre Distribution")
    plt.tight_layout()
    plt.show()


def plot_vocal_instrumental_distribution(vocal_instrumental):
    # Count the occurrence of each category
    category_counts = {
        category: vocal_instrumental.count(category)
        for category in set(vocal_instrumental)
    }
    categories, counts = zip(*category_counts.items())

    # Plot as a bar chart
    plt.figure(figsize=(10, 8))
    plt.bar(categories, counts, color="b", alpha=0.7)
    plt.title("Vocal - Instrumental Distribution")
    plt.xlabel("Instrumentation")
    plt.ylabel("Count")
    plt.show()


def plot_major_minor_distribution(scales):
    # Count the occurrence of each scale type
    scale_counts = {scale: scales.count(scale) for scale in set(scales)}
    scale_types, counts = zip(*scale_counts.items())

    # Plot as a bar chart
    plt.figure(figsize=(10, 8))
    plt.bar(scale_types, counts, color="b", alpha=0.7)
    plt.title("Scale Distribution")
    plt.xlabel("Scale")
    plt.ylabel("Count")
    plt.show()


# Plot a histogram for BPM distribution in bins of 5
def plot_bpm_distribution(bpms):
    plt.figure(figsize=(10, 8))
    plt.hist(
        bpms,
        bins=np.arange(min(bpms), max(bpms) + 5, 5),
        alpha=0.7,
        color="g",
        label="BPM",
    )
    plt.title("BPM Distribution")
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# Plot a histogram for genre concatenated (genre--style)
def plot_genre_style_distribution(genres_concatenated):
    plt.figure(figsize=(10, 8))
    plt.hist(
        genres_concatenated,
        bins=len(set(genres_concatenated)),
        alpha=0.7,
        color="c",
        label="Genre--Style",
    )
    plt.xticks(rotation=30)
    plt.title("Genre--Style Distribution")
    plt.xlabel("Genre--Style")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Plot a histogram for danceability
def plot_danceability_distribution(danceabilities):
    plt.figure(figsize=(10, 8))
    plt.hist(danceabilities, bins=20, alpha=0.7, color="m", label="Danceability")
    plt.title("Danceability Distribution")
    plt.xlabel("Danceability")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# plot histrogram for song keys
def plot_key_distribution(keys_temperley, keys_krumhansl, keys_edma):
    chromatic_keys = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

    # Prepare data for plotting
    def count_keys(keys, chromatic_keys):
        return [keys.count(k) for k in chromatic_keys]

    temperley_counts = count_keys(keys_temperley, chromatic_keys)
    krumhansl_counts = count_keys(keys_krumhansl, chromatic_keys)
    edma_counts = count_keys(keys_edma, chromatic_keys)

    x = np.arange(len(chromatic_keys))

    # Bar width for each group
    bar_width = 0.25

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.bar(
        x - bar_width,
        temperley_counts,
        width=bar_width,
        color="r",
        label="Temperley",
        edgecolor="black",
    )
    plt.bar(
        x,
        krumhansl_counts,
        width=bar_width,
        color="g",
        label="Krumhansl",
        edgecolor="black",
    )
    plt.bar(
        x + bar_width,
        edma_counts,
        width=bar_width,
        color="b",
        label="EDMA",
        edgecolor="black",
    )

    # Setup the labels and legend
    plt.xticks(x, chromatic_keys)
    plt.title("Key Distribution Comparison")
    plt.xlabel("Key")
    plt.ylabel("Count")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# Plot a histogram for loudness
def plot_loudness_distribution(loudness_values):
    plt.figure(figsize=(10, 8))
    plt.hist(loudness_values, bins=20, alpha=0.7, color="y", label="Loudness")
    plt.title("Loudness Distribution")
    plt.xlabel("Loudness (dB)")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


# Plot a 2D histogram for valence/arousal
def plot_emotion_distribution(valence_arousal):
    valence_values = [v[0] for v in valence_arousal]
    arousal_values = [v[1] for v in valence_arousal]

    plt.figure(figsize=(10, 8))
    plt.hist2d(valence_values, arousal_values, bins=[20, 20], cmap="Blues")
    plt.colorbar(label="Count")
    plt.title("Valence/Arousal 2D Distribution")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.show()


def analyze_key_scale_pairs(
    keys_temperley,
    keys_krumhansl,
    keys_edma,
    scales_temperley,
    scales_krumhansl,
    scales_edma,
):
    # Define the categories
    categories = {
        "all_match": 0,
        "temperley_krumhansl_match_only": 0,
        "temperley_edma_match_only": 0,
        "krumhansl_edma_match_only": 0,
        "no_match": 0,
    }

    # Iterate through the key-scale data
    for kt, kk, ke, st, sk, se in zip(
        keys_temperley,
        keys_krumhansl,
        keys_edma,
        scales_temperley,
        scales_krumhansl,
        scales_edma,
    ):
        # Check for matches
        tk_match = (kt == kk) and (st == sk)
        te_match = (kt == ke) and (st == se)
        ke_match = (kk == ke) and (sk == se)

        if tk_match and te_match and ke_match:
            categories["all_match"] += 1
        elif tk_match and not te_match and not ke_match:
            categories["temperley_krumhansl_match_only"] += 1
        elif te_match and not tk_match and not ke_match:
            categories["temperley_edma_match_only"] += 1
        elif ke_match and not tk_match and not te_match:
            categories["krumhansl_edma_match_only"] += 1
        else:
            categories["no_match"] += 1

    # Prepare data for plotting
    categories_labels = list(categories.keys())
    categories_values = list(categories.values())

    # Plot the results as a horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(categories_labels, categories_values, color="skyblue", edgecolor="black")
    plt.xlabel("Count")
    plt.ylabel("Match Type")
    plt.title("Key-Scale Match Analysis")
    plt.tight_layout()
    plt.show()


def plot_key_scale_distribution(
    keys_temperley,
    keys_krumhansl,
    keys_edma,
    scales_temperley,
    scales_krumhansl,
    scales_edma,
):
    chromatic_keys = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    scales = ["major", "minor"]  # Assuming scales are simplified to major and minor

    # Combine each key with each possible scale
    key_scale_profiles = [
        f"{key} {scale}" for key in chromatic_keys for scale in scales
    ]

    # Count occurrences for each key-scale combination
    def count_key_scales(keys, scales, key_scale_profiles):
        key_scale_count = {profile: 0 for profile in key_scale_profiles}

        for key, scale in zip(keys, scales):
            profile = f"{key} {scale}"
            if profile in key_scale_count:
                key_scale_count[profile] += 1

        return key_scale_count

    # Counting for each method
    temperley_counts = count_key_scales(
        keys_temperley, scales_temperley, key_scale_profiles
    )
    krumhansl_counts = count_key_scales(
        keys_krumhansl, scales_krumhansl, key_scale_profiles
    )
    edma_counts = count_key_scales(keys_edma, scales_edma, key_scale_profiles)

    # Prepare the plot
    x = np.arange(len(key_scale_profiles))  # the label locations
    bar_width = 0.25  # Width of bars

    # Create the figure and subplot
    plt.figure(figsize=(10, 8))

    # Plot each method
    plt.bar(
        x - bar_width,
        temperley_counts.values(),
        width=bar_width,
        color="r",
        label="Temperley",
        edgecolor="black",
    )
    plt.bar(
        x,
        krumhansl_counts.values(),
        width=bar_width,
        color="g",
        label="Krumhansl",
        edgecolor="black",
    )
    plt.bar(
        x + bar_width,
        edma_counts.values(),
        width=bar_width,
        color="b",
        label="EDMA",
        edgecolor="black",
    )

    # Add some text for labels, title and custom x-axis tick labels
    plt.xlabel("Key-Scale")
    plt.ylabel("Count")
    plt.title("Key-Scale Distribution Comparison")
    plt.xticks(ticks=x, labels=key_scale_profiles, rotation=30)
    plt.legend(loc="upper right")

    # Adjust plot to fit labels
    plt.tight_layout()
    plt.show()


def get_top_n_embedding_indices(
    dataframe, reference_vector, embedding_column="embedding", top_n=10
):
    # Validate that reference_vector is a 1D numpy array
    if not isinstance(reference_vector, np.ndarray) or reference_vector.ndim != 1:
        raise ValueError("reference_vector should be a 1D numpy array")

    # Convert the embeddings from the DataFrame into a NumPy array for efficient computation
    embeddings = np.array(dataframe[embedding_column].tolist())

    # Calculate the cosine distances
    distances = cosine_distances(embeddings, reference_vector.reshape(1, -1)).flatten()

    # Get the indices of the top N smallest distances
    top_n_indices = np.argsort(distances)[:top_n]

    # Convert the integer indices to the DataFrame's index
    closest_indices = dataframe.index[top_n_indices]

    return closest_indices
