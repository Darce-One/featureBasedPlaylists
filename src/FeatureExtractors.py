import json
from typing import Any, Dict, Tuple

import essentia
import essentia.standard as es
import numpy as np
from essentia.standard import (
    TensorflowPredict2D,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredictMusiCNN,
)


class FeatureExtractor:
    """Base class for a feature"""

    feature_name = "base_feature"

    def __init__(self):
        self.mono_mixer = es.MonoMixer()
        self.resampler = es.Resample(inputSampleRate=44100, outputSampleRate=16000)

    def _convert_audio(self, audio):
        """
        Base method for converting the audio to the appropriate
        format for extraction
        """
        return audio

    @staticmethod
    def _get_top_classes(predictions, labels, num_top):
        predictions = np.array(predictions)

        # Get indices of the top three predictions (sorted from high to low)
        top_indices = predictions.argsort()[-num_top:][::-1]

        # Map indices to class labels
        top_classes = [labels[i] for i in top_indices]

        return top_classes

    def extract(self, audio: np.ndarray):
        """
        Base method for extracting features. expects stereo audio
        sampled at 44100 kHz
        """
        audio = self._convert_audio(audio)
        raise NotImplementedError


class BpmExtractor(FeatureExtractor):
    "Extracts the bpm using two different methods"

    feature_name = "Bpm"
    _graph_filename = "models/deepsquare-k16-3.pb"

    def __init__(self):
        super().__init__()
        self.tempo_cnn = es.TempoCNN(graphFilename=self._graph_filename)
        self.resampler.configure(inputSampleRate=44100, outputSampleRate=11025)
        self.rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

    def _convert_audio(self, audio):
        return self.mono_mixer(audio, 2)  # returns mono_44100

    def extract(self, audio) -> Dict[str, float]:
        """
        extracts the bpm of an audio file using rhythmExtractor2013 and TempoCNN.

        parameters:
            audio: sampled at 44100, MONO

        returns:
            bpm (Dict)
        """
        audio = self._convert_audio(audio)
        bpm_re = self.rhythm_extractor(audio)[0]
        resampled_audio = self.resampler(audio)
        bpm_cnn = self.tempo_cnn(resampled_audio)[0]
        return {"bpm_re": bpm_re, "bpm_cnn": bpm_cnn}


class KeyExtractor(FeatureExtractor):
    feature_name = "Key"

    def __init__(self):
        super().__init__()
        self.key_extractor_temperley = es.KeyExtractor(
            profileType="temperley", sampleRate=44100
        )
        self.key_extractor_krumhansl = es.KeyExtractor(
            profileType="krumhansl", sampleRate=44100
        )
        self.key_extractor_edma = es.KeyExtractor(profileType="edma", sampleRate=44100)

    def _convert_audio(self, audio):
        return self.mono_mixer(audio, 2)  # returns mono_44100

    def extract(self, audio) -> Dict[str, Any]:
        """
        extracts the key of the audio using KeyExtractor.
        algorithm resamples at 44100, MONO.
        parameters:
            audio: sampled at 44100, STEREO

        returns:
            extracted_keys (Dict)
        """
        audio = self._convert_audio(audio)
        key_temperley, scale_temperley, _ = self.key_extractor_temperley(audio)
        key_krumhansl, scale_krumhansl, _ = self.key_extractor_krumhansl(audio)
        key_edma, scale_edma, _ = self.key_extractor_edma(audio)

        # Major or minor?
        scale_counts = {"major": 0, "minor": 0}
        scales = [scale_temperley, scale_krumhansl, scale_edma]

        for scale in scales:
            if scale in scale_counts:
                scale_counts[scale] += 1

        majority_scale = max(scale_counts, key=scale_counts.get)

        return {
            "key_temperley": key_temperley,
            "key_krumhansl": key_krumhansl,
            "key_edma": key_edma,
            "scale_temperley": scale_temperley,
            "scale_krumhansl": scale_krumhansl,
            "scale_edma": scale_edma,
            "scale": majority_scale,
        }


class LoudnessExtactor(FeatureExtractor):
    feature_name = "Loudness"

    def __init__(self):
        self.loudness_meter = es.LoudnessEBUR128()

    def extract(self, audio) -> float:
        """
        extracts the loudness using LoudnessEBUR128.

        parameters:
            audio: sampled at 44100, STEREO

        returns:
            integrated_loudness
        """
        integrated_loudness = self.loudness_meter(audio)[2]
        return integrated_loudness


class GenreExtractor(FeatureExtractor):
    feature_name = "Genre"

    def __init__(self):
        super().__init__()
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename="models/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )
        self.model = TensorflowPredict2D(
            graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )

        with open("models/genre_discogs400-discogs-effnet-1.json", "r") as file:
            # Load the JSON data
            self.labels = json.load(file)["classes"]

    def _convert_audio(self, audio):
        return self.resampler(self.mono_mixer(audio, 2))  # returns mono_16000

    @staticmethod
    def _genre_formatting(label: str) -> Tuple[str, str]:
        # Check if the marker exists in the string
        if "---" in label:
            # Split the string at the first occurrence of the marker
            parts = label.split("---", 1)
            # Return the two parts as a tuple
            return parts[0], parts[1]
        else:
            # If the marker is not found, return the original string and an empty string
            return label, ""

    def extract(self, audio) -> Tuple[str, str]:
        """
        extracts the genre of the audio using Effnet Discogs.
        algorithm resamples to 16000, Mono

        Parameters:
            audio: sampled at 44100, STEREO

        Returns:
            main_genre, sub_genre
        """
        audio = self._convert_audio(audio)
        embeddings = self.embedding_model(audio)
        predictions = self.model(embeddings).mean(axis=0)

        genre = self._get_top_classes(predictions, self.labels, 1)[0]
        return self._genre_formatting(genre)


class DanceabilityExtractor(FeatureExtractor):
    feature_name = "Danceability"

    def __init__(self):
        super().__init__()
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
        )
        self.model = TensorflowPredict2D(
            graphFilename="models/danceability-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        with open("models/danceability-discogs-effnet-1.json", "r") as file:
            # Load the JSON data
            self.labels = json.load(file)["classes"]

    def _convert_audio(self, audio):
        return self.resampler(self.mono_mixer(audio, 2))  # returns mono_16000

    def extract(self, audio) -> float:
        """
        extracts the danceability of the audio file.
        algorithm resamples to 16000, MONO

        parameters:
            audio: sampled at 44100, STEREO

        return:
            danceability: probability of being danceable
        """
        audio = self._convert_audio(audio)
        embeddings = self.embedding_model(audio)
        danceability = self.model(embeddings).mean(axis=0)[0]
        return float(danceability)


class EmotionExtractor(FeatureExtractor):
    feature_name = "Emotion"

    def __init__(self):
        super().__init__()
        self.embedding_model = TensorflowPredictMusiCNN(
            graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd"
        )
        self.model = TensorflowPredict2D(
            graphFilename="models/emomusic-msd-musicnn-2.pb", output="model/Identity"
        )

    @staticmethod
    def _get_arousal_valence(predictions):
        return {"valence": float(predictions[0]), "arousal": float(predictions[1])}

    def _convert_audio(self, audio):
        return self.resampler(self.mono_mixer(audio, 2))  # returns mono_16000

    def extract(self, audio) -> Dict[str, float]:
        """
        extracts Valence and arousal from audio track.
        algorithm resamples to 16000, mono.

        parameters:
            audio: sampled at 44100, stereo

        returns: Valence and arousal
        """
        audio = self._convert_audio(audio)
        embeddings = self.embedding_model(audio)
        predictions = self.model(embeddings).mean(axis=0)
        return self._get_arousal_valence(predictions)


class UnifiedMSDMusicCNNExtractor(FeatureExtractor):
    """Unified class using MSD-MusicCNN model for emotion and embeddings extraction"""

    feature_name = "MSD-MusicCNN-features"

    def __init__(self):
        super().__init__()
        self.embedding_model = TensorflowPredictMusiCNN(
            graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd"
        )
        self.emotion_model = TensorflowPredict2D(
            graphFilename="models/emomusic-msd-musicnn-2.pb", output="model/Identity"
        )

    def _convert_audio(self, audio):
        return self.resampler(self.mono_mixer(audio, 2))  # returns mono_16000

    def extract_embeddings(self, audio) -> np.ndarray:
        """Extract embeddings once"""
        audio = self._convert_audio(audio)
        return self.embedding_model(audio)

    @staticmethod
    def _get_arousal_valence(predictions):
        return {"valence": float(predictions[0]), "arousal": float(predictions[1])}

    def extract_emotion(self, embeddings) -> Dict[str, float]:
        """
        Extracts Valence and arousal from audio track based on the embeddings.
        Returns: Valence and arousal
        """
        predictions = self.emotion_model(embeddings).mean(axis=0)
        return self._get_arousal_valence(predictions)

    def extract(self, audio) -> Dict[str, Any]:
        """Unified extraction"""
        embeddings = self.extract_embeddings(audio)
        # Extract emotion features
        emotion = self.extract_emotion(embeddings)

        # Return both embeddings and emotion
        return {"music-cnn-embeddings": embeddings.mean(axis=0).tolist(), **emotion}


class UnifiedEffnetDiscogsExtractor(FeatureExtractor):
    """Unified class using Effnet Discogs model"""

    feature_name = "Effnet-Discogs-features"

    def __init__(self):
        super().__init__()
        self.embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1"
        )
        self.genre_model = es.TensorflowPredict2D(
            graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        with open("models/genre_discogs400-discogs-effnet-1.json", "r") as file:
            self.genre_labels = json.load(file)["classes"]

        self.danceability_model = es.TensorflowPredict2D(
            graphFilename="models/danceability-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        with open("models/danceability-discogs-effnet-1.json", "r") as file:
            self.danceability_labels = json.load(file)["classes"]

        self.voice_model = es.TensorflowPredict2D(
            graphFilename="models/voice_instrumental-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        with open("models/voice_instrumental-discogs-effnet-1.json", "r") as file:
            self.voice_labels = json.load(file)["classes"]

    def extract_embeddings(self, audio) -> np.ndarray:
        """Extract embeddings once"""
        audio = self._convert_audio(audio)
        return self.embedding_model(audio)

    def _convert_audio(self, audio):
        return self.resampler(self.mono_mixer(audio, 2))  # returns mono_16000

    @staticmethod
    def _get_top_classes(predictions: np.ndarray, labels: list, num_top: int) -> list:
        predictions = np.array(predictions)
        top_indices = predictions.argsort()[-num_top:][::-1]
        return [labels[i] for i in top_indices]

    @staticmethod
    def _genre_formatting(label: str) -> Tuple[str, str]:
        if "---" in label:
            parts = label.split("---", 1)
            return parts[0], parts[1]
        return label, ""

    def extract(self, audio) -> Dict[str, Any]:
        embeddings = self.extract_embeddings(audio)
        genre_predictions = self.genre_model(embeddings).mean(axis=0)
        genre = self._get_top_classes(genre_predictions, self.genre_labels, 1)[0]
        main_genre, sub_genre = self._genre_formatting(genre)

        vocal_instrumental_prediction = int(
            np.round(self.voice_model(embeddings).mean(axis=0))[1]
        )
        vocal_instrumental = self.voice_labels[vocal_instrumental_prediction]

        danceability = self.danceability_model(embeddings).mean(axis=0)[0]

        embeddings = embeddings.mean(axis=0).tolist()

        return {
            "main_genre": main_genre,
            "sub_genre": sub_genre,
            "danceability": float(danceability),
            "vocal-instrumental": vocal_instrumental,
            "effnet-discogs-embeddings": embeddings,
        }
