import json
import os
from pathlib import Path
from typing import List

import essentia
import essentia.standard as es
import numpy as np
from tqdm import tqdm

from FeatureExtractors import (
    BpmExtractor,
    DanceabilityExtractor,
    UnifiedMSDMusicCNNExtractor,
    EmotionExtractor,
    FeatureExtractor,
    GenreExtractor,
    KeyExtractor,
    LoudnessExtactor,
    UnifiedEffnetDiscogsExtractor,
)

DATASET_PATH = "MusAV/"
# DATASET_PATH = "test-data/"
JSON_SAVE_PATH = "audio_features.json"
# JSON_SAVE_PATH = "audio_features_test.json"


class FeatureCalculator:
    """
    class responsible for loading in an audio file, performing
    the feature extractions and returning a dictionary of the features.
    """

    def __init__(self, extractors: List[FeatureExtractor]):
        self.audio_loader = es.AudioLoader()
        self.extractors = extractors

    def get_audio(self, audio_file_path: str) -> np.ndarray:
        self.audio_loader.configure(filename=audio_file_path)
        stereo_44100 = self.audio_loader()[0]
        return stereo_44100

    def extract_features(self, audio_file_path: str):
        stereo_audio = self.get_audio(audio_file_path)
        feature_dict = {"audio_file": audio_file_path}
        for extractor in self.extractors:
            # Extract feature using the extractor's extract method
            feature_result = extractor.extract(stereo_audio)

            # Store the result in feature_dict with feature_name as the key
            feature_dict[extractor.feature_name] = feature_result

        return feature_dict


class AudioFeatureDatabase:
    def __init__(self, json_path: str):
        """Initializes the database with a path to the JSON file."""
        self.json_path = json_path
        self.data = self.load_data(json_path)

    @staticmethod
    def get_mp3_files(dataset_path: str) -> List[Path]:
        dataset_path_object = Path(dataset_path)
        mp3_files = list(dataset_path_object.rglob("*.mp3"))
        return mp3_files

    def process_dataset(self, feature_calculator: FeatureCalculator, dataset_path: str):
        self.feature_calculator = feature_calculator
        self.mp3_paths = self.get_mp3_files(dataset_path)

        for mp3_file in tqdm(self.mp3_paths):
            features = self.feature_calculator.extract_features(
                audio_file_path=str(mp3_file.absolute())
            )
            self.add_audio_features(str(mp3_file.name), features)

    def load_data(self, json_path):
        """Loads data from the JSON file if it exists, otherwise returns an empty dictionary."""
        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                return json.load(file)
        else:
            return {}

    def save_data(self):
        """Saves the current state of the data dictionary to the JSON file."""
        with open(self.json_path, "w") as file:
            json.dump(self.data, file, indent=2)

    def add_audio_features(self, audio_id, features):
        """Adds or updates the entry for an audio file identified by audio_id."""
        self.data[audio_id] = features
        self.save_data()

    def get_audio_features(self, audio_id):
        """Retrieves features for a given audio file."""
        return self.data.get(audio_id, None)

    def remove_audio_features(self, audio_id):
        """Removes the entry for a given audio file."""
        if audio_id in self.data:
            del self.data[audio_id]
            self.save_data()


def main():
    db = AudioFeatureDatabase(json_path=JSON_SAVE_PATH)

    extractors = [
        BpmExtractor(),
        KeyExtractor(),
        LoudnessExtactor(),
        UnifiedEffnetDiscogsExtractor(),
        UnifiedMSDMusicCNNExtractor(),
    ]

    fc = FeatureCalculator(extractors)

    db.process_dataset(feature_calculator=fc, dataset_path=DATASET_PATH)


if __name__ == "__main__":
    main()
