import os
import sys
import glob
from pprint import pprint as pp
import numpy as np

sys.path.append(os.path.abspath("../"))

from deepaudiobooktuner.sentiment_analysis.audio_analysis import (
    load_audio_assets,
    featureExtraction,
    analyzeAudio,
)
from deepaudiobooktuner.utils.file_processing import segmentAudioFile, alphanum_key


class TestAudioAnalysis:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.audio_classes = None

    def test_load_assets(self):
        self.model, self.scaler, self.audio_classes = load_audio_assets(
            model_path="../assets/audio_sentiment_data_v2/models/hyperband_tuned_model_19_april_[0.3624165654182434, 0.8720930218696594]",
            pickles_path="../assets/audio_sentiment_data_v2/pickles",
        )
        print(self.model.summary())
        print(self.audio_classes.classes_)
        print(self.scaler)

    def test_model(self, file):
        audio_emotions = analyzeAudio(
            file_name=file, model=self.model, scaler=self.scaler
        )
        raw_emotions = np.copy(audio_emotions)

        audio_emotions = audio_emotions.argmax()
        audio_emotions = audio_emotions.astype(int).flatten()
        final_emotion = self.audio_classes.inverse_transform((audio_emotions))
        return [final_emotion[0], raw_emotions]

    def test_model_bias(self, file):
        audio_emotions = analyzeAudio(
            file_name=file, model=self.model, scaler=self.scaler
        )
        audio_emotions[3] *= 0.5
        audio_emotions[2] *= 1.5
        biased_emotions = np.copy(audio_emotions)

        audio_emotions = audio_emotions.argmax()
        audio_emotions = audio_emotions.astype(int).flatten()
        final_emotion = self.audio_classes.inverse_transform((audio_emotions))
        return [final_emotion[0], biased_emotions]


def main(audiobook_path):
    emotions = []
    tester = TestAudioAnalysis()
    tester.test_load_assets()

    segmentAudioFile(
        file_name="test_book", file_path=audiobook_path, save_path="clips/"
    )

    files = glob.glob("clips/*.wav")
    files.sort(key=alphanum_key)
    pp(files)

    for file in files:
        print(file)
        emotions.append(
            [file, tester.test_model(file=file), tester.test_model_bias(file=file)]
        )

    pp(emotions)


if __name__ == "__main__":
    main(
        r"D:\Projects\BEProject\deep-audiobook-tuner\assets\audiobooks\grimms-fairy-tales-050-the-golden-goose.3147.wav"
    )
