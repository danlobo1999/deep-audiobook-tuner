from glob import glob
import os
import shutil
import time

from deepaudiobooktuner.utils.paths import createDir
from deepaudiobooktuner.utils.load_assets import loadAssets
from deepaudiobooktuner.utils.file_processing import (
    convertToWav,
    segmentAudioFile,
    convertToMp3,
    saveMusicClips,
)
from deepaudiobooktuner.sentiment_analysis.text_analysis import analyzeText
from deepaudiobooktuner.sentiment_analysis.audio_analysis import analyzeAudio
from deepaudiobooktuner.music_generation.music_generation import generateMusicClips


class deepAudiobookTuner:
    def __init__(self, audiobook_path):
        self.audiobook_path = audiobook_path
        self.songs = {}  # Dictionary to store songs
        self.transcriptions = (
            []
        )  # List to store transcriptions of all the segmented clips
        self.emotions = []  # List to store emotions of all the segmented clips

        # Creating a temperory directory to store the segmented audiobook clips and generated music clips
        print("\nCreating temporary directory.")
        self.file_name, self.paths = createDir(audiobook_path)

        # Loading assets.
        print("\nLoading assets.")
        self.assets = loadAssets(self.paths)

        # Converting the mp3 file to a wav file
        print("\nConverting mp3 to wav")
        self.wav_file_path = convertToWav(
            file_name=self.file_name,
            file_path=self.audiobook_path,
            save_path=self.paths["wav_save_path"],
        )

        # Segmenting the audio file into 30 second clips
        print("\nSegmenting audiobook")
        segmentAudioFile(
            file_name=self.file_name,
            file_path=self.wav_file_path,
            save_path=self.paths["clips_save_path"],
        )

    def analyzeSentiments(self):
        print("\n\nPerforming sentiment analysis")
        sentiment_analysis_time = time.time()

        for i, file_name in enumerate(glob(f'{self.paths["clips_save_path"]}/*.wav')):
            clip_time = time.time()
            print(f"\nProcessing clip {i+1}:")

            # Performing text sentiment analysis
            print("----Text sentiment analysis")
            text_emotions, transcription = analyzeText(
                file_name=file_name,
                stt=self.assets["stt"],
                predictor=self.assets["text_predictor"],
            )

            # Performing text sentiment analysis
            print("----Audio sentiment analysis")
            audio_emotions = analyzeAudio(
                file_name=file_name,
                model=self.assets["audio_model"],
                scaler=self.assets["audio_scaler"],
            )

            # Taking the average of text and audio emotions
            print("----Predicting final emotion")
            weighted_emotions = text_emotions * 0.8 + audio_emotions * 0.2

            # Picking the dominant emotion and labelling it
            weighted_emotions = weighted_emotions.argmax()
            weighted_emotions = weighted_emotions.astype(int).flatten()
            final_emotion = self.assets["audio_classes"].inverse_transform(
                (weighted_emotions)
            )

            self.transcriptions.append(transcription)
            self.emotions.append(final_emotion)

            print(
                f"----Clip {i+1} processed. Time taken: {round(time.time() - clip_time, 1)} s"
            )

        print(
            f"----\nSentiment Analysis Complete. Time taken: {round(time.time() - sentiment_analysis_time, 1)} s"
        )

    def generateMusic(self, music_emotions=["Angry", "Happy", "Neutral", "Sad"]):
        # Generating music clips
        print("\n\nGenerating music")
        music_generation_time = time.time()

        self.songs = generateMusicClips(
            music_emotions=music_emotions,
            paths=self.paths,
            music_model=self.assets["music_model"],
            music_data=self.assets["music_data"],
            songs=self.songs,
        )

        saveMusicClips(
            music_emotions=music_emotions, songs=self.songs, paths=self.paths
        )

        print(
            f"----\nMusic Generation Complete. Time taken: {round(time.time() - music_generation_time, 1)} s"
        )

    def deleteTempDirectory(self):
        try:
            shutil.rmtree(self.paths["wav_save_path"])
        except OSError as e:
            print("Error: %s : %s" % (self.paths["wav_save_path"], e.strerror))

