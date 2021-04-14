import glob
import numpy as np
import random
import time

from fastai.basics import *

from deepaudiobooktuner.utils.attention_mask import *
from deepaudiobooktuner.utils.top_k_top_p import *
from deepaudiobooktuner.utils.paths import path
from deepaudiobooktuner.music_generation.music_transformer.transform import *
from deepaudiobooktuner.music_generation.music_transformer.model import *
from deepaudiobooktuner.music_generation.music_transformer.learner import *


def loadMusicAssets(paths):
    music_data = load_data(paths["music_data"], "musicitem_data_save.pkl")
    music_model = music_model_learner(music_data, pretrained_path=paths["music_model"])
    return music_data, music_model


def fetchMidi(emotion, music_folder):
    folder = f"{music_folder}/{emotion}"
    songs_list = glob.glob(f"{folder}/*.mid")
    song = songs_list[random.randrange(len(songs_list))]

    return song


def generateMusicClips(paths, music_model, music_data, songs, music_emotions):
    # Generating music for each emotion
    for music_emotion in music_emotions:
        current_time = time.time()

        # Fetch a random song for the given emotion
        midi_file = path(fetchMidi(music_emotion, f"{paths['music_samples']}"))

        # Define the number of beats to be used from the seed song
        cuttoff_beat = 10
        item = MusicItem.from_file(midi_file, music_data.vocab)
        seed_item = item.trim_to_beat(cuttoff_beat)

        # Predict the next n words of the song
        pred, full = music_model.predict(
            seed_item,
            n_words=400,
            temperatures=(1.1, 0.4),
            min_bars=12,
            top_k=24,
            top_p=0.7,
        )

        # Append the prediction to the beats from the seed song
        full_song = seed_item.append(pred)

        print(
            f"----generated {music_emotion} clip. Time taken: {round(time.time()-current_time, 1)} s"
        )

        # Adding the song created for the emotion to a dictionary
        songs[music_emotion] = full_song

    return songs
