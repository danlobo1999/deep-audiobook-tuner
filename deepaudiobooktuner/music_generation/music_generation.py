import glob
import numpy as np
import random
import time

from fastai.basics import *
from pydub import AudioSegment
from pydub.effects import normalize

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


def createSountrack(music_dict, emotion_list):

    song_seek = {"happy": 0, "neutral": 0, "sad": 0, "angry": 0}
    emo_distinct = ["angry", "happy", "neutral", "sad"]
    fade_time = 2000
    emo_dur = {}
    songs_list = {}
    songs_looped = {}

    # Finding duration of each emotion
    for x in emo_distinct:
        emo_dur[x] = emotion_list.count(x) * 30

    # Loading the mp3 songs and getting their duration
    for emotion in music_dict:
        temp = AudioSegment.from_mp3(music_dict[emotion])
        songs_list[emotion] = [temp, int(temp.duration_seconds)]

    # Looping the mp3 tracks for required duration
    for song in songs_list:
        dur_multiplier = math.ceil(emo_dur[song] / songs_list[song][1])
        songs_looped[song] = songs_list[song][0] * dur_multiplier

    # Duration of the first emotion in the soundtrack
    dur0 = song_seek[emotion_list[0]] * 1000
    dur1 = (song_seek[emotion_list[0]] + 30) * 1000

    song_seek[emotion_list[0]] = song_seek[emotion_list[0]] + 30

    # Appending duration of emotion 1 from songs looped to final track

    # if next emotion is different
    if emotion_list[0] != emotion_list[1]:
        final_track = songs_looped[emotion_list[0]][dur0:dur1].fade_out(fade_time)
    # if next emotion is same
    else:
        final_track = songs_looped[emotion_list[0]][dur0:dur1]

    # Appending durations of remaining emotions from songs looped to final track
    for i in range(1, len(emotion_list)):
        dur0 = song_seek[emotion_list[i]] * 1000
        dur1 = (song_seek[emotion_list[i]] + 30) * 1000
        song_seek[emotion_list[i]] = song_seek[emotion_list[i]] + 30

        # if last clip
        if i == (len(emotion_list) - 1):

            # if same as previous clip
            if emotion_list[i] == emotion_list[i - 1]:
                final_track = final_track + songs_looped[emotion_list[i]][
                    dur0:dur1
                ].fade_out(fade_time)

            # if different clip
            else:
                final_track = final_track + songs_looped[emotion_list[i]][
                    dur0:dur1
                ].fade_in(fade_time).fade_out(fade_time)

        # if not last clip
        else:

            # if same as prev and next
            if (
                emotion_list[i] == emotion_list[i - 1]
                and emotion_list[i] == emotion_list[i + 1]
            ):
                final_track = final_track + songs_looped[emotion_list[i]][dur0:dur1]

            # if same as prev only
            elif emotion_list[i] == emotion_list[i - 1]:
                final_track = final_track + songs_looped[emotion_list[i]][
                    dur0:dur1
                ].fade_out(fade_time)

            # if same as next only
            elif emotion_list[i] == emotion_list[i + 1]:
                final_track = final_track + songs_looped[emotion_list[i]][
                    dur0:dur1
                ].fade_in(fade_time)

            # if not same as prev or next
            else:
                final_track = final_track + songs_looped[emotion_list[i]][
                    dur0:dur1
                ].fade_in(fade_time).fade_out(fade_time)

    # Normalizing the final soundtrack
    final_track = normalize(final_track)

    final_track = final_track - 15

    return final_track


def overlaySoundtrack(audiobook_path, final_track):
    audio_book = AudioSegment.from_mp3(audiobook_path) + 5
    mixed = audio_book.overlay(final_track)

    return mixed

