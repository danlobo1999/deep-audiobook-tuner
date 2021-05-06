import subprocess
import os
import time

import librosa
from midi2audio import FluidSynth
from pydub import AudioSegment
import soundfile as sf
import re


fs_font_path = "../../assets/music_generation_data/soundfont/GeneralUser GS v1.471.sf2"


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def convertToWav(file_name, file_path, save_path):
    current_time = time.time()

    wav_file_name = f"{save_path}/{file_name}.wav"
    subprocess.call(["ffmpeg", "-i", file_path, wav_file_name])

    print(f"----Converted to wav. Time taken: {round(time.time()-current_time, 1)} s")

    return wav_file_name


def convertToMp3(file_name, file_path, save_path):
    # Defining a Fluidsynth instance
    fs = FluidSynth(fs_font_path)

    wav_save_name = f"{save_path}/{file_name}.wav"
    mp3_save_name = f"{save_path}/{file_name}.mp3"

    # Converting and saving the midi file as a wav file
    fs.midi_to_audio(file_path, wav_save_name)

    # Loading the wav file, converting and saving it as a mp3 file
    AudioSegment.from_wav(wav_save_name).export(mp3_save_name, format="mp3")

    # Removing the wav file
    try:
        os.remove(wav_save_name)
    except:
        pass

    return mp3_save_name


def saveMusicClips(music_emotions, songs, paths):
    music_dict = {}

    # Saving the music clips
    for song_emotion in music_emotions:

        # Saving the Music21 MusicItem as a midi file
        out_file_name = f'{paths["music_clips_save_path"]}/{song_emotion}.midi'
        songs[song_emotion].stream.write("midi", out_file_name)

        # Converting the saved midi file to a mp3 file
        mp3_save_name = convertToMp3(
            file_name=song_emotion,
            file_path=out_file_name,
            save_path=paths["music_clips_save_path"],
        )

        # Removing the midi file
        try:
            os.remove(out_file_name)
        except:
            pass

        music_dict[song_emotion.lower()] = mp3_save_name

    return music_dict


def segmentAudioFile(file_name, file_path, save_path):
    current_time = time.time()

    audio, sr = librosa.load(file_path)
    buffer = 30 * sr

    samples_total = len(audio)
    samples_wrote = 0
    counter = 1

    while samples_wrote < samples_total:

        # check if the buffer is not exceeding total samples
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = audio[samples_wrote : (samples_wrote + buffer)]
        out_file_name = f"{file_name}_clip_" + str(counter) + ".wav"
        complete_name = f"{save_path}/{out_file_name}"

        sf.write(complete_name, block, sr)

        counter += 1
        samples_wrote += buffer

    print(
        f"----Segmented audio file. Time taken: {round(time.time()-current_time, 1)} s"
    )
