import librosa
import soundfile as sf
import subprocess
import time


def convertToWav(file_name, file_path, save_path):
    current_time = time.time()

    wav_file_name = f"{save_path}\{file_name}.wav"
    subprocess.call(["ffmpeg", "-i", file_path, wav_file_name])

    print(f"----Converted to wav. Time taken: {round(time.time()-current_time, 1)} s")

    return wav_file_name


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
