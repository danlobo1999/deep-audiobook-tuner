import librosa
import numpy as np
import tensorflow as tf
import keras
import pickle
import time


def featureExtraction(y, sr):
    rmse = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

    data_features = [
        rmse,
        spec_cent,
        spec_bw,
        rolloff,
        zcr,
        chroma_stft[0],
        chroma_stft[1],
        chroma_stft[2],
        chroma_stft[3],
        chroma_stft[4],
        chroma_stft[5],
        chroma_stft[6],
        chroma_stft[7],
        chroma_stft[8],
        chroma_stft[9],
        chroma_stft[10],
        chroma_stft[11],
        mfcc[0],
        mfcc[1],
        mfcc[2],
        mfcc[3],
        mfcc[4],
        mfcc[5],
        mfcc[6],
        mfcc[7],
        mfcc[8],
        mfcc[9],
        mfcc[10],
        mfcc[11],
        mfcc[12],
        mfcc[13],
        mfcc[14],
        mfcc[15],
        mfcc[16],
        mfcc[17],
        mfcc[18],
        mfcc[19],
    ]
    return data_features


def loadAudioAssets(model_path, pickles_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load the scaler
    pickle_in = open(f"{pickles_path}/scaler.pickle", "rb")
    scaler = pickle.load(pickle_in)

    # Laod the audio classes
    pickle_in = open(f"{pickles_path}/labels.pickle", "rb")
    audio_classes = pickle.load(pickle_in)

    return model, scaler, audio_classes


def analyzeAudio(file_name, model, scaler):
    current_time = time.time()

    y, sr = librosa.load(file_name, res_type="kaiser_fast", sr=22050 * 2)

    buffer = 3 * sr

    samples_total = len(y)
    samples_wrote = 0

    predictions = []

    # Splits the block into 3 second clips and analyzes them
    while samples_wrote < samples_total:
        # check if the buffer is not exceeding total samples
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote

        block = y[samples_wrote : (samples_wrote + buffer)]

        data_features = np.array(featureExtraction(block, sr))

        scaled_features = scaler.transform(data_features.reshape(1, -1))

        predictions.append(model.predict(scaled_features))

        samples_wrote += buffer

    audio_emotions = np.squeeze(predictions, axis=None)
    audio_emotions_length = len(audio_emotions)
    audio_emotions = audio_emotions.sum(axis=0)
    audio_emotions = audio_emotions / audio_emotions_length

    print(
        f"--------Audio analysis complete. Time taken: {round(time.time()-current_time, 1)} s"
    )

    return audio_emotions
