import time

from deepaudiobooktuner.sentiment_analysis.audio_analysis import loadAudioAssets
from deepaudiobooktuner.sentiment_analysis.text_analysis import loadTextAssets
from deepaudiobooktuner.music_generation.music_generation import loadMusicAssets
from deepaudiobooktuner.sentiment_analysis.ibm_transcription import setUpIBM


def loadAssets(paths):
    # Loading the audio analyzer model, scaler and classes
    current_time = time.time()
    audio_model, audio_scaler, audio_classes = loadAudioAssets(
        model_path=paths["audio_model"], pickles_path=paths["pickles"]
    )
    print(
        f"----Loaded audio model assets. Time taken: {round(time.time()-current_time, 1)} s"
    )

    # Loading the text analyzer model and classes
    current_time = time.time()
    text_predictor, text_classes = loadTextAssets(model_path=paths["text_model"])
    print(
        f"----Loaded text model assets. Time taken: {round(time.time()-current_time, 1)} s"
    )

    # Loading the music generation model and music_data
    current_time = time.time()
    music_data, music_model = loadMusicAssets(
        music_data_path=paths["music_data"], music_model_path=paths["music_model"]
    )
    print(
        f"----Loaded music model assets. Time taken: {round(time.time()-current_time, 1)} s"
    )

    # Setting up IBM
    current_time = time.time()
    stt = setUpIBM()
    print(
        f"----Setup IBM transcription service. Time taken: {round(time.time()-current_time, 1)} s"
    )

    assets = {
        "audio_model": audio_model,
        "audio_scaler": audio_scaler,
        "audio_classes": audio_classes,
        "text_predictor": text_predictor,
        "text_classes": text_classes,
        "music_data": music_data,
        "music_model": music_model,
        "stt": stt,
    }

    return assets
