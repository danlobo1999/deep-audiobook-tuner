import os
import time


def path(relative_path):
    return os.path.abspath(relative_path)


def createDir(file_path):
    file_name = file_path.split("\\")[-1]
    file_name = file_name.split("/")[-1][:-4]
    creation_time = time.time()

    paths = {
        "audio_model": path(
            "../assets/audio_sentiment_data_v2/models/hyperband_tuned_model_19_april_[0.3624165654182434, 0.8720930218696594]/"
        ),
        "pickles": path("../assets/audio_sentiment_data_v2/pickles"),
        "text_model": path("../assets/text_sentiment_data/models/neubias_bert_model/"),
        "music_model": path(
            "../assets/music_generation/models/MusicTransformerKeyC.pth"
        ),
        "music_data": path("../assets/music_generation/pickles/"),
        "music_samples": path("../assets/music_generation/datasets/vg-midi-annotated"),
        "wav_save_path": path(f"../assets/temp/{file_name}-{creation_time}"),
        "clips_save_path": path(f"../assets/temp/{file_name}-{creation_time}/clips"),
        "music_clips_save_path": path(
            f"../assets/temp/{file_name}-{creation_time}/music_clips"
        ),
        "final_audiobook_save_path": path(
            f"../assets/temp/{file_name}-{creation_time}/final_audiobook"
        ),
    }

    # Creating directories in temp to store the converted wav file and the clips
    os.mkdir(paths["wav_save_path"])
    os.mkdir(paths["clips_save_path"])
    os.mkdir(paths["music_clips_save_path"])
    os.mkdir(paths["final_audiobook_save_path"])

    print("----Temporary directory created.")

    return file_name, paths
