import time
import os
import sys

sys.path.append(os.path.abspath("../"))

from deepaudiobooktuner.music_generation.music_generation import (
    generateMusicClips,
    loadMusicAssets,
)
from deepaudiobooktuner.utils.file_processing import convertToMp3, saveMusicClips
from deepaudiobooktuner.utils.paths import path


paths = {
    "music_samples": path("../assets/music_generation/datasets/vg-midi-annotated"),
    "music_data": path("../assets/music_generation/pickles/"),
    "music_clips_save_path": path("music_clips"),
    "music_model": path("../assets/music_generation/models/MusicTransformerKeyC.pth"),
}
# os.mkdir(paths["music_clips_save_path"])

music_data, music_model = loadMusicAssets(paths)


def generateMusic(music_emotions=["Angry", "Happy", "Neutral", "Sad"]):
    # Generating music clips
    print("\n\nGenerating music")
    music_generation_time = time.time()
    songs = {}

    songs = generateMusicClips(
        music_emotions=music_emotions,
        paths=paths,
        music_model=music_model,
        music_data=music_data,
        songs=songs,
    )

    saveMusicClips(music_emotions=music_emotions, songs=songs, paths=paths)

    print(
        f"----\nMusic Generation Complete. Time taken: {round(time.time() - music_generation_time, 1)} s"
    )


generateMusic()
