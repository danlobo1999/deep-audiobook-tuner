import ktrain
from ktrain import text
import numpy as np
import time

from deepaudiobooktuner.sentiment_analysis.ibm_transcription import transcribeAudio


def loadTextAssets(model_path):
    predictor = ktrain.load_predictor(model_path)
    classes = predictor.get_classes()
    return predictor, classes


def analyzeText(file_name, stt, predictor):

    # Transcribe the audio
    current_time = time.time()
    text, conf = transcribeAudio(file=file_name, stt=stt)
    print(
        f"--------Transcription complete. Time taken: {round(time.time()-current_time, 1)} s"
    )

    current_time = time.time()

    words = text.split(" ")
    split_length = len(words) // 3

    split_text = []
    split_index = 0

    for i in range(2):
        split_text.append(words[split_index : split_index + split_length])
        split_index = split_index + split_length
    split_text.append(words[split_index::])

    split_sentences = []
    for i in range(len(split_text)):
        sentence = " ".join(split_text[i])
        split_sentences.append(sentence)

    final_preds = []
    for i in range(len(split_sentences)):
        prediction = predictor.predict(split_sentences[i], return_proba=True)
        final_preds.append(prediction)
        # print(f"text-clip-{i+1}-prediction {prediction}")

    finpredval = np.sum(final_preds, axis=0) / 3

    text_emotions = np.array(
        [finpredval[2], finpredval[0], finpredval[3], finpredval[1]]
    )

    print(
        f"--------Text analysis complete. Time taken: {round(time.time()-current_time, 1)} s"
    )

    return text_emotions, text
