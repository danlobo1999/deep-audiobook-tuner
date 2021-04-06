import os
from dotenv import load_dotenv
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

load_dotenv("../.env")


def setUpIBM():
    api_key = os.getenv("api_key")
    url = os.getenv("url")

    # Setup Service
    authenticator = IAMAuthenticator(api_key)
    stt = SpeechToTextV1(authenticator=authenticator)
    stt.set_service_url(url)
    return stt


def transcribeAudio(file, stt):
    # Perform transcription
    with open(file, "rb") as f:
        res = stt.recognize(
            audio=f,
            content_type="audio/wav",
            model="en-US_NarrowbandModel",
            continuous=True,
        ).get_result()

    text = ""
    conf = 0.0

    for i in range(len(res["results"])):
        text += res["results"][i]["alternatives"][0]["transcript"][0:-1] + ". "
        conf += res["results"][i]["alternatives"][0]["confidence"]

    conf = conf / len(res["results"])

    return text, conf