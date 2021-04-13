# deep-audiobook-tuner

A system that generates an apt, emotionally pertinent, unique musical score for an audiobook automatically based on the current narrative for the purpose of ameliorating user-experience while being accurate, cost-efficient, and time saving.

## **About:**
Audiobooks are being used on a regular basis by hundreds of users. The system in this report aims to develop emotionally relevant music for preexisting audiobook recordings. The user will enter an audiobook MP3 file as an input to the system. This audiobook will then go through two processes, simultaneously. 

*   First, the input audiobook will be run through a transcription tool to extract the text from the audiobook. This text will then be analysed using a Text-Based Sentiment Analyzer (TBSA).  
*   Concomitantly, in the second process, the features of the audio from the audiobook that is given by the user, will be extracted. The audio features are then analysed by an Audio-Based Sentiment Analyzer (ABSA) that will predict the emotions being conveyed in the audio. 

Now the system will have obtained 2 values (sentiments) predicted by both, the TBSA as well as the ABSA. The values may vary and lead to an error. To avoid this, the weighted average of values will be calculated in order to generate the  final predicted sentiments. Utilizing these predicted sentiments as well as the music generation model that has been explained ahead in this report, our application generates a seamless, distinctive musical score for every segment. These scores are stitched together along with the input audio file to provide the user an audiobook with felicitous background tunes.

## Libraries to install

Install the requirements for [Tensorflow](https://www.tensorflow.org/install) before you run the following commands.

Run `pip install -r requirements.txt` to install all the required libraries (python version = 3.7) 

Or   

Create a conda environment: `conda env create -f environment.yml`  
(This method requires tensorflow 2.4 to be installed seperately in the environment.  
Run `conda activate deepaudiobooktuner` and `pip install tensorflow==2.4.1`)


ffmpeg is required by the system to covert mp3 files to wav files. It can be installed from [here.](https://www.ffmpeg.org/download.html)

---


## **Collaborators:**


*   [Daniel Lobo](https://github.com/danlobo1999)
*   [Jenny Dcruz](https://github.com/jendcruz22)
*   [Smita Deulkar](https://github.com/smita3199)
*   [Leander Fernandes](https://github.com/fernandeslder)

