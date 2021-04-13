import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template
import shutil

from clean_folder import *

UPLOAD_FOLDER = 'uploads/'
DOWNLOAD_FOLDER = 'downloads/'
EMOTIONS_FOLDER = './static/emotions/'
FINAL_AUDIO_FOLDER = './static/final_audio/'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['EMOTIONS_FOLDER'] = EMOTIONS_FOLDER
app.config['FINAL_AUDIO_FOLDER'] = FINAL_AUDIO_FOLDER

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload API
@app.route('/uploadfile', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # clean the existing files from the given folders
        clean_folder(UPLOAD_FOLDER)
        clean_folder(DOWNLOAD_FOLDER)
        clean_folder(EMOTIONS_FOLDER)
        clean_folder(FINAL_AUDIO_FOLDER)

        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)

        else:
            filename = secure_filename(file.filename)
            # save the input audio in the uploads folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # file.save only works once like a queue and is also called file buffer.
            # Since the audio has been parsed now, if we use the above line of code to save the above audio file, it will get saved empty. 
            # Therefore we use the audio saved once in the uploads folder and make copies of it as shown below
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['EMOTIONS_FOLDER'], filename))
            shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['FINAL_AUDIO_FOLDER'], filename))
            print("saved file successfully")

            # send file name as parameter to download
            return redirect('/player/'+ filename)
    return render_template('upload_file.html')

# Player API
@app.route("/player/<filename>", methods = ['GET', 'POST'])
def player(filename):
    file_path = EMOTIONS_FOLDER + filename
    if request.method == 'POST':
        # return send_file(file_path, as_attachment=True, attachment_filename='')
        return redirect('/final_product/'+ filename)
    return render_template('player.html', value=filename)

@app.route('/play_audio/<filename>')
def play_audio(filename):
    file_path = EMOTIONS_FOLDER + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')

# Final Audio API
@app.route("/final_product/<filename>", methods = ['GET', 'POST'])
def final_product(filename):
    return render_template('final_product.html', value=filename)

# Download API
@app.route("/downloadfile/<filename>", methods = ['GET', 'POST'])
def download_file(filename):
    return render_template('download.html', value=filename)

@app.route('/return-files/<filename>', methods = ['GET', 'POST'])
def return_files_tut(filename):
    file_path = DOWNLOAD_FOLDER + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')
    
if __name__ == "__main__":
    app.run(debug=True)