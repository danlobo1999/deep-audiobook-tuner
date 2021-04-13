import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template
import shutil

def clean_folder(folder_to_be_cleant):
    # Delete existing files from given folder
    if len(os.listdir(folder_to_be_cleant) ) != 0:
        for filename in os.listdir(folder_to_be_cleant):
            file_path = os.path.join(folder_to_be_cleant, filename)
            try:
                if filename!=".gitignore":
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)

                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

            except Exception as e:
                print('Failed to delete. Reason: %s' % (e))
    return