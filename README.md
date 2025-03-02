# Feature and Similarity based playlisting
This repository contains code to run through a given directory of mp3 files, analyse features,
present the data, and run an interactive web-app to build playlists based on features and similarity.

## Installation:
Create a python 3.10 environment in conda, and run `pip install -r requirements.txt`. Code was tested with python-3.10.
Essentia is known to be troublesome to install on linux and windows, sorry about that.

## Use:
All commands are meant to be run from the project directory.

There are 3 files to run:
1. **analysis.py:**
  This script scans a given folder for mp3 files and creates a json file that stores the data.
  Make sure to specify the paths in the top of the file before running. To run, execute `python src/analysis.py`.
  Note that this process may take some time.

2. **data_analysispy:**
  This script analyses the `audio_features.json` file and creates graphs of relevant data of the library.
  This script also creates `audio_features.pkl` for use in the playlisting web-app.
  To run, execute `python src/data_analysis.py`.

3. **myApp.py:**
  This script contains the code needed to run the web-app. To run, execute `streamlit run src/myApp.py`.
