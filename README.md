# Communication-Analysis-Tool-for-Human-AI-Interaction-Driving-Simulator-Experiments-Screening-Test

## Requirements
- *moviepy*
- *speesch_recognition*
- *pydub*
- *nltk*
- *pandas*
- *matplotlib*
- *pocketsphinx*
- ffmpeg*

## Execution Instructions
- *Set up the environment:*
  Use "pip" to install all required libraries: pip install moviepy SpeechRecognition pydub nltk pandas matplotlib pocketsphinx ffmpeg

- *Run data_processing.py:*
  Make sure dataset_folder to point to the folder containing the video files
  Execute: python data_processing.py
  CSV files will be generated in the specified output folder

- *Run data_visualization.py:*
  Make sure csv_file_path to the path of one of the generated CSV files
  Execute: python data_visualization.py
  Two plots will be displayed: one histogram and one sentiment distribution chart

