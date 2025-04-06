"""
Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments – Screening Test

This program processes video files (with audio from a simulated environment) by:
    - Extracting audio from each video using MoviePy.
    - Segmenting the audio into chunks no longer than 5 seconds.
    - Performing offline speech-to-text conversion (using pocketsphinx via the SpeechRecognition library).
    - Mapping each transcription to its starting timestamp in the video.
    - Analyzing the sentiment of each transcribed line with NLTK's VADER sentiment analyzer.
    - Saving the transcription, timestamp, sentiment classification, and additional details into a CSV file.

The program is structured to allow scalability over many video files.

Requirements:
    - moviepy
    - speech_recognition
    - pydub
    - nltk
    - pandas
    - pocketsphinx (for offline recognition)
    - (Optionally) ffmpeg installed in the system for moviepy and pydub to work properly.
"""

import os
import datetime
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already available
nltk.download('vader_lexicon')


def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from the video file using MoviePy and save it as a WAV file.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)


def segment_audio(audio_path, segment_length_ms=5000):
    """
    Segment the audio file into chunks of specified length (default 5000 ms = 5 seconds).
    
    Parameters:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Maximum length of each segment in milliseconds.
        
    Returns:
        List of tuples: Each tuple contains the start time (in seconds) and the audio segment (AudioSegment object).
    """
    audio = AudioSegment.from_file(audio_path)
    segments = []
    for start_ms in range(0, len(audio), segment_length_ms):
        segment = audio[start_ms:start_ms + segment_length_ms]
        # Convert milliseconds to seconds for the timestamp
        segments.append((start_ms / 1000.0, segment))
    return segments


def speech_to_text(audio_segment):
    """
    Convert a pydub AudioSegment to text using offline recognition with pocketsphinx.
    
    Parameters:
        audio_segment (AudioSegment): The audio segment to process.
        
    Returns:
        str: Transcribed text from the audio segment.
    """
    # Export the audio segment to a temporary WAV file
    temp_filename = "temp_segment.wav"
    audio_segment.export(temp_filename, format="wav")
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_filename) as source:
        audio_data = recognizer.record(source)
    try:
        # Use pocketsphinx (offline) for speech recognition
        transcription = recognizer.recognize_sphinx(audio_data)
    except sr.UnknownValueError:
        transcription = ""
    except sr.RequestError as e:
        transcription = ""
    
    # Clean up the temporary file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return transcription


def analyze_sentiment(text, analyzer):
    """
    Analyze the sentiment of the provided text using NLTK's VADER sentiment analyzer.
    
    Parameters:
        text (str): The text to analyze.
        analyzer (SentimentIntensityAnalyzer): An instance of the sentiment analyzer.
        
    Returns:
        str: Sentiment classification ('positive', 'negative', or 'neutral').
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def process_video_file(video_file, output_csv_path):
    """
    Process a single video file: extract its audio, segment the audio, perform speech-to-text on each segment,
    analyze sentiment, and save the results into a CSV file.
    
    Parameters:
        video_file (str): Path to the video file.
        output_csv_path (str): Path to save the resulting CSV file.
    """
    # Create a base name for temporary audio file extraction
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    audio_file = f"{base_name}.wav"
    
    # Extract audio from the video file
    extract_audio_from_video(video_file, audio_file)
    
    # Segment the extracted audio into chunks of 5 seconds
    segments = segment_audio(audio_file, segment_length_ms=5000)
    
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Process each audio segment
    data_rows = []
    for start_time, segment in segments:
        # Perform speech-to-text conversion on the segment
        transcription = speech_to_text(segment)
        # Analyze sentiment of the transcribed text
        sentiment = analyze_sentiment(transcription, analyzer)
        # Append the result as a row of data
        data_rows.append({
            "video_file": video_file,
            "start_timestamp": start_time,
            "transcription": transcription,
            "sentiment": sentiment
        })
    
    # Create a Pandas DataFrame from the collected data rows
    df = pd.DataFrame(data_rows)
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    
    # Remove the temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)


def process_dataset(dataset_folder, output_folder):
    """
    Iterate through a folder containing video files and process each file.
    Generates a uniquely named CSV file for each processed video.
    
    Parameters:
        dataset_folder (str): Path to the folder containing video files.
        output_folder (str): Folder where CSV files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for file in os.listdir(dataset_folder):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_file = os.path.join(dataset_folder, file)
            # Create a unique CSV file name using the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{timestamp}.csv")
            process_video_file(video_file, output_csv)
            print(f"Processed {video_file} -> {output_csv}")


if __name__ == "__main__":
    # Define the folder containing video files (update this path as needed)
    dataset_folder = "path_to_video_files"  # e.g., "data/videos"
    # Define the folder where output CSV files will be stored
    output_folder = "output_csv"
    # Process the entire dataset
    process_dataset(dataset_folder, output_folder)
