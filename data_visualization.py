"""
This program reads a CSV file and creates:
    - A histogram showing the count of transcribed words per 5-second time bucket.
    - A bar chart displaying the distribution of sentiment classifications (positive, negative, neutral).

Requirements:
    - pandas
    - matplotlib
"""

import math
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(csv_file):
    """
    Generate a histogram based on the count of transcribed words per 5-second bucket.
    
    Parameters:
        csv_file (str): Path to the CSV file containing transcription data.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert start_timestamp to numeric (seconds)
    df['start_timestamp'] = pd.to_numeric(df['start_timestamp'], errors='coerce')
    
    # Create a new column that counts the number of words in each transcription
    df['word_count'] = df['transcription'].fillna("").apply(lambda text: len(text.split()))
    
    # Determine the maximum timestamp to create time buckets
    max_time = df['start_timestamp'].max()
    num_buckets = math.ceil(max_time / 5)
    
    # Assign each transcription segment to a bucket (0-indexed)
    df['bucket'] = df['start_timestamp'].apply(lambda t: int(t // 5))
    
    # Sum word counts for each bucket
    bucket_counts = df.groupby('bucket')['word_count'].sum().reset_index()
    
    # Create readable labels for each bucket
    bucket_counts['bucket_label'] = bucket_counts['bucket'].apply(lambda b: f"{b*5}-{(b+1)*5} sec")
    
    # Plot the histogram (bar chart)
    plt.figure(figsize=(10, 6))
    plt.bar(bucket_counts['bucket_label'], bucket_counts['word_count'])
    plt.xlabel("Time Bucket (seconds)")
    plt.ylabel("Word Count")
    plt.title("Histogram of Transcribed Word Count per 5-second Bucket")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sentiment_distribution(csv_file):
    """
    Generate a bar chart that displays the sentiment classification results.
    
    Parameters:
        csv_file (str): Path to the CSV file containing transcription and sentiment data.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Count the number of occurrences for each sentiment category
    sentiment_counts = df['sentiment'].value_counts()
    
    # Plot the sentiment distribution as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Classification Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the path to the CSV file generated from Program 1.
    csv_file_path = "output_csv/generated_file.csv"
    
    # Generate and display the histogram of transcribed words per 5-second bucket.
    plot_histogram(csv_file_path)
    
    # Generate and display the sentiment distribution plot.
    plot_sentiment_distribution(csv_file_path)
