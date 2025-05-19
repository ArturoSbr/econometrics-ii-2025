"""Fetch comments from a YouTube video using YouTube Data API v3

This script fetches the Channel ID of a guy named Sliinky, searches for his five latest
video uploads and prints out the comment section of his most recent video.
"""

# Imports
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build  # Clean but only works for Google services
# import requests  # More general way to make API calls (requests)

# Load API key (must match name in .env file)
load_dotenv()
key = os.getenv('API_KEY')

# Build youtube client
youtube = build(
    serviceName='youtube',
    version='v3',
    developerKey=key
)  # This would be slightly more tedious with `requests` library

# 1. Get Sliinky's Channel ID
# Declare search parameters
request_id = youtube.search().list(
    part='snippet',
    q='Sliinky',
    type='channel',
    maxResults=1
)

# Execute search
response_id = request_id.execute()

# Get the good stuff from response
slinky_id = response_id['items'][0]['id']['channelId']

# 2. Search for Sliinky's latest videos
# Declare search parameters
request_uploads = youtube.search().list(
    part='snippet',
    channelId=slinky_id,
    maxResults=5,
    order='date',
    type='video'
)

# Execute search
response_uploads = request_uploads.execute()

# Get ID of latest video
latest_video_id = response_uploads['items'][0]['id']['videoId']

# 3. Scrape comments from latest video
# Declare comment section parameters
request_comments = youtube.commentThreads().list(
    part='snippet',
    videoId=latest_video_id,
    maxResults=100,
    textFormat='plainText'
)

# Execute request
response_comments = request_comments.execute()

# Print comments
for item in response_comments['items']:
    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
    print(comment, '\n')
