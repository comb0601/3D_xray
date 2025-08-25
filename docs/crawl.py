import os
import yt_dlp

# List of YouTube links
links = [
    "https://www.youtube.com/watch?v=2Q-qzci5jSE",

]

# Create videos folder if it doesn't exist
os.makedirs("videos", exist_ok=True)

# Download options
ydl_opts = {
    "outtmpl": "videos/%(title)s.%(ext)s",  # Save in videos/ with video title
    "format": "bestvideo+bestaudio/best",   # Best quality
    "merge_output_format": "mp4",           # Merge into mp4
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(links)