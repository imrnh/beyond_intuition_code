import requests
import subprocess
from pathlib import Path

# Check if dataset exists.
# If not, download
dataset_path = Path("gtsegs_ijcv.mat")
dataset_download_url = "http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat"
model_path


try:
    if not dataset_path.is_file():
       subprocess.call(['wget', dataset_download_url])
except Exception as e:
    print("Error checking or downloading dataset file. {}".format(e))