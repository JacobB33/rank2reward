# Rank2Reward ranking codebase:

## Installation
- Create a new conda environment / use an existing conda environment with the following command:
```bash
conda create -n rank2reward python=3.8
```
- Inside the conda environment, install Rank2Reward with the following command:
```bash
pip install -e .
```

## Usage:
- Inside the code, there is a Trajectory class that represents a single trajectory of images. Here is example code on 
how to load a list of mp4 files into a list of Trajectory objects:
```python
from rank2reward import Trajectory
from PIL import Image
import numpy as np
import os
import cv2

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(Image.fromarray(frame))
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

video_dir = 'PATH_TO_VIDEO_DIR'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

trajectories = []
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    trajectories.append(Trajectory(images=load_video(video_path)))
```
- The other class is the ranking network. When you instantiate a ranking network, you can call two methods on it
    - `ranking_network.rank(images)` returns a tensor of ranks for the images
    - `ranking_network.train(trajectories, num_epochs=xxx)` trains the ranking network on the passed trajectories.
    - **Look at example_ranking.py for a example ranking trjacetories and training the ranking network.**