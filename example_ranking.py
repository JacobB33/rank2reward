import pickle
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rank2reward import RankingNet, Trajectory
import cv2
import os

def load_video(video_path) -> List[np.ndarray]:
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

video_dir = './playback-rgb/train'
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

trajectories = []
print(video_files)
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    frames = load_video(video_path)
    trajectory = Trajectory(images=frames)
    trajectories.append(trajectory)
print(len(trajectories[0]))
ranking_net = RankingNet(epoch_size=64*3, transform_batch_size=64*3, episode_len=31, goal_conditioned=False, train_with_mixup=False, lr=0.0001)
# ranking_net.load('ranking_net.pkl')
# ranking_net = pickle.load(open('ranking_net.pkl', 'rb'))
loss_dict = ranking_net.train_ranking_net(trajectories, num_epochs=1000, with_tqdm=True)
print(loss_dict)

for traj in trajectories:
    traj_rank = []
    for i in range(len(traj)):
        rank = ranking_net.rank(traj[i])
        traj_rank.append(rank)
    plt.cla(); plt.clf()
    plt.plot(traj_rank)
    plt.show()


# pickle.dump(ranking_net, open('ranking_net.pkl', 'wb'))
# pickle.dump(loss_dict, open('loss_dict.pkl', 'wb'))

