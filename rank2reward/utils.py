from dataclasses import dataclass
import numpy as np
import torch
from typing import List, Union


@dataclass
class Trajectory:
    # images should be a list of images of shape (H, W, C) or PIL format from 0 to 255.
    images: List[np.ndarray]

    # None if not goal based, otherwise the image that represents the goal.
    goal: Union[None, np.ndarray] = None

    def __len__(self):
        """
        This function returns the length of the trajectory
        :return: The number of images in the trajectory
        """
        return len(self.images)

    def __getitem__(self, item):
        """
        This function returns the image at a given index
        :param item: index
        :return: the image at the given index
        """
        return self.images[item]

def mixup_data(x, y, alpha=1.0, goals=None):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size()[0]

    index = torch.randperm(bs).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    if goals is not None:
        mixed_goals = lam * goals + (1 - lam) * goals[index, :]
        return mixed_x, y_a, y_b, lam, mixed_goals
    else:
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
