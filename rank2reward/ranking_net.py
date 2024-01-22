import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Dict, Collection
import tqdm

from .models import R3MPolicy
from .utils import Trajectory, mixup_criterion, mixup_data

class RankingNet:
    def __init__(self,
                 epoch_size: int,
                 transform_batch_size: int,
                 episode_len: int,
                 lr: float = 1e-4,
                 r3m_net: str = "resnet18",
                 train_r3m: bool = False,
                 state_noise: float = 0.01,
                 device: str = "cuda:0",
                 goal_conditioned: bool = False,
                 train_with_mixup: bool = True,
                 ):
        """
        This is a ranking net based on images from rank2reward
        :param epoch_size: The size for each epoch of data
        :param transform_batch_size: The batch size to batch the data preprocessing in
        :param episode_len: The lengths of your trajectories
        :param lr: Learning rate f
        :param r3m_net:
        :param train_r3m:
        :param state_noise:
        :param device:
        :param goal_conditioned:
        :param train_with_mixup:
        """
        self.train_with_mixup = train_with_mixup
        assert epoch_size % transform_batch_size == 0, "epoch_size must be divisible by batch_size"

        self.epoch_size = epoch_size
        self.transform_batch_size = transform_batch_size
        self.episode_len = episode_len
        # self.state_noise = state_noise
        self.device = device
        self.goal_conditioned = goal_conditioned

        self.ranking_net = R3MPolicy(1, r3m_type=r3m_net, do_multiply_255=True, freeze_backbone=train_r3m,
                                     with_goal=goal_conditioned, film_layer_goal=True, state_noise=state_noise).to(
            device)
        self.ranking_optimizer = torch.optim.Adam(self.ranking_net.parameters(), lr=lr)

        # Define the transforms we apply to the images
        self._train_transforms = T.Compose([
            T.Resize(256),
            # T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.3),
            # T.RandomRotation(15),  # +/-15 degrees of random rotation
            T.CenterCrop(224)#T.RandomCrop(224),  # T.CenterCrop(224)
        ])

        if self.goal_conditioned:
            self._goal_transforms = T.Compose([
                T.Resize(256),
                T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.3),
                T.RandomCrop(224),  # T.CenterCrop(224)
            ])
        self.bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()

    def rank(self, image: np.ndarray, goal=None) -> float:
        """
        This function returns the rank of a given image
        :param image: the image to rank
        :return: float rank.
        """
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        output_tesnsor = self._train_transforms(image_tensor)
        if goal:
            goal = T.ToTensor()(goal).unsqueeze(0).to(self.device)
        with torch.no_grad():
            rank = self.ranking_net(image_tensor, goal=goal)
        return rank.item()

    def train_ranking_net(self, expert_trajectories: List[Trajectory], num_epochs: int = 1, with_tqdm: bool = False) -> Dict:
        """
        This function trains the ranking net on expert trajectories.
        :param with_tqdm: Render a TQDM bar with training
        :param expert_trajectories: list of expert trajectories
        :param num_epochs: number of epochs to train the ranking net
        :return: None
        """
        ranking_losses = []
        for i in tqdm.tqdm(range(num_epochs), disable=not with_tqdm):
            self.ranking_optimizer.zero_grad()
            ranking_loss = self._train_ranking_step(expert_trajectories)
            ranking_loss.backward()
            self.ranking_optimizer.step()
            ranking_losses.append(ranking_loss.item())
        return {"ranking_loss": ranking_losses}


    def _train_ranking_step(self, expert_trajectories: List[Trajectory]) -> torch.Tensor:
        '''
        sample from expert data (factuals) => train ranking, classifier positives
        '''
        expert_idxs = np.random.randint(len(expert_trajectories), size=(self.epoch_size,))
        expert_t_idxs = np.random.randint(low=0, high=self.episode_len, size=(self.epoch_size,))
        expert_other_t_idxs = np.random.randint(low=0, high=self.episode_len, size=(self.epoch_size,))
        labels = np.zeros((self.epoch_size,))
        first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
        labels[first_before] = 1.0  # idx is 1.0 if other timestep > timestep

        # get images
        expert_images_t_tensor = torch.cat(
            [T.ToTensor()(expert_trajectories[traj_idx][t_idx]).unsqueeze(0) for (traj_idx, t_idx) in
             zip(expert_idxs, expert_t_idxs)],
            dim=0
        ).to(self.device)

        expert_other_images_t_tensor = torch.cat(
            [T.ToTensor()(expert_trajectories[traj_idx][t_idx]).unsqueeze(0) for (traj_idx, t_idx) in
             zip(expert_idxs, expert_other_t_idxs)],
            dim=0
        ).to(self.device)

        # Get the goals if goal conditioned
        goals_processed_images_t_tensor = None
        if self.goal_conditioned:
            goals_images_t_tensor = torch.cat([
                T.ToTensor()(expert_trajectories[traj_idx].goal).unsqueeze(0) for traj_idx in expert_idxs
            ], dim=0).to(self.device)

            goals_processed_images_t_tensor = torch.cat([
                self._goal_transforms(goals_images_t_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size]) for i in
                range(int(self.epoch_size / self.transform_batch_size))
            ])

        # Batchify the data, and apply data augmentation
        expert_processed_images_tensor = torch.cat(
            [self._train_transforms(
                expert_images_t_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size]) for i in
                range(int(self.epoch_size / self.transform_batch_size))]
        )

        expert_processed_images_other_tensor = torch.cat(
            [self._train_transforms(
                expert_other_images_t_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size])
                for i in range(int(self.epoch_size / self.transform_batch_size))]
        )

        ranking_labels = F.one_hot(torch.Tensor(labels).long().to(self.device), 2).float()

        loss_monotonic = None
        if self.train_with_mixup:
            rank_states = torch.cat([expert_processed_images_tensor, expert_processed_images_other_tensor], dim=0)
            rank_labels = torch.cat([ranking_labels[:, 0], ranking_labels[:, 1]], dim=0).unsqueeze(1)

            mixed_rank_states, rank_labels_a, rank_labels_b, rank_lam = mixup_data(rank_states, rank_labels)
            mixed_rank_prediction_logits = self.ranking_net(mixed_rank_states)

            loss_monotonic = mixup_criterion(
                self.bce_with_logits_criterion, mixed_rank_prediction_logits, rank_labels_a, rank_labels_b, rank_lam
            )
        else:
            expert_logits_t = self.ranking_net(expert_processed_images_tensor, goal=goals_processed_images_t_tensor)
            expert_logits_other_t = self.ranking_net(expert_processed_images_other_tensor,
                                                     goal=goals_processed_images_t_tensor)
            expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)

            loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)
        return loss_monotonic

    def _assert_same_traj_len(self, trajectory_list: List) -> None:
        """
        This function asserts that all trajectories in the list have the same length.
        :param trajectory_list: list of trajectories
        :return: None
        """
        assert len(trajectory_list) > 0
        for traj in trajectory_list:
            assert (len(traj) == self.episode_len), "All trajectories must have the same length."

    def save(self, path: str) -> None:
        """
        This function saves the ranking net to the specified path.
        :param path: path to save the ranking net to.
        :return: None
        """
        state_dict = {
            "ranking_net": self.ranking_net.state_dict(),
            "ranking_optimizer": self.ranking_optimizer.state_dict(),
        }
        torch.save(state_dict, path)

    def load(self, path: str):
        """
        This function loads the ranking net from the specified path.
        :param path: path to load the ranking net from.
        :return: None
        """
        state_dict = torch.load(path)
        self.ranking_net.load_state_dict(state_dict["ranking_net"])
        self.ranking_optimizer.load_state_dict(state_dict["ranking_optimizer"])