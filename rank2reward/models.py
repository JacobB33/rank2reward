import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

from r3m import load_r3m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Utilities for defining neural nets
# def weight_init(m):
#     """Custom weight init for Conv2D and Linear layers."""
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight.data)
#         if hasattr(m.bias, "data"):
#             m.bias.data.fill_(0.0)
#
#
# class MLP(nn.Module):
#     def __init__(
#             self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
#     ):
#         super().__init__()
#         self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
#         self.apply(weight_init)
#
#     def forward(self, x):
#         return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, do_regularization=True):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        if do_regularization:
            # adding spectral norm
            # removing batch norm: nn.BatchNorm1d(hidden_dim)
            mods = [nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)), nn.ReLU(inplace=True)]
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]

        for i in range(hidden_depth - 1):
            if do_regularization:
                mods += [nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(inplace=True)]
            else:
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim)))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, output_mod=None, do_regularization=False):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth, output_mod=output_mod,
                         do_regularization=do_regularization)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred


class R3MPolicy(nn.Module):
    def __init__(
            self,
            output_size: int = 1,
            do_multiply_255: bool = True,
            freeze_backbone: bool = True,
            with_goal: bool = False,
            film_layer_goal: bool = True,
            state_noise: float = 0.0,
            r3m_type: str = "resnet18",
    ):
        super(R3MPolicy, self).__init__()

        assert r3m_type in ["resnet18", "resnet50"], "r3m_type must be one of resnet18 or resnet50"
        # get the backbone
        self.r3m = load_r3m(r3m_type)  # resnet18, resnet34, resnet50
        self.r3m.to(device)
        # r3m_embedding_dim = 2048  # for resnet18 - 512, for resnet50 - 2048
        self.r3m_embedding_dim = 512 if r3m_type == "resnet18" else 2048
        self.with_goal = with_goal
        self.film_layer_goal = film_layer_goal
        self.state_noise = state_noise
        self.freeze_r3m = freeze_backbone
        if self.freeze_r3m:
            self.r3m.eval()
            for param in self.r3m.parameters():
                param.requires_grad = False

        # add our head to the r3m output
        if self.with_goal:
            if self.film_layer_goal:
                fc_head_in = self.r3m_embedding_dim
                self.film_layer = nn.Sequential(
                    nn.Linear(self.r3m_embedding_dim, 4 * self.r3m_embedding_dim),
                    nn.LeakyReLU(),
                    nn.Linear(4 * self.r3m_embedding_dim, 2 * self.r3m_embedding_dim)
                )
            else:
                fc_head_in = 2 * self.r3m_embedding_dim
        else:
            fc_head_in = self.r3m_embedding_dim

        self.output_size = output_size
        self.fc_head = nn.Sequential(
            nn.Linear(fc_head_in, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_size),
        )

        self.do_multiply_255 = do_multiply_255

    def forward(self, x: torch.Tensor, goal: torch.Tensor = None):
        # r3m expects things to be [0-255] instead of [0-1]!!!
        if self.do_multiply_255:
            x = x * 255.0

        # some computational savings
        if self.freeze_r3m:
            with torch.no_grad():
                x = self.r3m(x)
        else:
            x = self.r3m(x)

        # Add state noise if needed
        if self.state_noise > 0.0:
            x += torch.randn_like(x).to(x.device) * self.state_noise

        if self.with_goal:
            # do it with the goal if we care about the goal
            if self.do_multiply_255:
                goal = goal * 255.0
            if self.freeze_r3m:
                with torch.no_grad():
                    goal = self.r3m(goal)
            else:
                goal = self.r3m(goal)

            # combine the state and goal (via film layer or concatenate)
            if self.film_layer_goal:
                gammabeta = self.film_layer(goal)
                gamma, beta = torch.split(gammabeta, self.r3m_embedding_dim, dim=1)
                x = x * gamma + beta
            else:
                # mix and run through head
                x = torch.cat([x, goal], dim=1)

        x = self.fc_head(x)
        return x

    def save_model(self, model_path: str):
        print(f"saved frame classification model to {model_path}")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path: str):
        print(f"loaded frame classification model from {model_path}")
        self.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    test = R3MPolicy()
    import pdb;

    pdb.set_trace()
