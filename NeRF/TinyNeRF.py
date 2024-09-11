import torch
import numpy as np
from typing import Tuple, Optional


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ = np.load(path)
    return _["images"], _["poses"], _["focal"]


def create_rays(width: int, height: int, focal: float, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ii, jj = torch.meshgrid(torch.arange(width).to(pose), torch.arange(height).to(pose))
    ii.transpose_(-1, -2)
    jj.transpose_(-1, -2)
    directions = torch.stack([(ii - width * .5) / focal, -(jj - height * .5) / focal, -torch.ones_like(ii)], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)
    ray_origins = pose[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


def create_sample_points(ray_origins: torch.Tensor, ray_directions: torch.Tensor, near: float, far: float, samples: int, randomize: Optional[bool] = True) -> (torch.Tensor, torch.Tensor):
    pass


class VeryTinyNerfModel(torch.nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


### Position Encoding
def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def tiny_nerf_iteration(width: int, height: int, focal: float, pose: torch.Tensor):
    ray_origins, ray_directions = create_rays(width, height, focal, pose)


if __name__ == '__main__':
    LR = 5e-3
    NEF = 6
    DEVICE = torch.device("cuda")

    IMAGES, POSES, FOCAL = load_data("tiny_nerf_data.npz")
    MODEL = VeryTinyNerfModel(num_encoding_functions=NEF)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR)

    for iter in range(1000):
        _idx = np.random.randint(IMAGES.shape[0])
        _img = torch.from_numpy(IMAGES[_idx]).to(DEVICE)
        _pose = torch.from_numpy(POSES[_idx]).to(DEVICE)
