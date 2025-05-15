import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from scipy.interpolate import CubicSpline  # type: ignore

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.generator import ResNetUpscaler
from pes_1D.utils import Normalizers
import os



num_epochs = 2000
batch_size = 25
pes_name_list = ["lennard_jones", "morse"]
n_samples = [10000, 10000]
upscale_factor = 100
lr_grid_size = 8
hr_grid_size = lr_grid_size * upscale_factor
test_split = 0
lr_G = 1e-3

# -- Models, losses, optimizers --
G = ResNetUpscaler(upscale_factor=upscale_factor, num_channels=16, num_blocks=2)


mse_loss = nn.MSELoss()
optG = optim.Adam(G.parameters(), lr=lr_G)

df_high_res = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    hr_grid_size,
)

train_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_res,
    batch_size=batch_size,
    up_scale=upscale_factor,
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
    properties_format="table_1D",
    test_split=0,
)


def train(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    dataset = SineDataset(2000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = SimpleMLP().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        epoch_loss = 0.0

        for x_in, y_out in loader:
            x_in = x_in.to(device)
            y_out = y_out.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(x_in)
            loss = criterion(outputs, y_out)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Explicitly using 2 GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
