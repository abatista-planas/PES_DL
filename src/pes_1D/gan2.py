import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.generator import ResNetUpscaler


# -- Discriminator: MLP that maps 150→1 (real/fake) --
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(150, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [batch, 150]
        return self.net(x).view(-1)  # → [batch]


# -- Hyperparameters --
batch_size = 32
lr_D, lr_G = 2e-4, 1e-3
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Models, losses, optimizers --
G = ResNetUpscaler(upscale_factor=30, num_channels=16, num_blocks=2).to(device)
D = Discriminator().to(device)
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
optD = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
optG = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))


# 1) Dataset: low-res and high-res data
# --- generate data ---
num_epochs = 1000
batch_size = 25
pes_name_list = ["lennard_jones", "morse"]
n_samples = [5000, 5000]
size = 150
test_split = 0.5

df_high_res = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    size,
)

train_loader, test_loader, _, _, _ = generate_generator_training_set_from_df(
    df_high_res,
    batch_size=batch_size,
    up_scale=30,
    properties_list=["energy"],
    properties_format="table_1D",
    test_split=test_split,
    device=device,
)

print("Training beginning...")
# -- Training loop --
for epoch in range(epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input
    for pes_low, pes_high in train_loader:

        # grid_hr = torch.linspace(0, 1, 150, device=device)            # [150]
        # sin_hr = torch.sin(2*torch.pi * grid_hr)                     # [150]
        # sin_hr = sin_hr.unsqueeze(0).repeat(batch_size,1)            # [B,150]
        # sin_lr = sin_hr[:, :10]

        # print(sin_hr.shape)

        hr = pes_high.squeeze(1)  # [B,150]
        lr = pes_low.squeeze(1)  # [B,10]

        # Labels
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # -- Train Discriminator on real --
        D.zero_grad()
        out_real = D(hr.squeeze(1))  # [B]
        lossD_real = bce_loss(out_real, real_labels)
        lossD_real.backward()

        # -- Train Discriminator on fake --
        fake_hr = G(lr).squeeze(1).detach()  # [B,150]
        out_fake = D(fake_hr)  # [B]
        lossD_fake = bce_loss(out_fake, fake_labels)
        lossD_fake.backward()
        optD.step()

        # -- Train Generator --
        G.zero_grad()
        gen_hr = G(lr)  # [B,150]
        # 1) adversarial loss (wants D(gen_hr)==1)
        adv_loss = bce_loss(D(gen_hr), real_labels)
        # 2) pixel-wise MSE against true high-res sine
        pixel_loss = mse_loss(gen_hr, hr)
        lossG = adv_loss + pixel_loss  # combine
        lossG.backward()
        optG.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | G_adv_loss: {adv_loss}, G_pixel_loss: {pixel_loss} | D_loss: {lossD_real + lossD_fake}"
        )
