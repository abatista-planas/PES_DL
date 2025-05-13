import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator2
from pes_1D.generator import ResNetUpscaler

num_epochs = 2000
batch_size = 25
pes_name_list = ["lennard_jones", "morse"]
n_samples = [25, 25]
upscale_factor = 30
lr_grid_size = 8
hr_grid_size = lr_grid_size * upscale_factor
test_split = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_high_res = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    hr_grid_size,
)

train_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_res,
    batch_size=batch_size,
    up_scale=upscale_factor,
    # properties_list=["r","energy","derivative","inverse_derivative"],
    properties_list=["energy"],
    properties_format="table_1D",
    test_split=0,
    device=device,
)

df_high_test = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    hr_grid_size,
)

test_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_test,
    batch_size=len(df_high_test),
    up_scale=upscale_factor,
    properties_list=["energy"],
    properties_format="array",
    test_split=0,
    device=device,
)


# # -- Discriminator: MLP that maps 240→1 (real/fake) --
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(240, 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # x: [batch, 150]
#         return self.net(x).view(-1)  # → [batch]


# -- Hyperparameters --
lr_D, lr_G = 2e-4, 1e-3
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Models, losses, optimizers --
G = ResNetUpscaler(upscale_factor=30, num_channels=16, num_blocks=2).to(device)
# D = Discriminator().to(device)

D = CnnDiscriminator2(
    {
        "in_channels": 1,  # energy, derivative, inverse_derivative
        "grid_size": hr_grid_size,
        "hidden_channels": [16, 32],
        "kernel_size": [3, 3],
        "pool_size": [2, 2],
    }
).to(device)

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
optD = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
optG = optim.Adam(G.parameters(), lr=lr_G)


print("Training beginning...")
# -- Training loop --
for epoch in range(epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input
    initial_params = [param.clone() for param in G.parameters()]
    for pes_low, pes_high in train_loader:

        hr = pes_high.squeeze(1)  # [B,150]
        lr = pes_low.squeeze(1)  # [B,10]

        # Labels
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # -- Train Discriminator on real --
        D.zero_grad()

        out_real = D(hr)  # [B]
        lossD_real = bce_loss(out_real, real_labels)
        lossD_real.backward()

        # -- Train Discriminator on fake --
        fake_hr = G(lr).detach()  # [B,150]
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
        lossG = adv_loss  # + 10**4 *pixel_loss  # combine
        lossG.backward()
        optG.step()

        # Check if parameters have changed
        trained_params = [param.clone() for param in G.parameters()]

        for initial, trained in zip(initial_params, trained_params):
            if not torch.equal(initial, trained):
                print("Model has been trained.")
                break
        else:
            print("Model has not been trained.")

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | G_adv_loss: {adv_loss}, G_pixel_loss: {pixel_loss} | D_loss: {lossD_real + lossD_fake}"
        )
