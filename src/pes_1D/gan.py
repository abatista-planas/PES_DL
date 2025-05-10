import numpy as np
import torch
from scipy.interpolate import CubicSpline  # type: ignore
from torch import nn

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator  # type: ignore
from pes_1D.generator import ResNetUpscaler  # type: ignore
from pes_1D.utils import Normalizers

# type: ignore


# # 1) Dataset: low-res and high-res data
# # --- generate data ---
num_epochs = 1000
batch_size = 50
pes_name_list = ["lennard_jones", "morse"]
n_samples = [10000, 10000]
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

train_loader, test_loader, _, _, _ = generate_generator_training_set_from_df(
    df_high_res,
    batch_size=batch_size,
    up_scale=upscale_factor,
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
    properties_format="table_1D",
    test_split=test_split,
    device=device,
)


# # G = torch.load("/home/albplanas/Desktop/Programming/PES_DL/PES_DL/saved_models/"
# #                     + "CNN_Generator.pth", weights_only=False).to(device)

G = ResNetUpscaler(upscale_factor=upscale_factor, num_channels=32, num_blocks=3).to(
    device
)

# D = CnnDiscriminator(model_parameters).to(device)
# # 3) Discriminator: judges real vs. fake high-res conditioned on low-res
D = CnnDiscriminator(
    {
        "in_channels": 3,  # energy, derivative, inverse_derivative
        "grid_size": hr_grid_size,
        "hidden_channels": [16, 32],
        "kernel_size": [3, 3],
        "pool_size": [2, 2],
    }
).to(device)


D.pre_train()


def df(r_input, v_input):

    cs = CubicSpline(r_input, v_input, extrapolate=True)
    deriv = cs.derivative(1)(r_input)
    # Normalize the derivative
    return Normalizers.normalize(deriv).reshape(deriv.shape)


def prepare_discriminator_data(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepare discriminator data for training.
    """
    # y = input_tensor.clone()

    output_tensor = torch.zeros((input_tensor.shape[0], 3, input_tensor.shape[-1]))
    output_tensor[:, 0, :] = input_tensor[:, 1, :]
    in_tensor = input_tensor.detach().cpu().numpy()

    for i in range(in_tensor.shape[0]):
        r_input = in_tensor[i, 0, :]
        v_input = in_tensor[i, 1, :]

        deriv = df(r_input, v_input)
        inverse_deriv = Normalizers.normalize(1 / np.array(deriv))
        output_tensor[i, 1, :] = torch.tensor(deriv)
        output_tensor[:, 2, :] = torch.tensor(inverse_deriv)

    # with Pool(12) as p:
    #     deriv = p.map(df, in_tensor)

    return output_tensor


# 4) Training loop
def train_gan(G, D, train_loader, epochs=500, device="cpu"):

    G = G.to(device)
    D = D.to(device)

    G.train()
    D.train()

    criterion_G = nn.MSELoss()
    criterion_D = nn.BCEWithLogitsLoss()

    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(epochs):
        fakeAcc = []
        trueAcc = []
        for pes_low, pes_high in train_loader:

            pes_low, pes_high = pes_low.to(device), pes_high.to(device)
            bs = pes_low.size(0)

            gen_output = G.forward(pes_low[:, 0, 1, :].unsqueeze(1))

            input_tensor = torch.zeros((gen_output.shape[0], 2, gen_output.shape[-1]))

            input_tensor[:, 0, :] = pes_high[:, 0, 0, :]
            input_tensor[:, 1, :] = gen_output[:, 0, :].clone().detach()

            # start1 = time.time()
            fake_pes = prepare_discriminator_data(input_tensor).to(device)
            # end1 = time.time()

            if epoch > 10000:

                true_pes = pes_high[:, 0, 1:, :].to(device)
                true_pred = D.forward(true_pes)

                D_true_loss = criterion_D(true_pred, torch.ones((bs, 1)).to(device))

                fake_pred = D.forward(fake_pes)
                D_fake_loss = criterion_D(fake_pred, torch.zeros((bs, 1)).to(device))

                loss_D = (D_true_loss + D_fake_loss) * 0.5

                #  # Backprop + optimize D
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            # # -- Generator step --
            # # -- Train Generator --
            # start2 = time.time()

            true_pred = D.forward(pes_high[:, 0, 1:, :])
            trueAcc.append(
                100
                * torch.mean(
                    ((true_pred > 0) == torch.ones((bs, 1)).to(device)).float()
                ).item()
            )

            fake_pred = D.forward(fake_pes)
            fakeAcc.append(
                100
                * torch.mean(
                    ((fake_pred > 0) == torch.zeros((bs, 1)).to(device)).float()
                ).item()
            )

            true_output = pes_high[:, 0, 1, :].unsqueeze(1).to(device)
            mse_G = criterion_G(gen_output, true_output)
            Df_loss = criterion_D(fake_pred, torch.ones((bs, 1)).to(device))

            lmda_D = 0.0001
            lmda_MSE = 100
            loss_G = lmda_MSE * mse_G + lmda_D * Df_loss

            # backprop
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # end2 = time.time()
            # print(end2 - start2,end1 - start1)

        if epoch % 10 == 0:
            msg = (
                f"Epoch {epoch}/{epochs}"
                "| Disc ({float(np.mean(trueAcc)):.2f} ; {float(np.mean(fakeAcc)):.2f})"
                "| rmse_G: {np.sqrt(mse_G.item()):.2e}"
                "| mse_G: {lmda_MSE*mse_G.item():.2e}"
                "| Df_loss: {lmda_D*Df_loss.item():.2e}"
                "| G_loss: {loss_G.item():.2e}"
            )
            # sys.stdout.write("\r" + msg)
            print(msg)

    return G, D


# 5) Usage example
if __name__ == "__main__":

    G_model, D_model = train_gan(G, D, train_loader, epochs=num_epochs, device=device)
