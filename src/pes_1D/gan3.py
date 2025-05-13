import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline  # type: ignore

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator
from pes_1D.generator import ResNetUpscaler
from pes_1D.utils import Normalizers

num_epochs = 2000
batch_size = 50
pes_name_list = ["lennard_jones", "morse"]
n_samples = [2500, 2500]
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
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
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


# -- Hyperparameters --
lr_D, lr_G = 1e-3, 1e-3
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Models, losses, optimizers --
G = ResNetUpscaler(upscale_factor=30, num_channels=16, num_blocks=2).to(device)

D = CnnDiscriminator(
    {
        "in_channels": 3,  # energy, derivative, inverse_derivative
        "grid_size": hr_grid_size,
        "hidden_channels": [16, 32],
        "kernel_size": [3, 3],
        "pool_size": [2, 2],
    }
).to(device)


bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
optD = optim.Adam(D.parameters(), lr=lr_D)
optG = optim.Adam(G.parameters(), lr=lr_G)

D.pre_train(bce_loss, n_samples=[10000], num_epochs=1000)


def df(r_input, v_input):

    cs = CubicSpline(r_input, v_input, extrapolate=True)
    deriv = cs.derivative(1)(r_input)
    # Normalize the derivative
    return Normalizers.normalize(deriv).reshape(deriv.shape)


def prepare_discriminator_data(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepare discriminator data for training.
    """

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
    return output_tensor


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

r_hr = df_high_test.iloc[0].pes["r"].to_numpy()
energy_hr = df_high_test.iloc[0].pes["energy"].to_numpy()
indices = np.arange(0, energy_hr.shape[0] - 1, upscale_factor)

energy_lr = energy_hr[indices]
r_lr = r_hr[indices]

plt.plot(r_hr, energy_hr, label="hr-grid", color="blue")
plt.scatter(r_lr, energy_lr, label="lr-grid", color="g")


(line1,) = ax.plot(
    r_hr, np.zeros_like(r_hr), "r.--", label="model"
)  # Returns a tuple of line objects, thus the comma

plt.xlabel("r")
plt.ylabel("Energy")
plt.title("Sample TEST PES")
plt.legend()


def eval_models():
    G.eval()
    D.eval()
    input_tensor_D = torch.zeros((1, 2, r_hr.shape[0]), dtype=torch.float32)
    input_tensor_D[0, 0, :] = torch.from_numpy(r_hr)

    energy_pred = G(
        torch.from_numpy(energy_lr).float().unsqueeze(0).unsqueeze(1).to(device)
    )

    fake_hr = energy_pred.detach().cpu()  # [B,150]
    input_tensor_D[:, 1, :] = fake_hr[0, 0, :]

    fake_hr_np = fake_hr.squeeze(0).squeeze(0).numpy()

    loss_fn = nn.MSELoss()
    test_mse = 0
    with torch.no_grad():  # deactivates autograd
        for X, y in test_loader:
            y_pred = G.forward(X)
            loss = loss_fn(y_pred, y)
            test_mse += loss.item()

    # # -- Train Discriminator on fake --

    fake_pes = prepare_discriminator_data(input_tensor_D).to(device)

    out_fake = D(fake_pes)
    D_says = out_fake < 0
    test_rmse = np.sqrt(np.mean((fake_hr_np - energy_hr) ** 2))
    if D_says:
        new_title = f"D caught G ({out_fake.item():.2e}) / rmse: {test_rmse:.2e}"
    else:
        new_title = f"D get fooled ({out_fake.item():.2e}) / rmse: {test_rmse:.2e}"

    ax.set_title(new_title)
    line1.set_ydata(fake_hr_np)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return np.sqrt(test_mse)


test_rmse = eval_models()
print(f"Test RMSE: {test_rmse:.2e}")


print("Training beginning...")
# -- Training loop --
for epoch in range(epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input

    fakeAcc = []
    initial_params = [param.clone() for param in D.parameters()]

    for pes_low, pes_high in train_loader:
        G.train()
        D.train()

        hr = pes_high.squeeze(1)  # [B,150]
        lr = pes_low.squeeze(1)  # [B,10]

        input_tensor = hr[:, :2, :].clone().detach().cpu()  # [B,150]

        # Labels
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # -- Train Discriminator on real --
        # if epoch >1 and epoch % 2 == 0:
        D.zero_grad()

        out_real = D(hr[:, 1:, :])  # [B]
        lossD_real = bce_loss(out_real, real_labels)
        lossD_real.backward()

        # -- Train Discriminator on fake --
        fake_hr = G(lr[:, 1, :].unsqueeze(1)).detach().cpu()  # [B,150]
        input_tensor[:, 1, :] = fake_hr[:, 0, :]
        fake_pes = prepare_discriminator_data(input_tensor).to(device)

        out_fake = D(fake_pes)  # D(fake_hr)  # [B]
        lossD_fake = bce_loss(out_fake, fake_labels)
        lossD_fake.backward()
        optD.step()

        if epoch % 5 == 0:
            # -- Train Generator --
            G.zero_grad()
            gen_hr = G(lr[:, 1, :].unsqueeze(1))  # [B,150]

            # 1) adversarial loss (wants D(gen_hr)==1)
            input_tensor[:, 1, :] = gen_hr[:, 0, :]
            fake_pes = prepare_discriminator_data(input_tensor).to(device)
            D_fake = D(fake_pes)

            fakeAcc.append(
                100 * torch.mean(((D_fake < 0) == fake_labels).float()).item()
            )

            adv_loss = bce_loss(D_fake, real_labels)

            # 2) pixel-wise MSE against true high-res sine
            mse = mse_loss(gen_hr, hr[:, 1, :].unsqueeze(1))
            mse_weight = 10**5
            lossG = adv_loss + mse_weight * mse  # combine
            lossG.backward()
            optG.step()

    trained_params = [param.clone() for param in D.parameters()]

    for initial, trained in zip(initial_params, trained_params):
        if torch.equal(initial, trained):
            print("D Model has not been trained.")

    if epoch % 10 == 0:
        test_mse = eval_models()

        print(
            f"Epoch {epoch} | G_adv_loss: {adv_loss.item():.2e}, rmse: {np.sqrt(mse.item()): .2e} | test_rmse: {test_mse:.2e} | D_acc_over_G : {float(np.mean(fakeAcc)):.3f} "
        )
