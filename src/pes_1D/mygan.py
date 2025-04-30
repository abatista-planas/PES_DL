import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_analytical_pes_samples,
    generate_generator_training_set_from_df,
)

# r = np.linspace(2.5, 10, 250, dtype=np.float64)
# r0 = np.linspace(2.59, 9.99, 1000, dtype=np.float64)
# values = PesModels.lennard_jones([3.0,200.0],r)

# v0 = PesModels.lennard_jones([3.0,200.0],r0)

# nearest = griddata(r, values, r0, method='nearest')
# linear = griddata(r, values, r0, method='linear')
# cubic = griddata(r, values, r0, method='cubic')

# rmse = lambda y,y0: np.sqrt(np.mean((y - y0)**2))


# print("RMSE nearest: ", rmse(nearest, v0))
# print("RMSE linear: ", rmse(linear, v0))
# print("RMSE cubic: ", rmse(cubic, v0))


# #plt.plot(r0, v0, label="Original")
# plt.plot(r0, nearest, label="nearest")
# #plt.plot(r0, linear, label="linear")
# plt.plot(r0, cubic, label="cubic")
# # plt.subplot(222)
# # plt.imshow(r0, nearest, label="nearest")
# # plt.title('Nearest')
# # plt.subplot(223)
# # plt.imshow(r0, linear, label="linear")
# # plt.title('Linear')
# # plt.subplot(224)
# # plt.imshow(r0, cubic, label="cubic")
# # plt.title('Cubic')
# # plt.gcf().set_size_inches(6, 6)
# plt.xlim(3, 10)
# plt.ylim(np.min(v0), 10)
# plt.show()


class SuperRes1D(nn.Module):
    def __init__(self, upscale_factor=4, base_channels=64):
        super().__init__()
        # 1) Learnable upsampling: ConvTranspose1d takes (C_in, C_out, kernel, stride, padding)
        #    Here we go from 1→base_channels, upsampling by factor=4.
        self.upconv = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=upscale_factor,
            stride=upscale_factor,
            padding=0,
            output_padding=0,
        )
        # 2) Refinement layers
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        x: (batch, 1, n)
        returns: (batch, 1, 4*n)
        """
        x = self.upconv(x)
        x = self.refine(x)
        return x


def get_performance(grid_size, up_scale):
    num_epochs = 1000
    size = grid_size
    up_scale = up_scale
    n_samples = 2000
    batch_size = 50
    # --- setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperRes1D(upscale_factor=up_scale, base_channels=batch_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- generate data ---

    parameters_array = np.zeros((n_samples, 2))
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples)  # sigma
    parameters_array[:, 1] = np.random.uniform(5.0, 100000, n_samples)  # epsilon

    df_high_res = generate_analytical_pes_samples(
        "lennard_jones", parameters_array, size * up_scale
    )

    train_loader, test_loader, shuffled_df, _ = generate_generator_training_set_from_df(
        df_high_res,
        batch_size=batch_size,
        up_scale=up_scale,
        properties_list=["energy"],
        properties_format="array",
        test_split=0.5,
        gpu=True,
    )

    # initialize losses
    trainAcc = []

    # loop over epochs
    for epochi in range(num_epochs):
        # switch on training mode
        model.train()

        # loop over training data batches
        running_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            # forward pass and loss

            y_pred = model.forward(X)

            loss = criterion(y_pred, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        avg_loss = running_loss / total_samples
        trainAcc.append(avg_loss)

    model.eval()
    with torch.no_grad():  # deactivates autograd
        for X, y in test_loader:
            y_pred = model.forward(X)
            loss = criterion(y_pred, y)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        test_avg_loss = running_loss / total_samples

    return trainAcc[-1], test_avg_loss


# n = 25
# loss = np.zeros((2, n, n))

# for i in range(4, n):
#     for j in range(4, n):
#         print(f"({i},{j})")
#         loss[0, i, j], loss[1, i, j] = get_performance(i, j)
#         print(f"Train Loss: {loss[0, i, j]}, Test Loss: {loss[1, i, j]}")


# fig, axes = plt.subplots(nrows=1, ncols=2)
# for i, ax in enumerate(axes if isinstance(axes, np.ndarray) else [axes]):
#     im = ax.imshow(loss[i, 4:n, 4:n], cmap="viridis")
#     ax.set_xticklabels([""] + list(np.arange(4, n)))
#     ax.set_yticklabels([""] + list(np.arange(4, n)))

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
# fig.colorbar(im, cax=cbar_ax)

# plt.show()


# # Save the plot as a PNG image
# plt.savefig("sample_plot.png")


def my_function(x):
    return x * x


if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(my_function, range(10))
    print(results)
