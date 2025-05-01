import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_analytical_pes_samples,
    generate_generator_training_set_from_df,
)
from pes_1D.generator import SuperResolution1D

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


def get_performance(grid_size, up_scale):
    num_epochs = 1000
    size = grid_size
    up_scale = up_scale
    n_samples = 2000
    batch_size = 50
    # --- setup ---
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    model = SuperResolution1D(upscale_factor=up_scale, base_channels=batch_size).to(
        device
    )

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
        gpu=gpu,
    )

    _, train_avg_loss = model.train_model(
        train_loader,
        criterion,
        optimizer,
        num_epochs,
    )

    test_avg_loss, _, _ = model.test_model(
        test_loader,
        criterion,
    )

    return train_avg_loss, test_avg_loss


n = 4
loss = np.zeros((2, n, n))


print("GPU available: ", torch.cuda.is_available())
print("CPU available: ", os.cpu_count())


loss = np.zeros((2, n, n))
num_processors = os.cpu_count()

for i in range(2, n):
    for j in range(2, n):
        print(f"Grid size: {2 * i}, Upscale: {2 * j}")
        loss[0, i, j], loss[1, i, j] = get_performance(2 * i, 2 * j)
        print(f"Train Loss: {loss[0, i, j]}, Test Loss: {loss[1, i, j]}")

np.save("loss.npy", loss)
