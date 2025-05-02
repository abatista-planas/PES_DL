import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import griddata  # type: ignore
from sklearn.metrics import root_mean_squared_error as rmse  # type: ignore

from pes_1D.data_generator import (
    generate_analytical_pes_samples,
    generate_generator_training_set_from_df,
)
from pes_1D.generator import Upscale1D
from pes_1D.utils import PesModels

# --- setup ---
gpu = torch.cuda.is_available()


# def check_cubic_interpolation( pes_name = "lennard_jones",
#                                 n_max = 36,
#                                 n_samples = 1000,
#                                 ):

#     arr = np.zeros((n_max,n_samples))

#     parameters_array = np.zeros((n_samples, 2), dtype=np.float64)
#     parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples).astype(np.float64) # sigma
#     parameters_array[:, 1] = np.random.uniform(5, 100000, n_samples).astype(np.float64)

#     wall_max_high = np.random.uniform(1000, 2000, n_samples)
#     long_range_limit = np.random.uniform(0.01, 0.40, n_samples)

#     for n in range(4,n_max,2):
#         for i in range(n_samples):
#             # Get random parameters
#             parameters = parameters_array[i]
#             zero = getattr(PesModels, pes_name)(parameters, np.array(100))

#             def pes(r):
#                 return getattr(PesModels, pes_name)(parameters, r) - zero

#             r_trial = np.linspace(0.1, 50.0, 5000)
#             energy_array = pes(r_trial)

#             min_index = np.argmin(energy_array)
#             r_0 = r_trial[min_index]
#             approx_well_depth = energy_array[min_index]

#             max_high = wall_max_high[i]
#             long_range_max = abs(long_range_limit[i] * approx_well_depth)

#             # Find Proper boundary: Using bisection (requires a bracketing interval)
#             def f_min(r):
#                 return pes(r) - max_high

#             r_min = optimize.bisect(f_min, 0.001, 100)

#             def f_max(r):
#                 return abs(pes(r)) - long_range_max

#             r_max = optimize.bisect(f_max, r_0, 100)

#             r = np.linspace(r_min, r_max, n, dtype=np.float64)
#             r0 = np.linspace(r_min, r_max, 1000, dtype=np.float64)

#             v = PesModels.lennard_jones(parameters.tolist(), r)
#             v0 = PesModels.lennard_jones(parameters.tolist(), r0)

#             mx = np.max(v0)
#             mn = np.min(v0)

#             # Normalize the potential
#             v0 = (v0 - mn) / (mx - mn)
#             v = (v - mn) / (mx - mn)


#             # Interpolate the
#             cubic = griddata(r, v, r0, method='cubic')
#             arr[n,i] = rmse(cubic, v0)
#     return arr


# arr = check_cubic_interpolation()
# n_max = arr.shape[0]

# plt.errorbar(
#     np.arange(4, n_max, 2),
#     np.mean(arr[np.arange(4, n_max, 2), :], axis=1),
#     yerr=[
#         np.min(arr[np.arange(4, n_max, 2), :], axis=1),
#         np.max(arr[np.arange(4, n_max, 2), :], axis=1),
#     ],
#     fmt='ro--',
#     label="Cubic Interpolation",
# )

# plt.title('RMSE of Cubic Interpolation for Lennard-Jones Potential')
# plt.xlabel('Number of points')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()


def check_generator(
    model,
    n_pts,
    up_scale,
    device="cpu",
    pes_name="lennard_jones",
    n_samples=1000,
):
    model.eval()
    model = model.to(device, dtype=torch.float64)

    arr = np.zeros((n_samples))

    parameters_array = np.zeros((n_samples, 2), dtype=np.float64)
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples).astype(
        np.float64
    )  # sigma
    parameters_array[:, 1] = np.random.uniform(5, 100000, n_samples).astype(np.float64)

    wall_max_high = np.random.uniform(1000, 2000, n_samples)
    long_range_limit = np.random.uniform(0.01, 0.40, n_samples)

    for i in range(n_samples):
        # Get random parameters
        parameters = parameters_array[i]
        zero = getattr(PesModels, pes_name)(parameters, np.array(100))

        def pes(r):
            return getattr(PesModels, pes_name)(parameters, r) - zero

        r_trial = np.linspace(0.1, 50.0, 5000)
        energy_array = pes(r_trial)

        min_index = np.argmin(energy_array)
        r_0 = r_trial[min_index]
        approx_well_depth = energy_array[min_index]

        max_high = wall_max_high[i]
        long_range_max = abs(long_range_limit[i] * approx_well_depth)

        # Find Proper boundary: Using bisection (requires a bracketing interval)
        def f_min(r):
            return pes(r) - max_high

        r_min = optimize.bisect(f_min, 0.001, 100)

        def f_max(r):
            return abs(pes(r)) - long_range_max

        r_max = optimize.bisect(f_max, r_0, 100)

        input_r = np.linspace(r_min, r_max, n_pts, dtype=np.float64)
        output_r = np.linspace(r_min, r_max, n_pts * up_scale, dtype=np.float64)
        r0 = np.linspace(r_min, r_max, 1000, dtype=np.float64)

        v = PesModels.lennard_jones(parameters.tolist(), input_r)
        input_v = (
            torch.from_numpy(v)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device, dtype=torch.float64)
        )

        v0 = PesModels.lennard_jones(parameters.tolist(), r0)

        mx = np.max(v0)
        mn = np.min(v0)

        # Normalize the potential
        v0 = (v0 - mn) / (mx - mn)
        input_v = (input_v - mn) / (mx - mn)

        with torch.no_grad():
            y_pred = model.forward(input_v)
        output_v = y_pred[0, 0, :].detach().cpu().numpy()

        cubic = griddata(output_r, output_v, r0, method="cubic")
        arr[i] = rmse(cubic, v0)
    return np.min(arr), np.max(arr), np.mean(arr)


def get_performance(grid_size, up_scale, device):
    num_epochs = 1000
    size = grid_size
    up_scale = up_scale
    n_samples = 5000
    batch_size = 16

    print("Performance Initialized ", device)
    # model = SuperResolution1D(upscale_factor=up_scale, base_channels=batch_size)
    model = Upscale1D(grid_size=grid_size, scale_factor=up_scale)
    # model = SmoothUpscale1D(input_points=grid_size ,upscale_factor=up_scale)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- generate data ---

    parameters_array = np.zeros((n_samples, 2))
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples)  # sigma
    parameters_array[:, 1] = np.random.uniform(5.0, 100000, n_samples)  # epsilon

    df_high_res = generate_analytical_pes_samples(
        "lennard_jones", parameters_array, size * up_scale, device=device
    )

    train_loader, test_loader, shuffled_df, _ = generate_generator_training_set_from_df(
        df_high_res,
        batch_size=batch_size,
        up_scale=up_scale,
        properties_list=["energy"],
        properties_format="array",
        test_split=0.5,
        device=device,
    )

    loss_arr, train_avg_loss = model.train_model(
        train_loader,
        criterion,
        optimizer,
        num_epochs,
    )

    plt.plot(loss_arr)
    plt.title("Training Loss")
    plt.show()

    test_avg_loss, _, _ = model.test_model(
        test_loader,
        criterion,
    )

    return train_avg_loss, test_avg_loss, model


n = 18
res = np.zeros((5, n, n))

print("GPU available: ", torch.cuda.is_available())
print("CPU available: ", os.cpu_count())
print("GPU count", torch.cuda.device_count())

# # num gpu_count = torch.cuda.device_count()
# # for i in range(gpu_count):


def gpu_work(device, initial, final):
    count = 0

    for i in range(initial, final):
        for j in range(2, n):
            count = count + 1
            loss_train, loss_test, model = get_performance(2 * i, 2 * j, device)
            print(
                f"count : {count} -- {2*i},{2*j} RMSE Train Loss: {np.sqrt(loss_train)}, Test Loss: {np.sqrt(loss_test)}"
            )
            best, worst, mean = check_generator(
                model, 2 * i, 2 * j, device, "lennard_jones", n_samples=1000
            )
            print(f"RMSE Best: {best}, Worst: {worst}, Mean: {mean}")
            res[0, i, j] = np.sqrt(loss_train)
            res[1, i, j] = np.sqrt(loss_test)
            res[2, i, j] = best
            res[3, i, j] = worst
            res[4, i, j] = mean


# # device = torch.device("cuda" if gpu else "cpu")
# # gpu_work(device, 2, n)


# # thread1 = threading.Thread(target = gpu_work, args=(torch.device('cuda:0'),2 , int(n/2)))
# # thread2 = threading.Thread(target = gpu_work, args=(torch.device('cuda:1'),int(n/2) , n))


# # thread1.start()
# # thread2.start()

# # thread1.join()
# # thread2.join()


# # count = 0
# # for ngpu in range(gpu_count):
# #     device = torch.device("cuda" if gpu else "cpu")
# #     for i in range(2,n):
# #         for j in range(2, n):
# #             count = count + 1
# #             loss_train, loss_test,model = get_performance(2*i,2*j)
# #             print(f"count : {count} -- {2*i},{2*j} RMSE Train Loss: {np.sqrt(loss_train)}, Test Loss: {np.sqrt(loss_test)}")
# #             best,worst,mean = check_generator(model, 2*i , 2*j, "lennard_jones", n_samples=1000)
# #             print(f"RMSE Best: {best}, Worst: {worst}, Mean: {mean}")
# #             res[0,i,j] =  np.sqrt(loss_train)
# #             res[1,i,j] =  np.sqrt(loss_test)
# #             res[2,i,j] =  best
# #             res[3,i,j] =  worst
# #             res[4,i,j] =  mean


# # np.save("res_2.npy", res)
count = 1
device = torch.device("cuda" if gpu else "cpu")
i = 8
j = 2

loss_train, loss_test, model = get_performance(2 * i, 2 * j, device)
print(
    f"count : {count} -- {2*i},{2*j} RMSE Train Loss: {np.sqrt(loss_train)}, Test Loss: {np.sqrt(loss_test)}"
)
best, worst, mean = check_generator(
    model, 2 * i, 2 * j, device, "lennard_jones", n_samples=1000
)
print(f"RMSE Best: {best}, Worst: {worst}, Mean: {mean}")
