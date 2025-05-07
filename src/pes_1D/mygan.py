import os
import threading
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


def check_cubic_interpolation(
    pes_name="lennard_jones",
    n_max=36,
    n_samples=1000,
):

    arr = np.zeros((n_max, n_samples))

    parameters_array = np.zeros((n_samples, 2), dtype=np.float64)
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples).astype(
        np.float64
    )  # sigma
    parameters_array[:, 1] = np.random.uniform(5, 100000, n_samples).astype(np.float64)

    wall_max_high = np.random.uniform(1000, 2000, n_samples)
    long_range_limit = np.random.uniform(0.01, 0.40, n_samples)

    for n in range(4, n_max, 2):
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

            r = np.linspace(r_min, r_max, n, dtype=np.float64)
            r0 = np.linspace(r_min, r_max, 1000, dtype=np.float64)

            v = PesModels.lennard_jones(parameters.tolist(), r)
            v0 = PesModels.lennard_jones(parameters.tolist(), r0)

            mx = np.max(v0)
            mn = np.min(v0)

            # Normalize the potential
            v0 = (v0 - mn) / (mx - mn)
            v = (v - mn) / (mx - mn)

            # Interpolate the
            cubic = griddata(r, v, r0, method="cubic")
            arr[n, i] = rmse(cubic, v0)
    return arr


# arr = check_cubic_interpolation()

# np.save("cubic_rmse.npy", arr)


def evaluate_pes(model, pes_name, n_pts, up_scale, device, parameters):

    # Get random parameters

    zero = getattr(PesModels, pes_name)(parameters, np.array(100))

    def pes(r):
        return getattr(PesModels, pes_name)(parameters, r) - zero

    r_trial = np.linspace(0.1, 50.0, 5000)
    energy_array = pes(r_trial)

    min_index = np.argmin(energy_array)
    r_0 = r_trial[min_index]
    approx_well_depth = energy_array[min_index]

    max_high = np.max(
        np.random.uniform(1000.0, 2000.0) * np.random.uniform(1.0, 3.0) * parameters[1]
    )
    long_range_max = abs(np.random.uniform(0.01, 0.40) * approx_well_depth)

    # Find Proper boundary: Using bisection (requires a bracketing interval)
    def f_min(r):
        return pes(r) - max_high

    r_min = optimize.bisect(f_min, 0.001, 100)

    def f_max(r):
        return abs(pes(r)) - long_range_max

    r_max = optimize.bisect(f_max, r_0, 100)

    input_r = np.linspace(r_min, r_max, n_pts, dtype=np.float64)
    output_r = np.linspace(r_min, r_max, n_pts * up_scale, dtype=np.float64)
    # r0 = np.linspace(r_min, r_max, 1000, dtype=np.float64)

    v_in = PesModels.lennard_jones(parameters.tolist(), input_r)
    v_out = PesModels.lennard_jones(parameters.tolist(), output_r)
    input_v = (
        torch.from_numpy(v_in).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float64)
    )

    # v0 = PesModels.lennard_jones(parameters.tolist(), r0)

    mx = np.max(v_out)
    mn = np.min(v_out)

    # Normalize the potential
    v_out = (v_out - mn) / (mx - mn)
    input_v = (input_v - mn) / (mx - mn)

    with torch.no_grad():
        y_pred = model.forward(input_v)

    output_v = y_pred[0, 0, :].detach().cpu().numpy()
    input_v = input_v[0, 0, :].detach().cpu().numpy()

    cubic_output = griddata(input_r, input_v, output_r, method="cubic")
    return rmse(output_v, v_out), rmse(cubic_output, v_out)


def check_generator_v2(
    model,
    n_pts,
    up_scale,
    device="cpu",
    pes_name="lennard_jones",
    n_samples=1000,
):
    model.eval()
    model = model.to(device, dtype=torch.float64)

    arr = np.zeros((n_samples, 2))

    parameters_array = np.zeros((n_samples, 2), dtype=np.float64)
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples).astype(
        np.float64
    )  # sigma
    parameters_array[:, 1] = np.random.uniform(5, 100000, n_samples).astype(np.float64)

    def task(params):
        return evaluate_pes(model, pes_name, n_pts, up_scale, device, params)

    for i in range(n_samples):
        # Get random parameters
        arr[i, 0], arr[i, 1] = task(parameters_array[i])

    return (
        np.min(arr[:, 0]),
        np.max(arr[:, 0]),
        np.mean(arr[:, 0]),
        np.min(arr[:, 1]),
        np.max(arr[:, 1]),
        np.mean(arr[:, 1]),
    )


def get_performance(grid_size, up_scale, device):
    num_epochs = 500
    size = grid_size
    up_scale = up_scale
    n_samples = 10000
    batch_size = 25

    print("Performance Initialized ", device)
    model = Upscale1D(scale_factor=up_scale)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- generate data ---

    parameters_array = np.zeros((n_samples, 2))
    parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples)  # sigma
    parameters_array[:, 1] = np.random.uniform(5.0, 100000, n_samples)  # epsilon

    df_high_res = generate_analytical_pes_samples(
        "lennard_jones", parameters_array, size * up_scale
    )

    train_loader, test_loader, _, _ = generate_generator_training_set_from_df(
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

    # plt.plot(loss_arr[10:])
    # plt.title("Training Loss")
    # plt.show()

    test_avg_loss, _, _ = model.test_model(
        test_loader,
        criterion,
    )

    return train_avg_loss, test_avg_loss, model


grid_size_arr = [4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
scaling_arr = [4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]


print("GPU available: ", torch.cuda.is_available())
print("CPU available: ", os.cpu_count())
print("GPU count", torch.cuda.device_count())


def gpu_work(device, gs_arr, scl_arr, arr):

    count = 0
    for i, gz in enumerate(gs_arr):
        for j, scl in enumerate(scl_arr):
            count = count + 1
            loss_train, loss_test, model = get_performance(gz, scl, device)
            print(
                f"count : {count} -- {gz},{scl} RMSE Train Loss: {np.sqrt(loss_train)}, Test Loss: {np.sqrt(loss_test)}"
            )
            (
                best_model,
                worst_model,
                mean_model,
                best_spline,
                worst_spline,
                mean_spline,
            ) = check_generator_v2(
                model, gz, scl, device, "lennard_jones", n_samples=1000
            )
            print(f"RMSE Mean Model: {mean_model}, Spline: {mean_spline}")

            arr[0, i, j] = np.sqrt(loss_train)
            arr[1, i, j] = np.sqrt(loss_test)
            arr[2, i, j] = best_model
            arr[3, i, j] = worst_model
            arr[4, i, j] = mean_model
            arr[5, i, j] = best_spline
            arr[6, i, j] = worst_spline
            arr[7, i, j] = mean_spline


if torch.cuda.device_count() > 1:
    
        
    ni = len(grid_size_arr)
    gs_arr_1 = grid_size_arr[:int(ni/2)]
    gs_arr_2 = grid_size_arr[int(ni/2):]
    ni_1 = len(gs_arr_1)
    ni_2 = len(gs_arr_2)
    nj = len(scaling_arr)

    res = np.zeros((8, ni, nj))
    res_1 = np.zeros((8,ni_1 , nj))
    res_2 = np.zeros((8, ni_2, nj))
    
    thread1 = threading.Thread(
        target=gpu_work, args=(torch.device("cuda:0"), gs_arr_1, scaling_arr, res_1)
    )
    thread2 = threading.Thread(
        target=gpu_work, args=(torch.device("cuda:1"), gs_arr_2, scaling_arr, res_2)
    )

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    
    res[:, : ni_1, :] = res_1
    res[:, ni_1 :, :] = res_2
    np.save("res.npy", res)

else:
    print("hello")
    ni = len(grid_size_arr)
    nj = len(scaling_arr)

    res = np.zeros((8, ni, nj))
    device = torch.device("cuda" if gpu else "cpu")

    gpu_work(device, grid_size_arr, scaling_arr, res)

    np.save("res.npy", res)
