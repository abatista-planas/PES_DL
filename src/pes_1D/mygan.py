import os
import threading
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import griddata  # type: ignore
from sklearn.metrics import root_mean_squared_error as rmse  # type: ignore

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.generator import ResNetUpscaler

# --- setup ---
gpu = torch.cuda.is_available()


#######################################################
def check_generator_v3(model, pes_name_list, n_samples, grid_size, up_scale, device):

    size = grid_size * up_scale

    model.eval()
    total_samples = np.sum(n_samples)

    arr = np.zeros((total_samples, 2))

    df_high_res = generate_true_pes_samples(
        pes_name_list,
        n_samples,
        size,
    )

    _, _, _, input_lr, input_hr = generate_generator_training_set_from_df(
        df_high_res,
        batch_size=1,
        up_scale=up_scale,
        properties_list=["r", "energy"],
        properties_format="table_1D",
        test_split=0,
        device=device,
    )

    indices = torch.tensor([i for i in range(size) if i % up_scale != 0])

    for i in range(total_samples):
        r_input = input_lr[i, 0, 0, :].to("cpu").numpy()
        v_input = input_lr[i, 0, 1, :].to("cpu").numpy()
        r_output = input_hr[i, 0, 0, indices].to("cpu").numpy()
        v_output = input_hr[i, 0, 1, indices].to("cpu").numpy()

        input_model = input_lr[i, 0, 1, :].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            y_pred = model.forward(input_model)

        v_model = y_pred[0, 0, indices].detach().cpu().numpy()
        cubic_output = griddata(r_input, v_input, r_output, method="cubic")

        arr[i, 0] = rmse(v_output, v_model)
        arr[i, 1] = rmse(v_output, cubic_output)

    return (
        np.min(arr[:, 0]),
        np.max(arr[:, 0]),
        np.mean(arr[:, 0]),
        np.min(arr[:, 1]),
        np.max(arr[:, 1]),
        np.mean(arr[:, 1]),
    )


def get_performance(grid_size, up_scale, device):
    num_epochs = 1000
    batch_size = 50
    pes_name_list = ["lennard_jones", "morse"]
    n_samples = [10000, 10000]
    size = grid_size * up_scale
    test_split = 0.5

    print("Performance Initialized ", device)
    # model = Upscale1D(scale_factor=up_scale)
    model = ResNetUpscaler(upscale_factor=up_scale, num_channels=8, num_blocks=2)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- generate data ---
    df_high_res = generate_true_pes_samples(
        pes_name_list,
        n_samples,
        size,
    )

    train_loader, test_loader, _, _, _ = generate_generator_training_set_from_df(
        df_high_res,
        batch_size=batch_size,
        up_scale=up_scale,
        properties_list=["energy"],
        properties_format="array",
        test_split=test_split,
        device=device,
    )
    # --- train model ---
    _, train_avg_loss = model.train_model(
        train_loader,
        criterion,
        optimizer,
        num_epochs,
    )

    # --- test model ---
    test_avg_loss, _, _ = model.test_model(
        test_loader,
        criterion,
    )

    return train_avg_loss, test_avg_loss, model










def main(grid_size_arr:list[int], filename:str):
   
    scaling_arr =  [4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]


    print("GPU available: ", torch.cuda.is_available())
    print("CPU available: ", os.cpu_count())
    print("GPU count", torch.cuda.device_count())
    
    ni = len(grid_size_arr)
    nj = len(scaling_arr)

    res = np.zeros((10, ni, nj))
    
    def gpu_work(device, gs_arr, scl_arr, arr):

        count = 0
        for i, gz in enumerate(gs_arr):
            for j, scl in enumerate(scl_arr):
                count = count + 1
                loss_train, loss_test, model = get_performance(gz, scl, device)
                print(
                    f"count : {count}-- {device} -- {gz},{scl} RMSE Train Loss: {np.sqrt(loss_train)}, Test Loss: {np.sqrt(loss_test)}"
                )
                (
                    best_model,
                    worst_model,
                    mean_model,
                    best_spline,
                    worst_spline,
                    mean_spline,
                ) = check_generator_v3(
                    model,
                    ["lennard_jones", "morse"],
                    n_samples=[5000, 5000],
                    grid_size=gz,
                    up_scale=scl,
                    device=device,
                )
                print(f"RMSE Mean Model: {mean_model}, Spline: {mean_spline}")
                (
                    _,
                    _,
                    mean_model_O2,
                    _,
                    _,
                    mean_spline_O2,
                ) = check_generator_v3(
                    model,
                    ["reudenberg"],
                    n_samples=[1],
                    grid_size=gz,
                    up_scale=scl,
                    device=device,
                )
                print(f"RMSE for O2 Mean Model: {mean_model_O2}, Spline: {mean_spline_O2}")
                arr[0, i, j] = np.sqrt(loss_train)
                arr[1, i, j] = np.sqrt(loss_test)
                arr[2, i, j] = best_model
                arr[3, i, j] = worst_model
                arr[4, i, j] = mean_model
                arr[5, i, j] = best_spline
                arr[6, i, j] = worst_spline
                arr[7, i, j] = mean_spline
                arr[8, i, j] = mean_model_O2
                arr[9, i, j] = mean_spline_O2


    if torch.cuda.device_count() > 1:

   
        gs_arr_1 = grid_size_arr[: int(ni / 2)]
        gs_arr_2 = grid_size_arr[int(ni / 2) :]
        ni_1 = len(gs_arr_1)
        ni_2 = len(gs_arr_2)

        res_1 = np.zeros((10, ni_1, nj))
        res_2 = np.zeros((10, ni_2, nj))

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

        res[:, :ni_1, :] = res_1
        res[:, ni_1:, :] = res_2
        
        
        

    else:
        device = torch.device("cuda" if gpu else "cpu")
        gpu_work(device, grid_size_arr, scaling_arr, res)

 
    np.save(filename, res)




if __name__ == "__main__":
    
    n = len(sys.argv)
    if n < 3:
        print("Usage: python mygan.py <grid_size> <filename>")
        sys.exit(1) 

    main(grid_size_arr = list(map(int, sys.argv[1][1:-1].split(','))), 
         filename = "./results/gen_vs_spline"+str(sys.argv[2])+".npy"
         )   