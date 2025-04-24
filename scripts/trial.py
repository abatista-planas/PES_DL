from functools import partial
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from pes_1D.discriminator import AnnDiscriminator # type: ignore
from pes_1D.data_generator import generate_true_pes_samples  # type: ignore
from pes_1D.data_generator import generate_bad_samples  # type: ignore
from pes_1D.training import test_model  # type: ignore
# import optuna
# from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch

n_samples = [4000]
grid_size = 150
batch_size = 50
arr_split = [0.25, 0.25, 0.5 ]
pes_name_list = ["lennard_jones"]
properties_list = [
    "energy",
    "derivative",
    "inverse_derivative",
]  # List of properties to use for training
deformation_list = np.array(["outliers", "oscillation"])  # Types of deformation to generate
properties_format = "array"  # Format [concatenated array or table] of properties to use for training
gpu = True


def generate_data():
    df_real_pes = generate_true_pes_samples(pes_name_list, n_samples, grid_size)

    df_fake_pes = generate_bad_samples(
        pes_name_list=pes_name_list,
        n_samples=n_samples,
        deformation_list = deformation_list,
        size=grid_size,
    )

    return pd.concat([df_real_pes,df_fake_pes] ,axis=0, ignore_index=True)

def load_data(df):
    

    label_tensor = torch.tensor(df["true_pes"].values, dtype=torch.float).cuda()
    label_tensor = label_tensor[:, None]

    def input_format(pes):
        """Define input format"""
        if properties_format == "array":
            return pes[properties_list].values.flatten()
        else:
            return pes[properties_list].values

    # Convert continuous variables to a tensor
    input_arrays = [input_format(pes) for pes in df["pes"].values]
    input_stack = np.stack(input_arrays)
    input_tensor = torch.tensor(input_stack, dtype=torch.float).cuda()

    train_size = int(arr_split[0] * len(input_tensor))
    val_size = int(arr_split[1] * len(input_tensor))
    test_size = len(input_tensor) - train_size - val_size
    
    dataset = TensorDataset(input_tensor, label_tensor)

    # Use random_split to create the datasets
    return random_split(dataset, [train_size, val_size, test_size])


df = generate_data()

train_input, val_input, _ = load_data(df)


def train_discriminator(config):
    net = AnnDiscriminator(model_paramaters ={
    'in_features' : grid_size*3,   
    'hidden_layers' : config["hidden_layers"],
    'out_features' : 1,
    })

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_input, val_input, test_input = load_data(df)

    train_loader = DataLoader(
        train_input, batch_size=int(config["batch_size"]), shuffle=True, #num_workers=8
    )
    val_loader = DataLoader(
        val_input, batch_size=len(val_input), shuffle=True, #num_workers=8
    )
    test_loader = DataLoader(
        test_input, batch_size=len(test_input), shuffle=False,# num_workers=2
    )

    for epoch in range(start_epoch, 2000):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        i=0    
        for X, y in train_loader:

            X, y = X.to(device), y.to(device)
            # forward pass and loss

            y_pred = net.forward(X)
            loss = criterion(y_pred, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0
            i+=1    
        
        # Validation loss
        val_loss = 0.0

        net.eval()
        X, y = next(iter(val_loader))  # extract X,y from test dataloader
        X, y = X.to(device), y.to(device)
        
        with torch.no_grad():
            y_eval = net.forward(X)
            
        
        val_accuracy = float(100 * torch.mean(((y_eval>0) == y).float()).item())    # deactivates autograd
        val_loss = criterion(y_eval, y).cpu().numpy().astype(float).tolist()

        
        

        test_accuracy = test_model(test_loader, net, device=device)
        # Save checkpoint
 
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss , "accuracy": val_accuracy , "test_accuracy":test_accuracy},
                checkpoint=checkpoint,
            )

    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    # Define HyperOpt-based search algorithm
    hyperopt_search = HyperOptSearch(
        metric="loss",
        mode="min"  # Minimize loss
    )

    # Pass the search algorithm to Ray Tune
    trainable_with_cpu_gpu = tune.with_resources(train_discriminator, {"cpu": 2, "gpu": gpus_per_trial})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        param_space={
        "hidden_layers": tune.choice([  2**np.random.randint(1, 10, 
                                        size= np.random.randint(1,4,dtype = int), dtype=int) 
                                        for _ in range(9)
                                    ]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32,64,128]),
    },
        tune_config=tune.TuneConfig(search_alg=hyperopt_search, 
                                    num_samples=num_samples,
                                    scheduler=scheduler)
    )

    result_grid = tuner.fit()

    results_df = result_grid.get_dataframe()
    results_df.to_pickle("experiment_results.pkl")

    # # Get data from all trials
    # df = result.results_df

    # print(df.head())

    # best_trial = result.get_best_trial("loss", "min", "last")
    # print(f"Best trial config: {best_trial.config}")
    # print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # best_trained_model = AnnDiscriminator(model_paramaters ={
    # 'in_features' : grid_size*3,
    # 'hidden_layers' : best_trial.config["hidden_layers"],
    # 'out_features' : 1,
    # })

    # device = "cpu"

    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     data_path = Path(checkpoint_dir) / "data.pkl"
    #     with open(data_path, "rb") as fp:
    #         best_checkpoint_data = pickle.load(fp)

    #     best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    #     test_acc = test_accuracy(best_trained_model, device)
    #     print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=5, max_num_epochs=5, gpus_per_trial=1)
