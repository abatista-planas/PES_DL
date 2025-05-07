import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.data_generator import (
    generate_bad_samples,
    generate_discriminator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore

n_samples = [5000]
grid_size = 250
batch_size = 25
test_split = 0.5
pes_name_list = ["lennard_jones"]
deformation_list = np.array(
    [
        "outliers",
        "oscillation",
        "pulse_random_fn",
        "piecewise_random",
        "random_functions",
    ]
)  # Types of deformation to generate

probability_deformation = np.array(
    [0.25, 0.35, 0.3, 0.05, 0.05]
)  # Probability of deformation to generate

properties_list = [
    "energy",
    "derivative",
    "inverse_derivative",
]  # List of properties to use for training

properties_format = (
    "table_1D"  # Format [concatenated array or table] of properties to use for training
)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, df_samples, _ = generate_discriminator_training_set(
    n_samples=n_samples,
    batch_size=batch_size,
    grid_size=grid_size,
    pes_name_list=pes_name_list,
    properties_list=properties_list,
    deformation_list=deformation_list,
    probability_deformation=probability_deformation,
    properties_format=properties_format,
    test_split=test_split,
    device=device,
)

model_parameters = {
    "in_channels": len(properties_list),
    "grid_size": grid_size,
    "hidden_channels": [16, 32],
    "kernel_size": [3, 3],
    "pool_size": [2, 2],
}


model = CnnDiscriminator(model_parameters).cuda()
model.summary()

num_epochs = 2000
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
print("Training the model...")
time_start = time.process_time()
trainAcc, losses = model.train_model(
    train_loader,
    criterion,
    optimizer,
    num_epochs,
)
time_end = time.process_time()
print(f"Training time: {time_end - time_start:.2f} seconds")


print("Accuracy Training: ", trainAcc[-1])

acc_test = model.test_model(test_loader, device="cuda")[0]
print("Accuracy Test: ", acc_test)


# All trues
df_real_pes = generate_true_pes_samples(["reudenberg", "morse"], [1, 999], grid_size)

# All falses
df_random_fns = generate_bad_samples(
    pes_name_list=["morse"],
    n_samples=[1000],
    deformation_list=np.array(["outliers", "oscillation"]),
    size=grid_size,
)

df_pes_non_included = pd.concat([df_real_pes, df_random_fns], axis=0, ignore_index=True)

(
    test_loader_non_included,
    _,
    df_non_included,
    _,
) = generate_discriminator_training_set_from_df(
    df_pes_non_included,
    batch_size=2000,
    properties_list=properties_list,
    test_split=0.00,
)

accuracy, y_pred, y_true = model.test_model(test_loader_non_included, device="cuda")


get_model_failure_info(df_non_included, y_pred, y_true)


print("Accuracy Non_Included Test: ", accuracy)

print("CNN Number of parameters: ", model.count_parameters())
