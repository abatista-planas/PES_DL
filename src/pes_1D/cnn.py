import numpy as np
import pandas as pd
import torch
from torch import nn

from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.data_generator import (
    generate_bad_samples,
    generate_discriminator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore
from pes_1D.visualization import sample_visualization  # type: ignore

n_samples = [5000]
grid_size = 150
batch_size = 50
test_split = 0.8
pes_name_list = ["lennard_jones"]
deformation_list = np.array(
    ["outliers", "oscillation"]
)  # Types of deformation to generate
properties_list = [
    "energy",
    "derivative",
    "inverse_derivative",
]  # List of properties to use for training

properties_format = (
    "table_1D"  # Format [concatenated array or table] of properties to use for training
)
gpu = True

train_loader, test_loader, df_samples, _ = generate_discriminator_training_set(
    n_samples=n_samples,
    batch_size=batch_size,
    grid_size=grid_size,
    pes_name_list=pes_name_list,
    properties_list=properties_list,
    deformation_list=deformation_list,
    properties_format=properties_format,
    test_split=test_split,
    gpu=gpu,
)

sample_visualization(df_samples)


model_paramaters = {
    "in_channels": 3,
}


model = CnnDiscriminator(model_paramaters).cuda()
model.summary()

num_epochs = 2000
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
print("Training the model...")
trainAcc, losses = model.train_model(
    train_loader,
    criterion,
    optimizer,
    num_epochs,
)

print("Accuracy Test: ", trainAcc[-1])

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
)[0]

accuracy, y_pred, y_true = model.test_model(test_loader_non_included, device="cuda")


get_model_failure_info(df_pes_non_included, y_pred, y_true)


print("Accuracy Non_Included Test: ", accuracy)

print("CNN Number of parameters: ", model.count_parameters())
