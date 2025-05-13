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
from pes_1D.visualization import sample_visualization  # type: ignore

n_samples = [1000]
grid_size = 150
batch_size = 50
test_split = 0.9
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

sample_visualization(df_samples)


model_parameters = {
    "in_channels": len(properties_list),
    "grid_size": grid_size,
    "hidden_channels": [16, 32],
    "kernel_size": [3, 3],
    "pool_size": [2, 2],
}


model = CnnDiscriminator(model_parameters).cuda()
# model.summary()

# global parameter
num_epochs = 300
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

# plt.plot(range(2, num_epochs), losses[2:])
# plt.show()

test_results = model.test_model(test_loader)
testAcc = test_results[0]  # Extract the first tensor as test accuracy

print(f"Training Accuracy:{trainAcc[-1]}")
print(f"Test Accuracy:{testAcc}")


# Data not included in the training set

# All trues
df_real_pes = generate_true_pes_samples(["reudenberg", "morse"], [1, 999], grid_size)

# All falses
df_random_fns = generate_bad_samples(
    pes_name_list=["morse"],
    n_samples=[1000],
    deformation_list=np.array(["outliers", "oscillation", "pulse_random_fn"]),
    probability_deformation=[0.3, 0.3, 0.4],
    size=grid_size,
)


df_pes_non_included = pd.concat([df_real_pes, df_random_fns], axis=0, ignore_index=True)

(
    test_loader_non_included,
    tensor_dataset_non_included,
    df_pes_non_included,
    _,
) = generate_discriminator_training_set_from_df(
    df_pes_non_included,
    batch_size=len(df_pes_non_included),
    properties_list=properties_list,
    properties_format=properties_format,
    test_split=0.00,
    device=device,
)

accuracy, y_pred, y_true = model.test_model(test_loader_non_included)


print(f"NonIncluded Accuracy :{accuracy}")

get_model_failure_info(df_pes_non_included, y_pred, y_true)
