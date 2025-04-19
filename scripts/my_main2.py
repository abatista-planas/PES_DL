import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.discriminator import AnnDiscriminator  # type: ignore
from pes_1D.training import test_model, train_model  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore


number_of_pts = 128
n_samples = 10000
test_split = 0.4
gpu = True
properties_list = ["energy","derivative","inverse_derivative"]  # List of properties to use for training
properties_format = "array"  # Format [concatenated array or table] of properties to use for training
deformation_list = np.array(["outliers","oscillation"])  # Types of deformation to generate

X_train, y_train, X_test, y_test, df_samples = generate_discriminator_training_set(
    n_samples, number_of_pts,
    properties_list,
    deformation_list,
    properties_format,
    test_split, gpu, 
    generator_seed=[37, 43]
)

data = {'X_train': X_train, 
        'y_train': y_train, 
        'X_test': X_test, 
        'y_test': y_test}
