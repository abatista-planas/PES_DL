"""Generates sythetic data and noised data for the Lennard-Jones model"""

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import scipy.optimize as optimize  # type: ignore
import torch
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

from pes_1D.utils import Normalizers  # type: ignore
from pes_1D.utils import NoiseFunctions, PesModels


def generate_discriminator_training_set(
    n_samples: list[int],
    batch_size: int,
    grid_size: int,
    pes_name_list: list[str] = ["lennard_jones"],
    properties_list: list[str] = ["energy"],
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]),
    probability_deformation: npt.NDArray[np.float64] = np.array([0.5, 0.5]),
    properties_format: str = "table_1D",
    test_split: float = 0.2,
    device="cpu",
    generator_seed: list[int] = [37, 43],
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, TensorDataset]:
    """Generates training sets for the Lennard-Jones potential"""

    df_good_sample = generate_true_pes_samples(
        pes_name_list, n_samples, grid_size, generator_seed[0]
    )

    df_bad_sample = generate_bad_samples(
        pes_name_list,
        n_samples,
        grid_size,
        deformation_list,
        probability_deformation,
        seed=generator_seed[1],
    )

    df_all = pd.concat([df_good_sample, df_bad_sample])

    return generate_discriminator_training_set_from_df(
        df_all, batch_size, properties_list, properties_format, test_split, device
    )


def split_data(
    df_all: pd.DataFrame,
    properties_list: list[str] = ["energy"],
    properties_format: str = "table_1D",
    test_split: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Generates training sets for the Lennard-Jones potential"""

    # Shuffle DataFrame in place
    shuffled_df = df_all.sample(frac=1).reset_index(drop=True)

    label_tensor = torch.tensor(shuffled_df["true_pes"].values, dtype=torch.float)
    label_tensor = label_tensor[:, None]

    def input_format(pes):
        """Define input format"""
        if properties_format == "array":
            return pes[properties_list].values.flatten()
        else:
            return pes[properties_list].values

    # Convert continuous variables to a tensor
    input_arrays = [input_format(pes) for pes in shuffled_df["pes"].values]
    input_stack = np.stack(input_arrays)
    input_tensor = torch.tensor(input_stack, dtype=torch.float)

    if properties_format == "table_1D":
        input_tensor = input_tensor.permute(0, 2, 1)
    elif properties_format == "table_2D":
        input_tensor = input_tensor.unsqueeze(1)

    # use scikitlearn to split the data
    if test_split <= 0.0 or test_split >= 1.0:
        train_input = input_tensor
        test_input = input_tensor
        train_labels = label_tensor
        test_labels = label_tensor
    else:
        train_input, test_input, train_labels, test_labels = train_test_split(
            input_tensor, label_tensor, test_size=test_split
        )

    return (
        train_input,
        test_input,
        train_labels,
        test_labels,
        shuffled_df,
    )


def generate_discriminator_training_set_from_df(
    df_all: pd.DataFrame,
    batch_size: int = 100,
    properties_list: list[str] = ["energy"],
    properties_format: str = "table_1D",
    test_split: float = 0.2,
    device="cpu",
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, TensorDataset]:
    """Generates training sets for the Lennard-Jones potential"""

    train_input, test_input, train_labels, test_labels, shuffled_df = split_data(
        df_all,
        properties_list,
        properties_format,
        test_split,
    )

    train_input = train_input.to(device)
    train_labels = train_labels.to(device)
    test_input = test_input.to(device)
    test_labels = test_labels.to(device)

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    train_data = TensorDataset(train_input, train_labels)
    test_data = TensorDataset(test_input, test_labels)

    # finally, translate into dataloader objects

    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return (
        train_loader,
        test_loader,
        shuffled_df,
        train_data,
    )


def generate_generator_training_set_from_df(
    df_all: pd.DataFrame,
    batch_size: int = 100,
    up_scale: int = 4,
    properties_list: list[str] = ["energy"],
    properties_format: str = "table_1D",
    test_split: float = 0.2,
    device="cpu",
):  # -> Tuple[DataLoader, DataLoader, pd.DataFrame, TensorDataset]:
    """Generates training sets for the Lennard-Jones potential"""

    train_input_hr, test_input_hr, _, _, shuffled_df = split_data(
        df_all,
        properties_list,
        properties_format,
        test_split,
    )

    train_input_hr = torch.unsqueeze(train_input_hr, dim=1)
    test_input_hr = torch.unsqueeze(test_input_hr, dim=1)

    tensor_len = train_input_hr.size(-1)

    indices = torch.linspace(
        0, tensor_len - 1, int(tensor_len / up_scale), dtype=torch.long
    )

    if properties_format == "array":

        train_input_lr = train_input_hr[:, :, indices].to(device)
        test_input_lr = test_input_hr[:, :, indices].to(device)
        train_input_hr = train_input_hr.to(device)
        test_input_hr = test_input_hr.to(device)

    if properties_format == "table_1D":
        train_input_lr = train_input_hr[:, :, :, indices].to(device)
        test_input_lr = test_input_hr[:, :, :, indices].to(device)
        train_input_hr = train_input_hr.to(device)
        test_input_hr = test_input_hr.to(device)

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    train_data = TensorDataset(train_input_lr, train_input_hr)
    test_data = TensorDataset(test_input_lr, test_input_hr)

    # # finally, translate into dataloader objects

    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return (train_loader, test_loader, shuffled_df, train_input_lr, train_input_hr)


def generate_bad_samples(
    pes_name_list: list[str],
    n_samples: list[int],
    size: int,
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]),
    probability_deformation: npt.NDArray[np.float64] = np.array([0.5, 0.5]),
    seed: int = 33,
) -> pd.DataFrame:
    np.random.seed(seed)

    """Generates a set of good samples from the Lennard-Jones potential with noise"""
    df_bad_sample = generate_true_pes_samples(pes_name_list, n_samples, size, seed)
    pb = probability_deformation[: deformation_list.size]
    pb = pb / np.sum(pb)
    # Randomly select deformation types for each sample
    deformed_list = deformation_list[
        # np.random.randint(0, deformation_list.size, size=n_samples)
        # Generate a random list of 10 integers
        np.random.choice(
            np.arange(deformation_list.size), size=n_samples, p=pb, replace=True
        )
    ].tolist()

    # Apply the deformation functions
    def assign_deformation_type(row):
        row.model_type = "pes*_"
        row.true_pes = 0
        row.modified_pes = 1
        row.deformation_type = deformed_list[row.name]

        if deformed_list[row.name] == "outliers":
            outliers_function, outliers_derivative = NoiseFunctions.outliers(
                row.pes["energy"], size
            )
            row.pes["energy"] += outliers_function
            row.pes["derivative"] += outliers_derivative

            row.pes["inverse_derivative"] = 1.0 / (row.pes["derivative"] + 1e-10)

        elif deformed_list[row.name] == "oscillation":
            r0 = np.random.uniform(3.0, 8.0)
            A = np.random.uniform(0.25, 0.5)
            n = int(np.random.uniform(1, 25))

            row.deformation_parameters["r0"] = r0
            row.deformation_parameters["A"] = A
            row.deformation_parameters["n"] = n

            oscillation_function, oscillation_derivative = NoiseFunctions.oscillation(
                row.pes["r"], r0, A, n
            )
            row.pes["energy"] = row.pes["energy"] * (1 + oscillation_function)
            row.pes["derivative"] = (
                row.pes["derivative"] * (1 + oscillation_function)
                + row.pes["energy"] * oscillation_derivative
            )
            row.pes["inverse_derivative"] = 1.0 / (row.pes["derivative"] + 1e-10)

        elif deformed_list[row.name] == "pulse_random_fn":

            mu = np.random.uniform(-3, 3)
            sigma = np.random.uniform(0.2, 2)

            row.deformation_parameters["mu"] = mu
            row.deformation_parameters["sigma"] = sigma

            pulse_function, pulse_derivative, fn_label = NoiseFunctions.pulse_random_fn(
                row.pes["r"], mu, sigma
            )
            row.model_type = fn_label + "_"
            row.deformation_parameters["random_fn"] = fn_label
            row.pes["energy"] = row.pes["energy"] * (1 + pulse_function)
            row.pes["derivative"] = (
                row.pes["derivative"] * (1 + pulse_function)
                + row.pes["energy"] * pulse_derivative
            )
            row.pes["inverse_derivative"] = 1.0 / (row.pes["derivative"] + 1e-10)

        elif deformed_list[row.name] == "piecewise_random":

            r0 = np.random.uniform(3.0, 10.0)

            row.deformation_parameters["r0"] = r0

            pw_function, pw_derivative, fn_label = NoiseFunctions.piecewise_random(
                row.pes["r"], r0
            )
            row.model_type = fn_label[0] + " - " + fn_label[1] + "_"
            row.deformation_parameters["random_fn"] = fn_label[0] + " - " + fn_label[1]
            row.pes["energy"] = pw_function
            row.pes["derivative"] = pw_derivative
            row.pes["inverse_derivative"] = 1.0 / (pw_derivative + 1e-10)

        elif deformed_list[row.name] == "random_functions":
            # Generate random function
            # r = np.linspace(1, 10,len(row.pes["r"]), dtype=np.float64)
            fn, dfn, fn_label, _ = NoiseFunctions.random_function(row.pes["r"])

            row.model_type = fn_label + "_"
            # row.pes["r"] = r
            row.pes["energy"] = fn
            row.pes["derivative"] = dfn
            row.pes["inverse_derivative"] = 1.0 / (dfn + 1e-10)

        # Normalize dataframe
        for col in row.pes.columns:
            if col == "r":
                continue
                # row.pes[col] = Normalizers.min_max_normalize(row.pes[col])
            else:
                row.pes[col] = Normalizers.normalize(row.pes[col])

        return row

    df_bad_sample = df_bad_sample.apply(assign_deformation_type, axis=1)

    return df_bad_sample


def generate_true_pes_samples(
    pes_name_list: list[str],
    n_samples: list[int],
    size: int,
    seed: int = 33,
) -> pd.DataFrame:
    """Generates a set of samples"""

    total_samples = sum(n_samples)

    if len(pes_name_list) == 0:
        return pd.DataFrame(
            {
                "model_type": [""] * total_samples,
                "true_pes": [0] * total_samples,
                "parameters": [{}] * total_samples,
                "pes": [pd.DataFrame()] * total_samples,
                "modified_pes": [0] * total_samples,
                "deformation_type": [""] * total_samples,
                "deformation_parameters": [{}] * total_samples,
            }
        )

    df_pes = pd.DataFrame()

    for i, pes_name in enumerate(pes_name_list):
        if pes_name == "lennard_jones":
            parameters_array = np.zeros((n_samples[i], 2))
            parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples[i])  # sigma
            parameters_array[:, 1] = np.random.uniform(
                5, 50000, n_samples[i]
            )  # epsilon
            df = generate_analytical_pes_samples(
                pes_name, parameters_array, size, seed=seed
            )

        elif pes_name == "morse":
            parameters_array = np.zeros((n_samples[i], 3))
            parameters_array[:, 0] = np.random.uniform(5, 100000, n_samples[i])  # D_e
            parameters_array[:, 1] = np.random.uniform(2.5, 10, n_samples[i])  # alpha
            parameters_array[:, 2] = np.random.uniform(1.2, 10.0, n_samples[i])  # r_0
            df = generate_analytical_pes_samples(
                pes_name, parameters_array, size, seed=seed
            )

        elif pes_name == "reudenberg":
            number_of_samples = min(n_samples[i], len(PesModels.reudenberg_parameters))
            keys = PesModels.reudenberg_parameters.keys()
            reudenberg_keys = list(keys)[:number_of_samples]

            parameters_array = np.zeros(
                (
                    number_of_samples,
                    len(PesModels.reudenberg_parameters[reudenberg_keys[0]]),
                )
            )
            for i, key in enumerate(reudenberg_keys):
                parameters_array[i, :] = PesModels.reudenberg_parameters[key]

            df = generate_analytical_pes_samples(
                pes_name, parameters_array, size, seed=seed
            )
            df["model_type"] = ["reudenberg_" + key for key in reudenberg_keys]

        df_pes = pd.concat([df_pes, df], axis=0, ignore_index=True)

    return df_pes


def generate_pes(pes_name, size, parameters):

    # Get random parameters
    zero = getattr(PesModels, pes_name)(parameters, np.array(100))

    def pes(r):
        return getattr(PesModels, pes_name)(parameters, r) - zero

    r_trial = np.linspace(0.1, 50.0, 500)
    energy_array = pes(r_trial)

    min_index = np.argmin(energy_array)
    r_0 = r_trial[min_index]
    approx_well_depth = energy_array[min_index]

    max_high = np.max(
        [
            np.random.uniform(1000.0, 2000.0),
            np.random.uniform(1.0, 3.0) * np.abs(approx_well_depth),
        ]
    )
    long_range_max = abs(np.random.uniform(0.01, 0.40) * approx_well_depth)

    # Find Proper boundary: Using bisection (requires a bracketing interval)
    def f_min(r):
        return pes(r) - max_high

    r_min = optimize.bisect(f_min, 0.001, 100.00)

    def f_max(r):
        return abs(pes(r)) - long_range_max

    r_max = optimize.bisect(f_max, r_0, 100.00)

    # Generate samples
    df = PesModels.analytical_pes(pes_name, parameters, r_min, r_max, size)

    # Normalize dataframe
    for col in df.columns:
        if col == "r":
            continue
        else:
            df[col] = Normalizers.normalize(df[col])

    return df


def generate_analytical_pes_samples(
    pes_name: str = "lennard_jones",
    parameters_array: npt.NDArray[np.float64] = np.array([]),
    size: int = 128,
    seed: int = 33,
) -> pd.DataFrame:
    """Generates a set of samples from the Lennard-Jones potential

    Args:
        n_samples (int): number of samples to generate
        size (int): number of points in each sample
        seed (int, optional): seed for random number generator. Defaults to 33.

    Returns:
        pd.DataFrame: DataFrame containing the generated samples
    """

    np.random.seed(seed)
    n_samples = parameters_array.shape[0]

    task = partial(generate_pes, pes_name, size)

    if n_samples > 1:
        with Pool(processes=cpu_count()) as p:
            df_samples = p.map(task, parameters_array)
    else:

        df_samples = [generate_pes(pes_name, size, parameters_array[0])]

    return pd.DataFrame(
        {
            "model_type": [pes_name] * n_samples,
            "true_pes": [1] * n_samples,
            "parameters": [
                {"parameters": parameters_array[i]} for i in range(n_samples)
            ],
            "pes": df_samples,
            "modified_pes": [0] * n_samples,
            "deformation_type": [""] * n_samples,
            "deformation_parameters": [{}] * n_samples,
        }
    )
