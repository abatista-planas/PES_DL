"""Generates sythetic data and noised data for the Lennard-Jones model"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize
import torch
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

from pes_1D.utils import NoiseFunctions, PesModels


def generate_discriminator_training_set(
    n_samples: int,
    batch_size: int,
    grid_size: int,
    pes_name_list: list[str] = ["lennard_jones"],
    properties_list: list[str] = ["energy"],
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]),
    properties_format: str = "table",
    test_split: float = 0.2,
    gpu: bool = True,
    generator_seed: list[int] = [37, 43],
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, TensorDataset]:
    """Generates training sets for the Lennard-Jones potential"""
    df_good_sample = generate_true_pes_samples(
        pes_name_list, int(n_samples / 2), grid_size, generator_seed[0]
    )

    df_bad_sample = generate_bad_samples(
        pes_name_list,
        n_samples - int(n_samples / 2),
        grid_size,
        deformation_list,
        seed=generator_seed[1],
    )

    df_all = pd.concat([df_good_sample, df_bad_sample])

    return generate_discriminator_training_set_from_df(
        df_all, batch_size, properties_list, properties_format, test_split, gpu
    )


def generate_discriminator_training_set_from_df(
    df_all: pd.DataFrame,
    batch_size: int = 100,
    properties_list: list[str] = ["energy"],
    properties_format: str = "table",
    test_split: float = 0.2,
    gpu: bool = True,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, TensorDataset]:
    """Generates training sets for the Lennard-Jones potential"""

    # Shuffle DataFrame in place
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    label_tensor = torch.tensor(df_all["true_pes"].values, dtype=torch.float)
    label_tensor = label_tensor[:, None]

    def input_format(pes):
        """Define input format"""
        if properties_format == "array":
            return pes[properties_list].values.flatten()
        else:
            return pes[properties_list].values

    # Convert continuous variables to a tensor
    input_arrays = [input_format(pes) for pes in df_all["pes"].values]
    input_stack = np.stack(input_arrays)
    input_tensor = torch.tensor(input_stack, dtype=torch.float)

    # use scikitlearn to split the data
    train_input, test_input, train_labels, test_labels = train_test_split(
        input_tensor, label_tensor, test_size=test_split
    )

    train_input = train_input.to("cuda" if gpu else "cpu")
    test_input = test_input.to("cuda" if gpu else "cpu")
    train_labels = train_labels.to("cuda" if gpu else "cpu")
    test_labels = test_labels.to("cuda" if gpu else "cpu")

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    train_data = TensorDataset(train_input, train_labels)
    test_data = TensorDataset(test_input, test_labels)

    # finally, translate into dataloader objects

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return (
        train_loader,
        test_loader,
        df_all,
        train_data,
    )


def generate_bad_samples(
    pes_name_list: list[str],
    n_samples: int,
    size: int,
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]),
    seed: int = 33,
) -> pd.DataFrame:
    np.random.seed(seed)

    """Generates a set of good samples from the Lennard-Jones potential with noise"""
    df_bad_sample = generate_true_pes_samples(pes_name_list, n_samples, size, seed)

    # Randomly select deformation types for each sample
    deformed_list = deformation_list[
        np.random.randint(0, deformation_list.size, size=n_samples)
    ].tolist()

    # Apply the deformation functions
    def assign_deformation_type(row):
        row.model_type = "twisted_pes"
        row.true_pes = 0
        row.modified_pes = 1
        row.deformation_type = deformed_list[row.name]

        if deformed_list[row.name] == "outliers":
            outliers_function, outliers_derivative = NoiseFunctions.outliers(
                row.pes["energy"], size
            )
            row.pes["energy"] += outliers_function
            row.pes["derivative"] += outliers_derivative

        elif deformed_list[row.name] == "oscillation":
            r0 = np.random.uniform(0.1, 1.5)
            A = np.random.uniform(1, 3)
            lmbda = np.random.uniform(-2, 2)
            omega = np.random.uniform(-1, 1)
            phi = np.random.uniform(0.0, 2 * np.pi)

            row.deformation_parameters["r0"] = r0
            row.deformation_parameters["A"] = A
            row.deformation_parameters["lmbda"] = lmbda
            row.deformation_parameters["omega"] = omega
            row.deformation_parameters["phi"] = phi

            oscillation_function, oscillation_derivative = NoiseFunctions.oscillation(
                row.pes["r"], r0, A, lmbda, omega, phi
            )
            row.pes["energy"] = row.pes["energy"] * (1 + oscillation_function)
            row.pes["derivative"] = (
                row.pes["derivative"] * (1 + oscillation_function)
                + row.pes["energy"] * oscillation_derivative
            )
            row.pes["inverse_derivative"] = 1.0 / (row.pes["derivative"] + 1e-10)

            for col in row.pes.columns:
                max_min_diff = row.pes[col].max() - row.pes[col].min()
                row.pes[col] = (
                    (row.pes[col] - row.pes[col].min()) / (max_min_diff)
                    if max_min_diff > 0
                    else row.pes[col] - row.pes[col].min()
                )
        elif deformed_list[row.name] == "random_functions":
            # Generate random function
            r = np.linspace(0.1, 10, size, dtype=np.float64)
            fn_label, fn, dfn = NoiseFunctions.random_function(r)

            row.model_type = fn_label
            row.pes["r"] = r
            row.pes["energy"] = fn
            row.pes["derivative"] = dfn
            row.pes["inverse_derivative"] = 1.0 / (dfn + 1e-10)

            # Normalize dataframe
            for col in row.pes.columns:
                max_min_diff = row.pes[col].max() - row.pes[col].min()
                row.pes[col] = (
                    (row.pes[col] - row.pes[col].min()) / (max_min_diff)
                    if max_min_diff > 0
                    else row.pes[col] - row.pes[col].min()
                )

        return row

    df_bad_sample = df_bad_sample.apply(assign_deformation_type, axis=1)

    return df_bad_sample


def generate_true_pes_samples(
    pes_name_list: list[str], n_samples: int, size: int, seed: int = 33
) -> pd.DataFrame:
    """Generates a set of samples"""

    if len(pes_name_list) == 0:
        return pd.DataFrame(
            {
                "model_type": [""] * n_samples,
                "true_pes": [0] * n_samples,
                "parameters": [{}] * n_samples,
                "pes": [pd.DataFrame()] * n_samples,
                "modified_pes": [0] * n_samples,
                "deformation_type": [""] * n_samples,
                "deformation_parameters": [{}] * n_samples,
            }
        )

    sample_count = n_samples
    df_pes = pd.DataFrame()

    for pes_name in pes_name_list:
        if sample_count < 1:
            break
        if pes_name == "lennard_jones":
            parameters_array = np.zeros((n_samples, 2))
            parameters_array[:, 0] = np.random.uniform(1.2, 10.0, n_samples)
            parameters_array[:, 1] = np.random.uniform(5, 2000, n_samples)
            df = generate_analytical_pes_samples(pes_name, parameters_array, size, seed)
            sample_count = 0

        elif pes_name == "reudenberg":
            number_of_samples = min(sample_count, len(PesModels.reudenberg_parameters))
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

            sample_count = sample_count - number_of_samples
            df = generate_analytical_pes_samples(pes_name, parameters_array, size, seed)
            df["model_type"] = ["reudenberg_" + key for key in reudenberg_keys]
        else:
            sample_count = 0

        df_pes = pd.concat([df_pes, df], axis=0, ignore_index=True)

    return df_pes


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
    wall_max_high = np.random.uniform(1000, 2000, n_samples)
    long_range_limit = np.random.uniform(0.01, 0.40, n_samples)
    df_samples = []

    for i in range(n_samples):
        # Get random parameters
        parameters = parameters_array[i]
        zero = getattr(PesModels, pes_name)(parameters, np.array(100))

        def pes(r):
            return getattr(PesModels, pes_name)(parameters, r) - zero

        r_trial = np.linspace(0.1, 50.0, 500)
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

        # Generate samples
        df = PesModels.analytical_pes(pes_name, parameters, r_min, r_max, size)

        # Normalize dataframe
        for col in df.columns:
            max_min_diff = df[col].max() - df[col].min()
            df[col] = (
                (df[col] - df[col].min()) / (max_min_diff)
                if max_min_diff > 0
                else df[col] - df[col].min()
            )

        df_samples.append(df)

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
