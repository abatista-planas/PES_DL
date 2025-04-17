"""Generates sythetic data and noised data for the Lennard-Jones model"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize
import torch

from pes_1D.utils import NoiseFunctions


def generate_discriminator_training_set(
    n_samples: int, size: int, test_split: float = 0.2, gpu: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Generates training sets for the Lennard-Jones potential"""

    df_good_sample = generate_lennard_jones_samples(int(n_samples / 2), size, seed=37)
    df_bad_sample = generate_bad_lennard_jones_samples(
        n_samples - int(n_samples / 2), size, seed=43
    )

    df_all = pd.concat([df_good_sample, df_bad_sample])
    # Shuffle DataFrame in place
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    label_tensor = torch.tensor(df_all["true_pes"]).flatten()

    # Convert continuous variables to a tensor
    input_arrays = [pes["energy"].values for pes in df_all["pes"].values]
    input_stack = np.stack(input_arrays)
    input_tensor = torch.tensor(input_stack, dtype=torch.float)

    test_size = int(n_samples * test_split)  # test_split% for testing

    input_train = input_tensor[: n_samples - test_size]
    input_test = input_tensor[n_samples - test_size : n_samples]
    label_train = label_tensor[: n_samples - test_size]
    label_test = label_tensor[n_samples - test_size : n_samples]

    return (
        input_train.cuda() if gpu else input_train,
        label_train.cuda() if gpu else label_train,
        input_test.cuda() if gpu else input_test,
        label_test.cuda() if gpu else label_test,
        df_all,
    )


def generate_bad_lennard_jones_samples(
    n_samples: int,
    size: int,
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscilation"]),
    seed: int = 33,
):
    np.random.seed(seed)

    """Generates a set of good samples from the Lennard-Jones potential with noise"""
    df_bad_sample = generate_lennard_jones_samples(n_samples, size, seed)

    # Randomly select deformation types for each sample
    deformed_list = deformation_list[
        np.random.randint(0, deformation_list.size, size=n_samples)
    ].tolist()

    # Apply the deformation functions
    def assign_deformation_type(row):
        row.model_type = "twisted_lennard_jones"
        row.true_pes = 0
        row.modified_pes = 1
        row.deformation_type = deformed_list[row.name]

        if deformed_list[row.name] == "outliers":
            row.pes["energy"] = NoiseFunctions.outliers(row.pes["energy"], size)
        elif deformed_list[row.name] == "oscilation":
            r0 = np.random.uniform(row.pes["r"].min(), row.pes["r"].max(), 1)
            A = np.random.uniform(-5.0, 5.0, 1)
            lmbda = np.random.uniform(-2, 2, 1)
            omega = np.random.uniform(-10.0, 10.0, 1)
            phi = np.random.uniform(0.0, 2 * np.pi, 1)

            row.deformation_parameters["r0"] = r0
            row.deformation_parameters["A"] = A
            row.deformation_parameters["lmbda"] = lmbda
            row.deformation_parameters["omega"] = omega
            row.deformation_parameters["phi"] = phi

            row.pes["energy"] = NoiseFunctions.oscilation(
                row.pes["energy"], row.pes["r"], r0, A, lmbda, omega, phi
            )

        return row

    df_bad_sample = df_bad_sample.apply(assign_deformation_type, axis=1)

    return df_bad_sample


def generate_lennard_jones_samples(
    n_samples: int, size: int, seed: int = 33
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

    sigma_array = np.random.uniform(1.0, 7.0, n_samples)
    epsilon_array = np.random.uniform(20, 300, n_samples)
    df_samples = []
    for i in range(n_samples):
        # Generate random parameters
        sigma = sigma_array[i]
        epsilon = epsilon_array[i]
        r_0 = sigma * 2 ** (1 / 6)

        # Calculate the well depth
        well_depth = lennard_jones(sigma, epsilon, np.array([r_0]))

        # Find Proper boundary: Using bisection (requires a bracketing interval)
        def f_min(r):
            return lennard_jones(sigma, epsilon, r) - 5 * abs(well_depth)

        r_min = optimize.bisect(f_min, 0.01, 100)

        def f_max(r):
            return abs(lennard_jones(sigma, epsilon, r)) - abs(0.05 * well_depth)

        r_max = optimize.bisect(f_max, r_min, 100)

        # Generate samples
        df_samples.append(lennard_jones_pes(sigma, epsilon, r_min, r_max, size))

    return pd.DataFrame(
        {
            "model_type": ["lennard_jones"] * n_samples,
            "true_pes": [1] * n_samples,
            "parameters": [
                {
                    "sigma": sigma_array[i],
                    "epsilon": epsilon_array[i],
                }
                for _ in range(n_samples)
            ],
            "pes": df_samples,
            "modified_pes": [0] * n_samples,
            "deformation_type": [""] * n_samples,
            "deformation_parameters": [{}] * n_samples,
        }
    )


def lennard_jones_pes(
    sigma: float, epsilon: float, R_min: float, R_max: float, size: int
) -> pd.DataFrame:
    """Generates a set of samples from the Lennard-Jones potential for given parameters

    Args:
        sigma (float): Sigma parameter of the Lennard-Jones potential
        epsilon (float): Epsilon parameter of the Lennard-Jones potential
        R_min (float):  _minimum distance for the potential
                        0.01 <= R_min < R_max
        R_max (float): _maximum distance for the potential
                        R_min < R_max <= 100
        size (int): number of points in each sample

    Returns:
        pd.DataFrame: DataFrame containing the generated samples
    """
    if size <= 0 or R_min <= 0 or R_max <= 0 or R_min >= R_max:
        raise Exception("Size and range must be positive")

    r = np.linspace(R_min, R_max, size, dtype=np.float64)
    return pd.DataFrame({"r": r, "energy": lennard_jones(sigma, epsilon, r)})


def lennard_jones(
    sigma: float, epsilon: float, r: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluates the Lennard-Jones potential for given parameters and distance"""

    if np.any(r <= 0):
        raise Exception("Size and range must be positive")

    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def lennard_jones_derivative(
    sigma: float, epsilon: float, r: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluates the Lennard-Jones potential for given parameters and distance"""

    if np.any(r <= 0):
        raise Exception("Size and range must be positive")

    return 4 * epsilon * (1 / r) * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
