"""Generates sythetic data and noised data for the Lennard-Jones model"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize
import torch

from pes_1D.utils import NoiseFunctions
from pes_1D.utils import PesModels

def generate_discriminator_training_set(
    n_samples: int, size: int, 
    properties_list: list[str] = ["energy"], 
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]), 
    properties_format: str = "table",
    test_split: float = 0.2, gpu: bool = True, 
    generator_seed: list[int] = [37,43]

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Generates training sets for the Lennard-Jones potential"""

    df_good_sample = generate_lennard_jones_samples(int(n_samples / 2), size, seed=generator_seed[0])
    df_bad_sample = generate_bad_lennard_jones_samples(
        n_samples - int(n_samples / 2), size,deformation_list, seed=generator_seed[1]
    )

    df_all = pd.concat([df_good_sample, df_bad_sample])
    # Shuffle DataFrame in place
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    label_tensor = torch.tensor(df_all["true_pes"]).flatten()

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
    deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscillation"]),
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
            outliers_function,outliers_derivative = NoiseFunctions.outliers(row.pes["energy"], size)
            row.pes["energy"] += outliers_function
            row.pes["derivative"] +=  outliers_derivative
        elif deformed_list[row.name] == "oscillation":
            r0 = np.random.uniform(0.1,1.5)
            A = np.random.uniform(1,3)
            lmbda = np.random.uniform(-2, 2)
            omega = np.random.uniform(-1, 1)
            phi = np.random.uniform(0.0, 2 * np.pi)

            row.deformation_parameters["r0"] = r0
            row.deformation_parameters["A"] = A
            row.deformation_parameters["lmbda"] = lmbda
            row.deformation_parameters["omega"] = omega
            row.deformation_parameters["phi"] = phi

            oscillation_function,oscillation_derivative =  NoiseFunctions.oscillation(
                row.pes["r"], r0, A, lmbda, omega, phi
            )
            row.pes["energy"] =  row.pes["energy"]*(1+oscillation_function)
            row.pes["derivative"] = row.pes["derivative"]*(1+oscillation_function) + row.pes["energy"]*oscillation_derivative
            row.pes["inverse_derivative"] =  1.0/(row.pes["derivative"]+1e-10)
            
            for col in row.pes.columns :
                max_min_diff = row.pes[col].max() - row.pes[col].min()
                row.pes[col] = (row.pes[col]-row.pes[col].min())/(max_min_diff) if max_min_diff >0 else row.pes[col]-row.pes[col].min()
                
            
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

    sigma_array = np.random.uniform(1.2, 10.0, n_samples)
    epsilon_array = np.random.uniform(5, 600, n_samples)
    wall_max_high = np.random.uniform(1000,2000, n_samples) 
    long_range_limit = np.random.uniform(0.05, 0.10, n_samples)
    df_samples = []

    for i in range(n_samples):
        # Generate random parameters
        sigma = sigma_array[i]
        epsilon = epsilon_array[i]
        r_0 = sigma * 2 ** (1 / 6)
        max_high = wall_max_high[i]

        # Calculate the well depth
        well_depth = PesModels.lennard_jones(sigma, epsilon, np.array([r_0]))
        long_range_max = abs(long_range_limit[i] * well_depth)

        # Find Proper boundary: Using bisection (requires a bracketing interval)
        def f_min(r):
            return PesModels.lennard_jones(sigma, epsilon, r) - max_high

        r_min = optimize.bisect(f_min, 0.001, 100)

        def f_max(r):
            return abs(PesModels.lennard_jones(sigma, epsilon, r)) - long_range_max

        r_max = optimize.bisect(f_max, r_min, 100)

        # Generate samples
        df = PesModels.lennard_jones_pes(sigma, epsilon, r_min, r_max, size)
        
        # Normalize dataframe
        for col in df.columns :
            max_min_diff = df[col].max() - df[col].min()
            df[col] = (df[col]-df[col].min())/(max_min_diff) if max_min_diff >0 else df[col]-df[col].min()
                 
            
        df_samples.append(df)


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