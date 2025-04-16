"""Generates sythetic data and noised data for the Lennard-Jones model"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize


class DataGenerator:
    """Generates sythetic data and noised data"""

    def __init__(self):
        """Constructor for DataGenerator"""
        pass

    @staticmethod
    def sample_visualization(df_samples, nrow: int = 2, ncol: int = 2):
        """Visualizes the samples generated by the DataGenerator"""

        fig, axs = plt.subplots(nrow, ncol)
        index_array = np.random.randint(0, len(df_samples), size=nrow * ncol)
        count = 0
        for i in range(nrow):
            for j in range(ncol):
                df_samples["pes"][index_array[count]].plot(
                    ax=axs[i][j], x="r", y="energy"
                )
                count = count + 1
        plt.show()

    @staticmethod
    def generate_bad_lennard_jones_samples(
        n_samples: int,
        size: int,
        deformation_list: npt.NDArray[np.str_] = np.array(["outliers", "oscilation"]),
        seed: int = 33,
    ):
        np.random.seed(seed)

        deforms_functions = {
            "outliers": lambda energy, size, p=0.1: energy
            * (1 + (np.random.random(size) <= p) * np.random.random(size)),
            "oscilation": lambda energy, r, r0, A, lmbda, omega, phi: energy
            * (1 + A * np.exp(-lmbda * (r - r0)) * np.cos(omega * (r - r0) + phi)),
        }

        """Generates a set of good samples from the Lennard-Jones potential with noise"""
        df_bad_sample = DataGenerator.generate_lennard_jones_samples(
            n_samples, size, seed
        )

        # Randomly select deformation types for each sample
        deformed_list = deformation_list[
            np.random.randint(0, deformation_list.size, size=size)
        ].tolist()

        # Apply the deformation functions
        def assign_deformation_type(row):
            row.model_type = "twisted_lennard_jones"
            row.true_pes = False
            row.parameters["deformation_type"] = deformed_list[row.name]
            if deformed_list[row.name] == "outliers":
                row.pes["energy"] = deforms_functions["outliers"](
                    row.pes["energy"], size
                )
            elif deformed_list[row.name] == "oscilation":
                r0 = np.random.uniform(row.pes["r"].min(), row.pes["r"].max(), 1)
                A = np.random.uniform(-5.0, 5.0, 1)
                lmbda = np.random.uniform(-2, 2, 1)
                omega = np.random.uniform(-10.0, 10.0, 1)
                phi = np.random.uniform(0.0, 2 * np.pi, 1)

                print(r0, A, lmbda, omega, phi)

                row.pes["energy"] = deforms_functions["oscilation"](
                    row.pes["energy"], row.pes["r"], r0, A, lmbda, omega, phi
                )

            return row

        df_bad_sample = df_bad_sample.apply(assign_deformation_type, axis=1)

        return df_bad_sample

    @staticmethod
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

        sigma_array = np.random.uniform(1.0, 5.0, n_samples)
        epsilon_array = np.random.uniform(20, 200, n_samples)
        df_samples = []
        for i in range(n_samples):
            # Generate random parameters
            sigma = sigma_array[i]
            epsilon = epsilon_array[i]
            r_0 = sigma * 2 ** (1 / 6)

            # Calculate the well depth
            well_depth = DataGenerator.lennard_jones(sigma, epsilon, np.array([r_0]))

            # Find Proper boundary: Using bisection (requires a bracketing interval)
            def f_min(r):
                return DataGenerator.lennard_jones(sigma, epsilon, r) - 5 * abs(
                    well_depth
                )

            r_min = optimize.bisect(f_min, 0.01, 100)

            def f_max(r):
                return abs(DataGenerator.lennard_jones(sigma, epsilon, r)) - abs(
                    0.05 * well_depth
                )

            r_max = optimize.bisect(f_max, r_min, 100)

            # Generate samples
            df_samples.append(
                DataGenerator.lennard_jones_pes(sigma, epsilon, r_min, r_max, size)
            )

        return pd.DataFrame(
            {
                "model_type": ["lennard_jones"] * n_samples,
                "true_pes": [True] * n_samples,
                "parameters": [
                    {
                        "sigma": sigma_array[i],
                        "epsilon": epsilon_array[i],
                    }
                    for _ in range(n_samples)
                ],
                "pes": df_samples,
            }
        )

    @staticmethod
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
        return pd.DataFrame(
            {"r": r, "energy": DataGenerator.lennard_jones(sigma, epsilon, r)}
        )

    @staticmethod
    def lennard_jones(
        sigma: float, epsilon: float, r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluates the Lennard-Jones potential for given parameters and distance"""

        if np.any(r <= 0):
            raise Exception("Size and range must be positive")

        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
