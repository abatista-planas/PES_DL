from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.differentiate import derivative  # type: ignore

from pes_1D.visualization import plot_confusion_matrix, sample_visualization


class NoiseFunctions:
    def __init__(self):
        pass

    @staticmethod
    def outliers(
        A: float, size: int, p: float = 0.05
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        outliers_array = (np.random.random(size) <= p) * np.random.uniform(
            0.25, 1.0, size
        )

        outliers_function = A * outliers_array

        outliers_derivative = np.random.uniform(10**3, 10**8, size) * outliers_array

        return outliers_function, outliers_derivative

    @staticmethod
    def oscillation(
        r: npt.NDArray[np.float64],
        r0: float,
        A: float,
        n: int,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        def f(r_):
            rs = r_ * (2 * np.pi / 20.0) - r0

            fr = np.sin((n * 1.0 + 0.5) * rs) / np.sin(rs / 2.0)
            max_fr = np.max(np.abs(fr))

            return A * fr / max_fr

        oscillation_function = f(r)
        oscillation_derivative = derivative(f, r).df

        return oscillation_function, oscillation_derivative

    @staticmethod
    def pulse_random_fn(
        r: npt.NDArray[np.float64],
        mu: float,
        sigma: int,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
        """Generates a random oscillation function"""
        _, _, label, f_rand = NoiseFunctions.random_function(r)

        def f(r_):
            sg = np.max([0.5, (2 * sigma**2)])
            rs = (r_ - np.mean(r_) + mu) ** 2 / sg

            gaussian = np.exp(rs) / np.sqrt(np.pi * sg)
            max_gs = np.max(np.abs(gaussian))
            nm_gaussian = gaussian / max_gs
            fr = f_rand(r_)
            max_fr = np.max(np.abs(fr))

            eps = 1e-10  # Define a small epsilon value
            if max_fr >= eps:
                nm_fr = fr / np.max(np.abs(fr))
            else:
                nm_fr = np.ones_like(fr)

            return nm_fr * nm_gaussian

        oscillation_function = f(r)
        oscillation_derivative = derivative(f, r).df

        return oscillation_function, oscillation_derivative, label

    @staticmethod
    def piecewise_random(
        r: npt.NDArray[np.float64], r0: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
        labels = ["", ""]
        _, _, labels[0], f1 = NoiseFunctions.random_function(r)
        f1_r0 = f1(r0)
        _, _, labels[1], f2 = NoiseFunctions.random_function(r)
        f2_r0 = f2(r0)

        def f(r_):
            less_cond = 1.0 * (r_ <= r0)
            greater_cond = 1.0 * (r_ > r0)

            return less_cond * f1(r_) + greater_cond * (f2(r_) - f2_r0 + f1_r0)

        return f(r), derivative(f, r).df, labels

    @staticmethod
    def noise(
        A: float, size: int, noise_level: float = 0.2
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Adds Gaussian noise to the energy values."""

        def f():
            return A * np.random.normal(0.0, noise_level, size=size)

        noise_function = f()
        noise_derivative = derivative(f).df
        return noise_function, noise_derivative

    @staticmethod
    def random_function(
        r: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str, Callable]:
        """Generates a random function"""
        # Generate random coefficients for the functions
        coeff = np.random.uniform(0.5, 2.0, 10)

        funct_dict = {
            "constant": lambda x: np.ones_like(x) * coeff[0],
            "normal": lambda x: np.exp(-((x - np.mean(x)) ** 2) / (2 * coeff[0] ** 2))
            / np.sqrt(2 * np.pi * coeff[0] ** 2),
            "linear": lambda x: coeff[0] * x + coeff[1],
            "quadratic": lambda x: coeff[0] * x**2 + coeff[1] * x + coeff[2],
            "cubic": lambda x: coeff[0] * x**3
            + coeff[1] * x**2
            + coeff[2] * x
            + coeff[3],
            "quartic": lambda x: coeff[0] * x**4
            + coeff[1] * x**3
            + coeff[2] * x**2
            + coeff[3] * x
            + coeff[4],
            "cosine": lambda x: np.cos(coeff[0] * x + coeff[1]),
            "sine": lambda x: np.sin(coeff[0] * x + coeff[1]),
            "tan": lambda x: np.tan(coeff[0] * x + coeff[1]),
            "exp": lambda x: np.exp(coeff[0] * (x - np.mean(x)) + coeff[1]),
            "sigmoid": lambda x: 1 / (1 + coeff[0] * np.exp(x - np.mean(x))),
            "tanh": lambda x: coeff[0] * np.tanh(x - np.mean(x)) + coeff[1],
            "sqrt": lambda x: np.sqrt(np.abs(coeff[0] * x + coeff[1])),
            "abs": lambda x: np.abs(coeff[0] * x + coeff[1]),
            "sinh": lambda x: np.sinh(coeff[0] * x + coeff[1]),
            "cosh": lambda x: np.cosh(coeff[0] * x + coeff[1]),
            "log": lambda x: np.log(np.abs(coeff[0] * x + coeff[1])),
            "relu": lambda x: np.maximum(0, np.abs(coeff[0]) * x + np.abs(coeff[1])),
        }

        func_label = np.random.choice(list(funct_dict.keys()))
        f = funct_dict[func_label]

        deriv = derivative(f, r)

        return f(r), deriv.df, func_label, f


class PesModels:
    reudenberg_parameters = {
        "oxigen_oxigen": [
            42030.046,
            -2388.564169,
            18086.977116,
            -71760.197585,
            154738.09175,
            -215074.85646,
            214799.54567,
            -148395.4285,
            73310.781453,
        ]
    }

    def __init__(self):
        pass

    @staticmethod
    def analytical_pes(
        pes_name: str,
        parameters: list[float],
        R_min: float,
        R_max: float,
        size: int,
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

        def pes(r_):
            func = getattr(PesModels, pes_name)
            return func(parameters, r_)

        pes_derivative = derivative(pes, r)
        return pd.DataFrame(
            {
                "r": r,
                "energy": pes(r),
                "derivative": pes_derivative.df,
                "inverse_derivative": 1.0 / (pes_derivative.df + 1e-10),
            }
        )

    @staticmethod
    def lennard_jones(
        parameters: list[float], r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluates the Lennard-Jones potential for given parameters and distance"""

        if np.any(r <= 0):
            raise Exception("Size and range must be positive")

        sigma = parameters[0]
        epsilon = parameters[1]
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    @staticmethod
    def reudenberg(
        coeff: list[float], r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate the Reudenberg potential for given parameters"""

        if np.any(r <= 0):
            raise Exception("Size and range must be positive")

        c1 = 219.47463
        c2 = 0.785
        c3 = 1.307

        pes = np.ones_like(r, dtype=np.float64) * coeff[0]

        for i in range(len(coeff) - 1):
            pes = pes + coeff[i + 1] * np.exp(-(c3**i) * c2 * r**2) * c1

        return pes

    @staticmethod
    def morse(
        parameters: list[float], r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate the Morse potential for given parameters"""
        if np.any(r <= 0):
            raise Exception("Size and range must be positive")
        D_e = parameters[0]
        a = parameters[1]
        r_0 = parameters[2]

        return D_e * (1.0 - np.exp(-a * (r - r_0))) ** 2


class Normalizers:
    """Normalizers for the data."""

    def __init__(self):
        pass

    @staticmethod
    def min_max_normalize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Min-max normalization of the data."""
        min_val = np.min(data)
        max_val = np.max(data)

        if np.abs(max_val - min_val) <= 1e-10:
            return data / max_val
        else:
            return (data - min_val) / (max_val - min_val)

    @staticmethod
    def zero_normalize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Zero-normalization of the data."""

        zero = data[-1]
        zero_data = data - zero
        min_val = np.min(zero_data)

        return zero_data / (min_val)

    @staticmethod
    def normalize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Min-max normalization in the interval from -1 to 1 of the data."""

        new_data = Normalizers.min_max_normalize(data)

        return new_data * 2.0 - 1.0

    @staticmethod
    def gaussian_normalize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Standardization of the data."""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std


def get_model_failure_info(
    df_samples: pd.DataFrame,
    y_eval: torch.Tensor,
    y: torch.Tensor,
):
    """Returns the failure information of the model."""
    # This function is a placeholder for the actual implementation
    # It should return the failure information of the model

    y_true_np = (1 * y).cpu().numpy()
    y_pred_np = (1 * (y_eval > 0)).cpu().numpy()

    plot_confusion_matrix(y_true_np, y_pred_np, title="Confusion Matrix")
    failure_index = np.nonzero(y_true_np != y_pred_np)[0].tolist()

    df_samples.reset_index(drop=True, inplace=True)
    df_failure = df_samples[df_samples.index.isin(failure_index)]

    value_counts = df_failure["deformation_type"].value_counts()
    print("\n")
    print("Failure Distribution by Deformation Type:")
    print(value_counts)
    print("\n")

    sample_visualization(df_failure)
