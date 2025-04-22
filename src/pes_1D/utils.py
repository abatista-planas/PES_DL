from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from scipy.differentiate import derivative

from pes_1D.visualization import plot_confusion_matrix, sample_visualization


class AuxiliarFunctions:
    def __init__(self):
        pass

    @staticmethod
    def min_max_normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)


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
        lmbda: float,
        omega: float,
        phi: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        oscillation_function = (
            A * np.exp(-lmbda * (r - r0)) * np.cos(omega * (r - r0) + phi)
        )
        oscillation_derivative = (
            A
            * np.exp(-lmbda * (r - r0))
            * (
                -lmbda * np.cos(omega * (r - r0) + phi)
                - omega * np.sin(omega * (r - r0) + phi)
            )
        )
        return oscillation_function, oscillation_derivative

    @staticmethod
    def noise(A: float, size: int, noise_level: float = 0.2) -> npt.NDArray[np.float64]:
        """Adds Gaussian noise to the energy values."""
        return A * np.random.normal(0.0, noise_level, size=size)

    @staticmethod
    def random_function(
        r: npt.NDArray[np.float64],
    ) -> Tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generates a random function"""
        # Generate random coefficients for the functions
        coeff = np.random.uniform(-1.0, 1.0, 10)

        def norm(x):
            return AuxiliarFunctions.min_max_normalize(x)

        funct_dict = {
            "constant": lambda x: np.ones_like(x) * coeff[0],
            "normal": lambda x: coeff[0] * np.exp(-coeff[1] * x**2),
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
            "np.exp": lambda x: np.exp(coeff[0] * x + coeff[1]),
            "sqrt": lambda x: np.sqrt(np.abs(coeff[0] * x + coeff[1])),
            "abs": lambda x: np.abs(coeff[0] * x + coeff[1]),
            "sinh": lambda x: np.sinh(coeff[0] * x + coeff[1]),
            "cosh": lambda x: np.cosh(coeff[0] * x + coeff[1]),
            "arcsin": lambda x: np.arcsin(norm(coeff[0] * x + coeff[1])),
            "arccos": lambda x: np.arccos(norm(coeff[0] * x + coeff[1])),
            "arctan": lambda x: np.arctan(norm(coeff[0] * x + coeff[1])),
            "log": lambda x: np.log(np.abs(coeff[0] * x + coeff[1])),
            "sigmoid": lambda x: 1 / (1 + np.exp(-coeff[0] * x + coeff[1])),
            "tanh": lambda x: np.tanh(coeff[0] * x + coeff[1]),
            "relu": lambda x: np.maximum(0, coeff[0] * x + coeff[1]),
        }

        func_label = np.random.choice(list(funct_dict.keys()))
        f = funct_dict[func_label]
        deriv = derivative(f, r)

        return func_label, f(r), deriv.df


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


def get_model_failure_info(
    df_samples: pd.DataFrame,
    x: torch.Tensor,
    y_true: torch.Tensor,
    model: nn.Module,
):
    """Returns the failure information of the model."""
    # This function is a placeholder for the actual implementation
    # It should return the failure information of the model
    with torch.no_grad():
        y_pred = torch.argmax(model.forward(x), dim=1)
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

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
