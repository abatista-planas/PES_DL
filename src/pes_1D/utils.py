import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

from pes_1D.visualization import plot_confusion_matrix, sample_visualization


class NoiseFunctions:
    def __init__(self):
        pass

    @staticmethod
    def outliers(energy, size, p=0.1):
        return energy * (
            1 + (np.random.random(size) <= p) * np.random.uniform(0.25, 1.0, size)
        )

    @staticmethod
    def oscilation(
        energy: npt.NDArray[np.float32],
        r: npt.NDArray[np.float32],
        r0: float,
        A: float,
        lmbda: float,
        omega: float,
        phi: float,
    ):
        return energy * (
            1 + A * np.exp(-lmbda * (r - r0)) * np.cos(omega * (r - r0) + phi)
        )

    @staticmethod
    def noise(energy: npt.NDArray[np.float32], size: int, noise_level: float = 0.1):
        """Adds Gaussian noise to the energy values."""
        return energy * (1 + np.random.normal(0, noise_level, size=size))


class PesModels:
    def __init__(self):
        pass

    @staticmethod
    def lennard_jones(
        sigma: float, epsilon: float, r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluates the Lennard-Jones potential for given parameters and distance"""

        if np.any(r <= 0):
            raise Exception("Size and range must be positive")

        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    @staticmethod
    def lennard_jones_derivative(
        sigma: float, epsilon: float, r: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluates the derivative of the Lennard-Jones potential for given parameters and distance"""

        if np.any(r <= 0):
            raise Exception("Size and range must be positive")

        return (
            -(24 * epsilon / r) * (2 * (sigma / r) ** 12 - (sigma / r) ** 6)
        ).astype(np.float64)

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
            {"r": r, "energy": PesModels.lennard_jones(sigma, epsilon, r)}
        )

    @staticmethod
    def lennard_jones_pes_derivatives(
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
            {
                "r": r,
                "energy": PesModels.lennard_jones_derivative(sigma, epsilon, r)
                + np.random.uniform(-5.0, 5.0),
            }
        )


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
