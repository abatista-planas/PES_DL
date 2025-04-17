import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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

    # plot_confusion_matrix(y_true_np, y_pred_np, title='Confusion Matrix')
    failure_index = np.nonzero(y_true_np != y_pred_np)[0].tolist()

    df_samples.reset_index(drop=True, inplace=True)
    df_failure = df_samples[df_samples.index.isin(failure_index)]

    value_counts = df_failure["deformation_type"].value_counts()
    print("\n")
    print("Failure Distribution by Deformation Type:")
    print(value_counts)
    print("\n")
