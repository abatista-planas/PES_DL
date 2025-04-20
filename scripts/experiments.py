"""Experiments with the model for hyperparameters tuning and evaluation."""
import numpy as np
import torch
import torch.nn as nn

from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.training import test_model, train_model  # type: ignore


class Experiment:
    def __init__(self):
        pass

    @staticmethod
    def variational_number_of_pts(
        model_class, model_paramaters, list_of_pts, verbose=True
    ):
        """Runs the model with different number of points and return the accuracies trend


        Args:
            model_class ( model): model class to be used for training
            model_paramaters (_type_): model parameters to be used for training
            list_of_pts (_type_):   list of number of points to be used for training
            verbose (bool, optional): If True, prints the progress of the training. Defaults to True.

        Returns:
            accuracy_list (list[float]): list of accuracies for each number of points
        """
        n_samples = 5000
        test_size = 0.6
        gpu = True
        properties_list = [
            "energy",
            "derivative",
            "inverse_derivative",
        ]  # List of properties to use for training
        properties_format = "array"  # Format [concatenated array or table] of properties to use for training

        accuracy_list = np.zeros((len(list_of_pts), 1))
        for i, number_of_pts in enumerate(list_of_pts):
            print("number of pts", number_of_pts)
            # generate training and test data
            X_train, y_train, X_test, y_test, _ = generate_discriminator_training_set(
                n_samples,
                number_of_pts,
                properties_list,
                properties_format=properties_format,
                test_split=test_size,
            )

            model_paramaters["in_features"] = (
                X_train.shape[1] if properties_format == "array" else number_of_pts
            )

            best_accuracy = 0
            for _ in range(3):
                model = model_class(model_paramaters)
                model = model.to("cuda" if gpu else "cpu")

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                epochs = 1500

                # Train the model
                train_model(
                    X_train, y_train, model, criterion, optimizer, epochs, verbose=False
                )

                # Test the model
                _, accuracy = test_model(
                    X_test, y_test, model, criterion, verbose=False
                )

                best_accuracy = max(accuracy, best_accuracy)

            accuracy_list[i] = best_accuracy
            if verbose:
                print("Accuracy for {} points: {}".format(number_of_pts, best_accuracy))

        return accuracy_list
