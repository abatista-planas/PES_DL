"""Experiments with the model for hyperparameters tuning and evaluation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pes_1D.data_generator import (  # type: ignore
    generate_discriminator_training_set,
    generate_discriminator_training_set_from_df,
)
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

    @staticmethod
    def variational_learning_rate(
        X_train, y_train, X_test, y_test, model, list_of_lr, verbose=True
    ):
        """Runs the model with different learning rates and return the accuracies trend


        Args:
            model_class ( model): model class to be used for training
            model_paramaters (_type_): model parameters to be used for training
            list_of_pts (_type_):   list of learning rates to be used for training
            verbose (bool, optional): If True, prints the progress of the training. Defaults to True.

        Returns:
            accuracy_list (list[float]): list of accuracies for each number of points
        """

        accuracy_list = np.zeros((len(list_of_lr), 1))

        for i, lr in enumerate(list_of_lr):
            print("lr: ", lr)

            best_accuracy = 0
            for _ in range(3):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                print("Accuracy for {} (lr): {}".format(lr, best_accuracy))

        return accuracy_list

    @staticmethod
    def variational_sample_size(df_samples, model, list_of_sz, verbose=True):
        """Runs the model with different learning rates and return the accuracies trend


        Args:
            model_class ( model): model class to be used for training
            model_paramaters (_type_): model parameters to be used for training
            list_of_pts (_type_):   list of learning rates to be used for training
            verbose (bool, optional): If True, prints the progress of the training. Defaults to True.

        Returns:
            accuracy_list (list[float]): list of accuracies for each number of points
        """

        best_accuracy_list = [0] * len(list_of_sz)
        mean_accuracy_list = [0] * len(list_of_sz)
        worst_accuracy_list = [0] * len(list_of_sz)

        test_split = 0.5
        properties_list = [
            "energy",
            "derivative",
            "inverse_derivative",
        ]  # List of properties to use for training
        properties_format = "array"  # Format [concatenated array or table] of properties to use for training

        (
            X_train,
            y_train,
            X_test,
            y_test,
            df_samples,
        ) = generate_discriminator_training_set_from_df(
            df_samples,
            properties_list,
            properties_format,
            test_split,
            gpu=True,
        )

        for i, sz in enumerate(list_of_sz):
            print("sz: ", sz)

            my_X_train = X_train[:sz]
            my_y_train = y_train[:sz]

            best_accuracy = 0
            worst_accuracy = 101
            mean_accuracy = 0
            n = 100
            for _ in range(n):
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                epochs = 2000

                # Train the model
                train_model(
                    my_X_train,
                    my_y_train,
                    model,
                    criterion,
                    optimizer,
                    epochs,
                    verbose=False,
                )

                # Test the model
                _, accuracy = test_model(
                    X_test, y_test, model, criterion, verbose=False
                )
                mean_accuracy += accuracy
                best_accuracy = max(accuracy, best_accuracy)
                worst_accuracy = min(accuracy, worst_accuracy)
            mean_accuracy /= n
            mean_accuracy_list[i] = mean_accuracy
            worst_accuracy_list[i] = worst_accuracy
            best_accuracy_list[i] = best_accuracy
            if verbose:
                print("Accuracy for {} (lr): {}".format(sz, best_accuracy))

        return pd.DataFrame(
            {
                "sz": list_of_sz,
                "best_accuracy": best_accuracy_list,
                "worst_accuracy": worst_accuracy_list,
                "mean_accuracy": mean_accuracy_list,
            }
        )

    # @staticmethod
    # def architectures_trial(
    #     X_train, y_train, X_test, y_test,
    #     modelclass, architectures_list, verbose=True
    # ):
    #     """Runs the model with different learning rates and return the accuracies trend

    #     Args:
    #         model_class ( model): model class to be used for training
    #         model_paramaters (_type_): model parameters to be used for training
    #         list_of_pts (_type_):   list of learning rates to be used for training
    #         verbose (bool, optional): If True, prints the progress of the training. Defaults to True.

    #     Returns:
    #         accuracy_list (list[float]): list of accuracies for each number of points
    #     """

    #     accuracy_list = np.zeros((len(architectures_list), 1))

    #     for i, lr in enumerate(architectures_list):
    #         print("lr: ", lr)

    #         best_accuracy = 0
    #         for _ in range(3):

    #             criterion = nn.CrossEntropyLoss()
    #             optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #             epochs = 1500

    #             # Train the model
    #             train_model(
    #                 X_train, y_train, model, criterion, optimizer, epochs, verbose=False
    #             )

    #             # Test the model
    #             _, accuracy = test_model(
    #                 X_test, y_test, model, criterion, verbose=False
    #             )

    #             best_accuracy = max(accuracy, best_accuracy)

    #         accuracy_list[i] = best_accuracy
    #         if verbose:
    #             print("Accuracy for {} (lr): {}".format(lr, best_accuracy))

    #     return accuracy_list
