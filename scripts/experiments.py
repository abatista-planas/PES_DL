"""Experiments with the model for hyperparameters tuning and evaluation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import pickle # type: ignore
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
    def hyperparameter_tuning(
        hyperparameter, variation_list, df_samples, model_class,model_paramaters = {}, n_repeat=100, verbose=False, save=False
    ) -> pd.DataFrame:
        """Runs the model with different hyperparameters values and return the accuracies trend


        Args:
            hyperparameter (str): hyperparameter to be varied
            variation_list (list): list of values to be used for training
            df_samples (pd.DataFrame): DataFrame containing the samples to be used for training
            model (nn.Module): model to be used for training
            n_repeat (int, optional): number of times to repeat the training. Defaults to 100.
            verbose (bool, optional): If True, prints the progress of the training. Defaults to False.
            save (bool, optional): If True, saves the accuracy list to a file and their plot. Defaults to False.
        Returns:
            accuracy_list (pd.DataFrame): DataFrame containing the accuracies for each hyperparameter value
        """

        best_accuracy_train = [0.0] * len(variation_list)
        mean_accuracy_train = [0.0] * len(variation_list)
        worst_accuracy_train = [0.0] * len(variation_list)

        best_accuracy_test = [0.0] * len(variation_list)
        mean_accuracy_test = [0.0] * len(variation_list)
        worst_accuracy_test = [0.0] * len(variation_list)

        test_split = 0.5
        pes_name_list = ["lennard_jones"]
        deformation_list = np.array(["outliers", "oscillation"])  # Types of deformation to generate
        properties_list = [
            "energy",
            "derivative",
            "inverse_derivative",
        ]  # List of properties to use for training
        properties_format = "array"  # Format [concatenated array or table] of properties to use for training

        n_samples = len(df_samples)  
        batch_size = 50
        num_epochs = 2000
        criterion = nn.BCEWithLogitsLoss()
        gpu = True


        if hyperparameter not in ["grid_size"]:
            
            (
                train_loader, 
                test_loader, 
                _,
                train_data,
            ) = generate_discriminator_training_set_from_df(
                df_samples,
                batch_size,
                properties_list,
                properties_format,
                test_split,
                gpu=gpu,
            )

            
        model = model_class(model_paramaters)
        model = model.to("cuda" if gpu else "cpu") 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for i, val in enumerate(variation_list):
            
            if verbose:
                print(hyperparameter,": ", val)

            best_accuracy_tr = 0.0
            worst_accuracy_tr = 101.0
            mean_accuracy_tr = 0.0
            
            best_accuracy_ts = 0.0
            worst_accuracy_ts = 101.0
            mean_accuracy_ts = 0.0
                        
            
            # Set the hyperparameter value
            if hyperparameter == "lr":
                
                optimizer = torch.optim.Adam(model.parameters(), lr=val)

            elif hyperparameter == "training_size":
                train_dataset = Subset(train_data, np.arange(int(val)).tolist())
                train_loader = DataLoader(train_dataset,batch_size = 50)
                
            elif hyperparameter == "grid_size":
                
                X_train, X_train, X_test, y_test, model, optimizer = Experiment.configure_model_(
                   hyperparameter, val, model_class, model_paramaters,n_samples,pes_name_list, properties_list,deformation_list, properties_format, test_split, gpu 
                )
            else:
                raise ValueError("Unknown hyperparameter: {}".format(hyperparameter))
            
            for _ in range(n_repeat):
                model.reset()
                
                # Train the model
                trainAcc,_ = train_model(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    num_epochs,
                ) 
                
                mean_accuracy_tr += trainAcc[-1]
                best_accuracy_tr = max(trainAcc[-1], best_accuracy_tr)
                worst_accuracy_tr = min(trainAcc[-1], worst_accuracy_tr)
                
                
                # Test the model
                testAcc = test_model(
                    test_loader,
                    model,
                )

                mean_accuracy_ts += testAcc[-1]
                best_accuracy_ts = max(testAcc[-1], best_accuracy_ts)
                worst_accuracy_ts = min(testAcc[-1], worst_accuracy_ts)

            # Compute the mean accuracy
             
            mean_accuracy_tr /= n_repeat
            mean_accuracy_train[i] = mean_accuracy_tr
            worst_accuracy_train[i] = worst_accuracy_tr
            best_accuracy_train[i] = best_accuracy_tr
                
            mean_accuracy_ts /= n_repeat
            mean_accuracy_test[i] = mean_accuracy_ts
            worst_accuracy_test[i] = worst_accuracy_ts
            best_accuracy_test[i] = best_accuracy_ts
            
            if verbose:
                print("Accuracy ({}) for {}, best training: {} / test: {}".format(hyperparameter,val, best_accuracy_tr,best_accuracy_ts))

        df = pd.DataFrame(
            {
                "variation_list": variation_list,
                "best_accuracy_train": best_accuracy_train,
                "worst_accuracy_train": worst_accuracy_train,
                "mean_accuracy_train": mean_accuracy_train,
                "best_accuracy_test": best_accuracy_test,
                "worst_accuracy_test": worst_accuracy_test,
                "mean_accuracy_test": mean_accuracy_test,
            }
        )
        if save:
            Experiment.save_df_(df, hyperparameter)
            
        return df
    
    @staticmethod
    # utils functions
    def save_df_(df, hyperparameter   ):
        df.to_pickle("accuracy_"+hyperparameter+".pkl")
        
        fig,ax = plt.subplots(1,figsize=(12,6))
        for column in df.drop("variation_list", axis=1).columns:
            accuracy_list = df[column].to_numpy()
            variation_list = df["variation_list"].to_numpy()
            ax.plot(variation_list,accuracy_list,'o-',label=column,markerfacecolor='w',markersize=9)

        ax.set_ylabel('accuracy')
        ax.set_xlabel(hyperparameter)
        ax.set_title('Accuracy vs '+ hyperparameter)
        if hyperparameter == "lr":
            ax.set_xscale('log')
        ax.legend()
        pickle.dump(fig, open('Accuracy_'+ hyperparameter+'.fig.pickle', 'wb'))
    
    @staticmethod
    def configure_model_(hyperparameter, value, model_class, model_paramaters,n_samples, pes_name_list, properties_list,deformation_list, properties_format, test_split, gpu):
                    # Set the hyperparameter value
            
            if hyperparameter == "grid_size":
                
                X_train, X_train, X_test, y_test, _ = generate_discriminator_training_set(
                    n_samples = n_samples,
                    size = value,
                    pes_name_list = pes_name_list,
                    properties_list = properties_list,
                    deformation_list = deformation_list,
                    properties_format=properties_format,
                    test_split = test_split,
                    gpu = gpu,
                )
                
                print(value)
                print(X_train.shape)
                model_paramaters["in_features"] = (
                    X_train.shape[1] if properties_format == "array" else value
                )
                
                model = model_class(model_paramaters)
                model = model.to("cuda" if gpu else "cpu")    
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                return X_train, X_train, X_test, y_test, model, optimizer
            else:
                raise ValueError("Unknown hyperparameter: {}".format(hyperparameter))
                