import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.data_generator import generate_discriminator_training_set_from_df
from pes_1D.data_generator import generate_bad_samples
from pes_1D.data_generator import generate_analytical_pes_samples
from pes_1D.data_generator import generate_true_pes_samples
from pes_1D.discriminator import AnnDiscriminator  # type: ignore
from pes_1D.training import test_model, train_model  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore
from experiments import Experiment  # type: ignore
import pandas as pd

# # variational_number_of_pts
# model_paramaters ={
#     'in_features' : 128,
#     'hidden_layers' : [512,128,32],
#     'out_features' : 2,
#     }
# list_of_points = np.arange(30, 200, 2).tolist()
# accuracy_list = Experiment.variational_number_of_pts(AnnDiscriminator,model_paramaters,list_of_points)

# # show accuracy as a function of model depth
# fig,ax = plt.subplots(1,figsize=(12,6))

# ax.plot(list_of_points,accuracy_list,'o-',markerfacecolor='w',markersize=9)
# ax.set_ylabel('accuracy')
# ax.set_xlabel('Number of hidden units')
# ax.set_title('Accuracy')
# plt.show()


# # variational_learning_rate
# number_of_pts = 128
# n_samples = 10000
# test_split = 0.8
# gpu = True
# properties_list = ["energy","derivative","inverse_derivative"]  # List of properties to use for training
# properties_format = "array"  # Format [concatenated array or table] of properties to use for training
# deformation_list = np.array(["outliers","oscillation"])  # Types of deformation to generate


# df_samples = pd.read_pickle("scripts/data/outliers_oscillation")

# X_train, y_train, X_test, y_test, df_samples = generate_discriminator_training_set_from_df(
#     df_samples,
#     properties_list,
#     properties_format,
#     test_split,
#     gpu,
#     )

# model_paramaters ={
#     'in_features' : X_train.shape[1] if properties_format == "array" else number_of_pts,   
#     'hidden_layers' : [512,128,32],
#     'out_features' : 2,
#     }

# model =  AnnDiscriminator(model_paramaters)
# model = model.to("cuda" if gpu else "cpu") 


# init_val = 0.00001
# list_of_lr = [init_val*2**n for n in range(17)]

# accuracy_list = Experiment.variational_learning_rate(
#         X_train, y_train, X_test, y_test,
#         model,list_of_lr, verbose=True
#     )

# # show accuracy as a function of model depth
# fig,ax = plt.subplots(1,figsize=(12,6))

# ax.plot(list_of_lr,accuracy_list,'o-',markerfacecolor='w',markersize=9)
# ax.set_ylabel('accuracy')
# ax.set_xlabel('learning rate')
# ax.set_xscale('log')
# ax.set_title('Accuracy vs lr')
# plt.show()



#variational_sample size


df_samples = pd.read_pickle("scripts/data/outliers_oscillation")

model_paramaters ={
    'in_features' : 128*3,#X_train.shape[1] if properties_format == "array" else number_of_pts,   
    'hidden_layers' : [512,256,32],
    'out_features' : 2,
    }

model =  AnnDiscriminator(model_paramaters)
model = model.to("cuda") 


list_of_sz = np.arange(100, 5000, 100).tolist()

df_accuracy = Experiment.variational_sample_size(
        df_samples,
        model,list_of_sz, verbose=True
    )

# show accuracy as a function of model depth
fig,ax = plt.subplots(1,figsize=(12,6))
for column in df_accuracy.columns:
    accuracy_list = df_accuracy[column].to_numpy()
    ax.plot(list_of_sz,accuracy_list,'o-',label=column,markerfacecolor='w',markersize=9)

ax.set_ylabel('accuracy')
ax.set_xlabel('training size')
ax.set_title('Accuracy vs training size')
ax.legend()
plt.show()




# # variational architectures
# model_paramaters ={
#     'in_features' : 128,
#     'hidden_layers' : [512,128,32],
#     'out_features' : 2,
#     }
# list_of_points = np.random.randint(2, 512, 1000).tolist()
# accuracy_list = Experiment.variational_number_of_pts(AnnDiscriminator,model_paramaters,list_of_points)

# # show accuracy as a function of model depth
# fig,ax = plt.subplots(1,figsize=(12,6))

# ax.plot(list_of_points,accuracy_list,'o-',markerfacecolor='w',markersize=9)
# ax.set_ylabel('accuracy')
# ax.set_xlabel('Number of hidden units')
# ax.set_title('Accuracy')
# plt.show()



n_samples = 100
# df = generate_reudenberg_samples(1,128)


# sigma_array = np.random.uniform(1.2, 10.0, n_samples)
# epsilon_array = np.random.uniform(5, 600, n_samples)



# number_of_pts = 100
# n_samples = 1000
# test_split = 0.6
# gpu = True
# pes_name_list = ["lennard_jones"]
# properties_list = ["energy","derivative","inverse_derivative"]  # List of properties to use for training
# properties_format = "array"  # Format [concatenated array or table] of properties to use for training
# deformation_list = np.array(["outliers","oscillation"])  # Types of deformation to generate

# X_train, y_train, X_test, y_test, df_samples = generate_discriminator_training_set(
#     n_samples, number_of_pts,
#     pes_name_list,
#     properties_list,
#     deformation_list,
#     properties_format,
#     test_split, gpu, 
#     generator_seed=[37, 43]
# )

# model_paramaters ={
#     'in_features' : X_train.shape[1] if properties_format == "array" else number_of_pts,   
#     'hidden_layers' : [512,256,32],
#     'out_features' : 2,
#     }

# model =  AnnDiscriminator(model_paramaters)
# model = model.to("cuda" if gpu else "cpu") 


# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# epochs = 2000

# # Train the model
# print("Training the model...")
# losses = train_model(X_train, y_train, model, criterion, optimizer, epochs,verbose=False)

# print(f"Training Losses:{losses[-1]}")
# plt.plot(range(2,epochs), losses[2:])
# plt.show()

# # Test the model
# test_loss, accuracy = test_model(X_test, y_test, model, criterion)

# df_train = df_samples[df_samples.index < n_samples - int(n_samples * test_split)]
# df_test = df_samples[df_samples.index >= n_samples - int(n_samples * test_split)]


# get_model_failure_info(df_test, X_test, y_test, model)

# # Data not included in the training set

# # All trues
# df_real_pes = generate_true_pes_samples(["reudenberg"], 1, number_of_pts)

# # All falses
# df_random_fns = generate_bad_samples(
#     pes_name_list=[],
#     n_samples=1000,
#     size=number_of_pts,
#     deformation_list=np.array(["random_functions"]),
# )

# df_pes_outsiders =pd.concat([df_real_pes,df_random_fns] ,axis=0, ignore_index=True)

# X_outsiders, y_outsiders,_,_,_ = generate_discriminator_training_set_from_df(
#     df_pes_outsiders,
#     properties_list ,
#     properties_format,
#     test_split=0.0,
#     gpu=gpu,
# )



# print("Testing the model on random functions...")
# # Test the model
# test_loss, accuracy = test_model(X_outsiders, y_outsiders, model, criterion)

# get_model_failure_info(df_pes_outsiders, X_outsiders, y_outsiders, model)

# df_real_pes = generate_true_pes_samples(["reudenberg"], 1, number_of_pts)
# X_real_pes, y_real_pes,_,_,df_real_pes= generate_discriminator_training_set_from_df(
#     df_real_pes,
#     properties_list ,
#     properties_format,
#     test_split=0.0,
#     gpu=gpu,
# )

# print("Testing the real model on random functions...")
# test_loss, accuracy = test_model(X_real_pes, y_real_pes, model, criterion)

# print("y eval, ytest",torch.argmax(model.forward(X_real_pes), dim=1),y_real_pes)


# df_real_pes.iloc[0].pes.plot(
#             x="r",
#             y="energy",
#         )
# plt.show()



# from pes_1D.utils import PesModels


# Oxigen_Oxigen = [  42030.046,
#          -2388.564169,
#          18086.977116,
#          -71760.197585,
#          154738.09175,
#          -215074.85646,
#          214799.54567,
#          -148395.4285,
#          73310.781453]

# r = np.linspace(0.9, 10, 100)
# pes_ = PesModels.reudenberg_pes(Oxigen_Oxigen, r)
# plt.plot(r, pes_)
# plt.show()