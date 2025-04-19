# import libraries
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.discriminator import NaiveDiscriminator  # type: ignore
from pes_1D.training import test_model, train_model  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore
from pes_1D.visualization import sample_visualization  # type: ignore

in_features = 128
n_samples = 2000
test_split = 0.5
gpu = False

X_train, y_train, X_test, y_test, df_samples = generate_discriminator_training_set(
    n_samples, in_features, test_split, gpu, generator_seed=[37, 43]
)


# # parameters
# nPerClust = 100
# blur = 1

# A = [  1, 1 ]
# B = [  5, 1 ]

# # generate data
# a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
# b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# # true labels
# labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# # concatanate into a matrix
# data_np = np.hstack((a,b)).T

# # convert to a pytorch tensor
# data = torch.tensor(data_np).float()
# labels = torch.tensor(labels_np).float()

# # show the data
# fig = plt.figure(figsize=(5,5))
# plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
# plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
# plt.title('The qwerties!')
# plt.xlabel('qwerty dimension 1')
# plt.ylabel('qwerty dimension 2')
# plt.show()

def createANNmodel(learningRate):

    # model architecture
    ANNclassify = nn.Sequential(
        nn.Linear(in_features, 512),  # input layer
        nn.ReLU(),  # activation unit
        nn.Linear(512, 256),  # hidden layer
        nn.ReLU(),  # activation unit
        nn.Linear(256, 64),  #  hidden layer
        nn.ReLU(),  # activation unit
        nn.Linear(64, 1),  # output unit
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(ANNclassify.parameters(),lr=learningRate)

    # model output
    return ANNclassify,criterion ,optimizer


# a function that trains the model

# a fixed parameter
numepochs = 300

def trainTheModel(ANNmodel):

    losses: list[float] = []
    epochs = 300

    for epoch in range(epochs):
        y_pred = ANNclassify(X_train)

        loss = criterion(y_pred, y_train)
        losses.append(loss.item())

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses  

#     # forward pass
#     yHat = ANNmodel(X_train)

#     # compute loss
#     loss = lossfun(yHat,y_train.float())
#     losses[epochi] = loss

#     # backprop
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


#   # final forward pass
#   predictions = ANNmodel(X_train)

#   # compute the predictions and report accuracy
#   # NOTE: shouldn't it be predictions>.5??
#   totalacc = 100*torch.mean(((predictions>0) == y_train).float())


# create everything
ANNclassify, criterion, optimizer = createANNmodel(0.001)

# run it
losses,predictions,totalacc = trainTheModel(ANNclassify)

# report accuracy
print('Final accuracy: %g%%' %totalacc)

print(losses)
# show the losses
plt.plot(losses.detach(),'.',markerfacecolor='w',linewidth=.1)
plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.show()

# # the set of learning rates to test
# learningrates = np.linspace(.001,.1,40)

# # initialize results output
# accByLR = []
# allLosses = np.zeros((len(learningrates),numepochs))


# # loop through learning rates
# for i,lr in enumerate(learningrates):

#   # create and run the model
#   ANNclassify,lossfun,optimizer = createANNmodel(lr)
#   losses,predictions,totalacc = trainTheModel(ANNclassify)

#   # store the results
#   accByLR.append(totalacc)
#   allLosses[i,:] = losses.detach()

# # plot the results
# fig,ax = plt.subplots(1,2,figsize=(12,4))

# ax[0].plot(learningrates,accByLR,'s-')
# ax[0].set_xlabel('Learning rate')
# ax[0].set_ylabel('Accuracy')
# ax[0].set_title('Accuracy by learning rate')

# ax[1].plot(allLosses.T)
# ax[1].set_title('Losses by learning rate')
# ax[1].set_xlabel('Epoch number')
# ax[1].set_ylabel('Loss')
# plt.show()


# # proportion of runs where the model had at least 70% accuracy
# sum(torch.tensor(accByLR)>70)/len(accByLR)
