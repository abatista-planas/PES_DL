import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from pes_1D.data_generator import generate_discriminator_training_set  # type: ignore
from pes_1D.discriminator import AnnDiscriminator  # type: ignore
from pes_1D.training import test_model, train_model  # type: ignore
from pes_1D.utils import get_model_failure_info  # type: ignore


number_of_pts = 128
n_samples = 10000
test_split = 0.4
gpu = True
properties_list = ["energy","derivative","inverse_derivative"]  # List of properties to use for training
properties_format = "array"  # Format [concatenated array or table] of properties to use for training
deformation_list = np.array(["outliers","oscillation"])  # Types of deformation to generate

X_train, y_train, X_test, y_test, df_samples = generate_discriminator_training_set(
    n_samples, number_of_pts,
    properties_list,
    deformation_list,
    properties_format,
    test_split, gpu, 
    generator_seed=[37, 43]
)

data = {'X_train': X_train, 
        'y_train': y_train, 
        'X_test': X_test, 
        'y_test': y_test}

torch.save(data, 'multiple_tensors.pt')
in_features = X_train.shape[1] if properties_format == "array" else number_of_pts

model =  AnnDiscriminator(in_features, [512,128,32], 2)
model = model.to("cuda" if gpu else "cpu") 


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 1000

# Train the model
print("Training the model...")
losses = train_model(X_train, y_train, model, criterion, optimizer, epochs,verbose=False)

print(f"Training Losses:{losses[-1]}")
plt.plot(range(2,epochs), losses[2:])
plt.show()

# Test the model
test_loss, accuracy = test_model(X_test, y_test, model, criterion)

df_train = df_samples[df_samples.index < n_samples - int(n_samples * test_split)]
df_test = df_samples[df_samples.index >= n_samples - int(n_samples * test_split)]


get_model_failure_info(df_test, X_test, y_test, model)

