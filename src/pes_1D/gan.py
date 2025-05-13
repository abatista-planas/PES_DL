# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from scipy.interpolate import CubicSpline  # type: ignore
# from torch import nn

# from pes_1D.data_generator import (
#     generate_generator_training_set_from_df,
#     generate_true_pes_samples,
# )
# from pes_1D.discriminator import CnnDiscriminator  # type: ignore
# from pes_1D.generator import ResNetUpscaler  # type: ignore
# from pes_1D.utils import Normalizers

# # type: ignore

# # # 1) Dataset: low-res and high-res data
# # # --- generate data ---
# num_epochs = 2000
# batch_size = 25
# pes_name_list = ["lennard_jones", "morse"]
# n_samples = [25, 25]
# upscale_factor = 30
# lr_grid_size = 8
# hr_grid_size = lr_grid_size * upscale_factor
# test_split = 0
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df_high_res = generate_true_pes_samples(
#     pes_name_list,
#     n_samples,
#     hr_grid_size,
# )

# train_loader, _, _, _, _ = generate_generator_training_set_from_df(
#     df_high_res,
#     batch_size=batch_size,
#     up_scale=upscale_factor,
#     properties_list=["r", "energy", "derivative", "inverse_derivative"],
#     properties_format="table_1D",
#     test_split=0,
#     device=device,
# )

# df_high_test = generate_true_pes_samples(
#     pes_name_list,
#     n_samples,
#     hr_grid_size,
# )

# test_loader, _, _, _, _ = generate_generator_training_set_from_df(
#     df_high_test,
#     batch_size=len(df_high_test),
#     up_scale=upscale_factor,
#     properties_list=["energy"],
#     properties_format="array",
#     test_split=0,
#     device=device,
# )


# # G = torch.load("/home/albplanas/Desktop/Programming/PES_DL/PES_DL/saved_models/"
# #                     + "CNN_Generator.pth", weights_only=False).to(device)

# G = ResNetUpscaler(upscale_factor=upscale_factor, num_channels=32, num_blocks=3).to(
#     device
# )

# # D = CnnDiscriminator(model_parameters).to(device)
# # 3) Discriminator: judges real vs. fake high-res conditioned on low-res
# D = CnnDiscriminator(
#     {
#         "in_channels": 3,  # energy, derivative, inverse_derivative
#         "grid_size": hr_grid_size,
#         "hidden_channels": [16, 32],
#         "kernel_size": [3, 3],
#         "pool_size": [2, 2],
#     }
# ).to(device)


# # D.pre_train(n_samples=[100],num_epochs=10)


# def df(r_input, v_input):

#     cs = CubicSpline(r_input, v_input, extrapolate=True)
#     deriv = cs.derivative(1)(r_input)
#     # Normalize the derivative
#     return Normalizers.normalize(deriv).reshape(deriv.shape)


# def prepare_discriminator_data(input_tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Prepare discriminator data for training.
#     """
#     # y = input_tensor.clone()

#     output_tensor = torch.zeros((input_tensor.shape[0], 3, input_tensor.shape[-1]))
#     output_tensor[:, 0, :] = input_tensor[:, 1, :]
#     in_tensor = input_tensor.detach().cpu().numpy()

#     for i in range(in_tensor.shape[0]):
#         r_input = in_tensor[i, 0, :]
#         v_input = in_tensor[i, 1, :]

#         deriv = df(r_input, v_input)
#         inverse_deriv = Normalizers.normalize(1 / np.array(deriv))
#         output_tensor[i, 1, :] = torch.tensor(deriv)
#         output_tensor[:, 2, :] = torch.tensor(inverse_deriv)
#     return output_tensor


# class CombinedLoss(nn.Module):
#     def __init__(self, mse_weight=0.5, bce_weight=0.5):
#         super(CombinedLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_weight = mse_weight
#         self.bce_weight = bce_weight
#         self.m = nn.Sigmoid()

#     def forward(self, fake_pred, target):
#         # mse_loss = self.mse_loss(gen_output, true_output)
#         bce_loss = self.bce_loss(self.m(fake_pred), target)
#         print("bce_loss", bce_loss)
#         # total_loss = self.mse_weight * mse_loss + self.bce_weight * bce_loss
#         total_loss = self.bce_weight * bce_loss
#         return total_loss


# myloss = CombinedLoss()


# # 4) Training loop
# def train_gan(G, D, train_loader, epochs=500, device="cpu"):

#     G = G.to(device)
#     D = D.to(device)

#     G.train()
#     D.train()

#     criterion_G = nn.MSELoss()
#     criterion_D = nn.BCEWithLogitsLoss()

#     optim_G = torch.optim.Adam(G.parameters(), lr=1e-3)
#     optim_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

#     for epoch in range(epochs):
#         fakeAcc = []
#         trueAcc = []
#         # loop over training data batches
#         running_loss = 0.0
#         total_samples = 0
#         initial_params = [param.clone() for param in G.parameters()]
#         for pes_low, pes_high in train_loader:

#             pes_low, pes_high = pes_low.to(device), pes_high.to(device)
#             bs = pes_low.size(0)

#             gen_output = G.forward(pes_low[:, 0, 1, :].unsqueeze(1))

#             fake_pes = torch.rand(
#                 (gen_output.shape[0], 3, gen_output.shape[-1]), requires_grad=True
#             )

#             # fake_pes[:,0,:] = pes_high[:,0,0,:]
#             # fake_pes[:,1,:] = pes_high[:,0,0,:]
#             # fake_pes[:,2,:] = pes_high[:,0,0,:]

#             # fake_pes.requires_grad=True
#             fake_pes = fake_pes.to(device)
#             print("Grad: ", fake_pes.grad)
#             # input_tensor[:,1,:] = gen_output[:,0,:].clone().detach()

#             # fake_pes = prepare_discriminator_data(input_tensor).to(device)

#             # if epoch % 10==0:

#             #     true_pes = pes_high[:,0,1:,:].to(device)
#             #     true_pred = D.forward(true_pes)

#             #     D_true_loss = criterion_D(true_pred, torch.ones((bs,1)).to(device))

#             #     fake_pred = D.forward(fake_pes)
#             #     D_fake_loss = criterion_D(fake_pred, torch.zeros((bs,1)).to(device))

#             #     loss_D = (D_true_loss + D_fake_loss ) * 0.5

#             #     #  # Backprop + optimize D
#             #     optim_D.zero_grad()
#             #     loss_D.backward()
#             #     optim_D.step()

#             # -- Generator step --
#             # -- Train Generator --

#             # true_pred = D.forward(pes_high[:,0,1:,:])
#             # trueAcc.append(100 * torch.mean(((true_pred > 0) == torch.ones((bs,1)).to(device)).float()).item())

#             fake_pred = D.forward(fake_pes)
#             # fakeAcc.append(100 * torch.mean(((fake_pred > 0) == torch.zeros((bs,1)).to(device)).float()).item())

#             # true_output = pes_high[:,0,1,:].unsqueeze(1).to(device)
#             # mse_G = criterion_G(gen_output, true_output)
#             # Df_loss = criterion_D(fake_pred, torch.ones((bs,1)).to(device))

#             # print(Df_loss)

#             # lmda_D = 0.0001
#             # lmda_MSE = 100
#             # loss_G = criterion_D(fake_pred, torch.ones((bs,1)).to(device))#Df_loss #lmda_MSE*mse_G + lmda_D*
#             # print(loss_G.item())

#             loss = nn.BCELoss()
#             m = nn.Sigmoid()

#             loss_G = loss(
#                 m(fake_pred),
#                 torch.ones((bs, 1), device=device),
#             )
#             print(m(loss_G))
#             # backprop
#             optim_G.zero_grad()
#             loss_G.backward()
#             optim_G.step()

#             # running_loss += mse_G.item() * batch_size
#             # total_samples += batch_size

#         # Check if parameters have changed
#         trained_params = [param.clone() for param in G.parameters()]

#         for initial, trained in zip(initial_params, trained_params):
#             if not torch.equal(initial, trained):
#                 print("Model has been trained.")
#                 break
#         else:
#             print("Model has not been trained.")
#         # if epoch % 50 == 0:
#         #     msg = (f"Epoch {epoch}/{epochs} | Disc ({float(np.mean(trueAcc)):.2f} ; {float(np.mean(fakeAcc)):.2f}) | rmse_G: {np.sqrt(running_loss / total_samples):.2e}| mse_G: {lmda_MSE*mse_G.item():.2e} | Df_loss: {lmda_D*Df_loss.item():.2e}")
#         #     #msg = (f"Epoch {epoch}/{epochs} rmse_G: {np.sqrt(running_loss / total_samples):.2e}")

#         #     #sys.stdout.write("\r" + msg)
#         #     print(msg)

#     return G, D


# # 5) Usage example
# if __name__ == "__main__":

#     G_model, D_model = train_gan(G, D, train_loader, epochs=num_epochs, device=device)

#     G.eval()
#     r_hr = df_high_test.iloc[0].pes["r"].to_numpy()
#     energy_hr = df_high_test.iloc[0].pes["energy"].to_numpy()
#     indices = np.arange(0, energy_hr.shape[0] - 1, upscale_factor)

#     energy_lr = energy_hr[indices]
#     r_lr = r_hr[indices]
#     energy_pred = (
#         G(torch.from_numpy(energy_lr).float().unsqueeze(0).unsqueeze(1).to(device))
#         .squeeze(0)
#         .squeeze(0)
#         .cpu()
#         .detach()
#         .numpy()
#     )

#     print(energy_pred.shape)
#     plt.plot(r_hr, energy_hr, label="High-PES", color="blue")
#     plt.scatter(r_lr, energy_lr, label="Low-PES", color="g")
#     plt.plot(r_hr, energy_pred, "r.--", label="Model PES")
#     plt.show()

#     G.eval()
#     loss_fn = nn.MSELoss()
#     with torch.no_grad():  # deactivates autograd
#         for X, y in test_loader:
#             y_pred = G.forward(X)
#             loss = loss_fn(y_pred, y)
#             test_mse = loss.item()

#     print(f"Test MSE: {np.sqrt(test_mse):.2e}")
