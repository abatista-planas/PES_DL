import numpy as np

# print(loss.shape)
# n = loss.shape[2]
# labels_strick = [2 * i + (i - 2) * 2 for i in range(1, n)]
# title = ["Train", "Test", "Best", "Worst", "Mean"]
# fig, axes = plt.subplots(nrows=1, ncols=2)
# for i, ax in enumerate(axes if isinstance(axes, np.ndarray) else [axes]):
#     arr = loss[i, 2:n, 2:n]
#     im = ax.matshow(arr, cmap="viridis", interpolation="none")
#     ax.set_xticklabels(labels_strick, fontsize=12)
#     ax.set_yticklabels(labels_strick, fontsize=12)
#     # Set axis labels
#     ax.set_xlabel("Upscale")
#     ax.set_ylabel("Grid size")
#     ax.set_title(title[i], fontsize=14)
#     ax.set_aspect("equal")

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
# fig.colorbar(im, cax=cbar_ax)

# plt.show()


# # plt.savefig("grid_size_vs_upscaling.png")
# fig, axes = plt.subplots(nrows=1, ncols=3)
# for i, ax in enumerate(axes if isinstance(axes, np.ndarray) else [axes]):
#     arr = loss[i+2, 2:n, 2:n]
#     im = ax.matshow(arr, cmap="viridis", interpolation="none")
#     ax.set_xticklabels(labels_strick, fontsize=12)
#     ax.set_yticklabels(labels_strick, fontsize=12)
#     # Set axis labels
#     ax.set_xlabel("Upscale")
#     ax.set_ylabel("Grid size")
#     ax.set_title(title[i+2], fontsize=14)
#     ax.set_aspect("equal")
#     print(i,title[i+2])

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
# fig.colorbar(im, cax=cbar_ax)

# plt.show()
# example data

# upcaling = [4, 8, 16, 32]
# color = ["r", "g", "b", "y"]
# fig, ax1 = plt.subplots(nrows=1, sharex=True)
# for i in range(2, n, 2):
#     grid_size = 2 * np.arange(2, 18)
#     mean = loss[
#         4,
#         i,
#         2:n,
#     ]
#     worst = loss[
#         3,
#         i,
#         2:n,
#     ]
#     best = loss[
#         2,
#         i,
#         2:n,
#     ]

#     asymmetric_error = [best, worst]
#     ax1.errorbar(grid_size, mean, label="grid_size = " + str(2 * i), fmt="o--")

# ax1.set_title("Upscale")
# ax1.set_ylabel("RMSE")
# ax1.set_xlabel("Upscale")
# ax1.legend()

# plt.show()

# n_max = 36
# arr = np.load("cubic_rmse.npy")

# gen = np.load("generator_rmse.npy")

# print(gen.shape)

# plt.errorbar(
#     np.arange(4, 16, 2),
#     gen[4, 2:, 2],
#     # yerr=[
#     #     np.min(arr[np.arange(8, n_max, 2), :], axis=1),
#     #     np.max(arr[np.arange(8, n_max, 2), :], axis=1),
#     # ],
#     fmt="o--",
#     label="x4 Upscaling",
# )
# plt.errorbar(
#     np.arange(4, 16, 2),
#     gen[4, 2:, 3],
#     # yerr=[
#     #     np.min(arr[np.arange(8, n_max, 2), :], axis=1),
#     #     np.max(arr[np.arange(8, n_max, 2), :], axis=1),
#     # ],
#     fmt="o--",
#     label="x6 Upscaling",
# )

# plt.errorbar(
#     np.arange(4, 16, 2),
#     gen[4, 2:, 7],
#     # yerr=[
#     #     np.min(arr[np.arange(8, n_max, 2), :], axis=1),
#     #     np.max(arr[np.arange(8, n_max, 2), :], axis=1),
#     # ],
#     fmt="o--",
#     label="x14 Upscaling",
# )
# plt.errorbar(
#     np.arange(4, n_max, 2),
#     np.mean(arr[np.arange(4, n_max, 2), :], axis=1),
#     # yerr=[
#     #     np.min(arr[np.arange(8, n_max, 2), :], axis=1),
#     #     np.max(arr[np.arange(8, n_max, 2), :], axis=1),
#     # ],
#     fmt="ro--",
#     label="Cubic Interpolation",
# )


# plt.title("RMSE of Cubic Interpolation for Lennard-Jones Potential")
# plt.xlabel("Number of points")
# plt.ylabel("RMSE")
# plt.legend()
# plt.show()


integers = [1, 2, 3, 4, 5]
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]

# Generate a random list of 10 integers
random_integers = np.random.choice(integers, size=10, p=probabilities, replace=True)

print(random_integers)
