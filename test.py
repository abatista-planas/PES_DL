import matplotlib.pyplot as plt
import numpy as np

loss = np.load("res_2.npy")

print(loss.shape)
n = loss.shape[2]
labels_strick = [2 * i + (i - 2) * 2 for i in range(1, n)]
title = ["Train", "Test", "Best", "Worst", "Mean"]
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

upcaling = [4, 8, 16, 32]
color = ["r", "g", "b", "y"]
fig, ax1 = plt.subplots(nrows=1, sharex=True)
for i in range(2, n, 2):
    grid_size = 2 * np.arange(2, 18)
    mean = loss[
        4,
        i,
        2:n,
    ]
    worst = loss[
        3,
        i,
        2:n,
    ]
    best = loss[
        2,
        i,
        2:n,
    ]

    asymmetric_error = [best, worst]
    ax1.errorbar(grid_size, mean, label="grid_size = " + str(2 * i), fmt="o--")

ax1.set_title("Upscale")
ax1.set_ylabel("RMSE")
ax1.set_xlabel("Upscale")
ax1.legend()

plt.show()
