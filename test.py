import matplotlib.pyplot as plt
import numpy as np

loss = np.load("loss.npy")

print(loss.shape)
n = loss.shape[1]
fig, axes = plt.subplots(nrows=1, ncols=2)
for i, ax in enumerate(axes if isinstance(axes, np.ndarray) else [axes]):
    arr = loss[i, 2:n, 2:n]
    im = ax.matshow(arr, cmap="viridis", interpolation="none")
    ax.set_xticklabels([2 * i for i in range(2, n)], fontsize=12)
    ax.set_yticklabels([2 * i for i in range(2, n)], fontsize=12)
    # Set axis labels
    ax.set_xlabel("Upscale")
    ax.set_ylabel("Grid size")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
fig.colorbar(im, cax=cbar_ax)

plt.show()
# plt.savefig("grid_size_vs_upscaling.png")
