import matplotlib.pyplot as plt
import numpy as np

nfiles = 5
gen_vs_spline = np.zeros((10, 2 * nfiles, 12))

for i in range(nfiles):
    gen_vs_spline[:, 2 * i : 2 * i + 2, :] = np.load(
        "results/gen_vs_spline" + str(i + 1) + ".npy"
    )

scaling = [4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]
grid_size = [4, 5, 6, 8, 10, 12, 14, 16, 20, 24]

# arr[0, i, j] = np.sqrt(loss_train)
# arr[1, i, j] = np.sqrt(loss_test)

# arr[4, i, j] = mean_model
# arr[7, i, j] = mean_spline

# arr[8, i, j] = mean_model_O2
# arr[9, i, j] = mean_spline_O2

# Create the first figure
plt.figure()

# Data for plotting
x = scaling
grid_size_plot = [0, 1, 4, 6, 8, -1]
for i in grid_size_plot:
    y = 100 * np.abs(
        (gen_vs_spline[0, i, :] - gen_vs_spline[1, i, :]) / gen_vs_spline[0, i, :]
    )
    plt.plot(x, y, label="Grid size: {}".format(grid_size[i]))

plt.title("Train - Test RMSE")
plt.xlabel("Upscaling factor")
plt.ylabel("(Train-Test)/Train (%)")
plt.legend()


spline_ljs = np.min(gen_vs_spline[7, :, :], axis=1)

# Create the second figure
plt.figure()
plt.plot(grid_size, spline_ljs, label="spline")

for i in range(len(scaling)):
    y = gen_vs_spline[4, :, i]
    plt.plot(grid_size, y, label="Scaling : {}".format(scaling[i]))


plt.title("10^4 Random Samples(LJ and Morse)")
plt.yscale("log")
plt.xlabel("grid size")
plt.ylabel("RMSE")
plt.legend()


# Create the second figure
plt.figure()
plt.plot(grid_size, spline_ljs, label="spline")

for i in range(0, len(scaling), 3):
    y = gen_vs_spline[4, :, i]
    plt.plot(grid_size, y, label="Scaling : {}".format(scaling[i]))


plt.title("10^4 Random Samples(LJ and Morse)")
plt.yscale("log")
plt.xlabel("grid size")
plt.ylabel("RMSE")
plt.legend()


# Create the second figure
plt.figure()

for i in range(0, len(grid_size), 2):
    y = gen_vs_spline[4, i, :]
    plt.plot(scaling, y, label="Grid size : {}".format(grid_size[i]))


plt.title("10^4 Random Samples(LJ and Morse)")
plt.yscale("log")
plt.xlabel("grid size")
plt.ylabel("RMSE")
plt.legend()

# Display the figures
plt.show()
