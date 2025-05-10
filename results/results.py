import numpy as np
import matplotlib.pyplot as plt


nfiles = 5
gen_vs_spline = np.zeros((10,2*nfiles, 12))

for i in range(nfiles):
    gen_vs_spline[:,2*i:2*i+2,:]= np.load('/home/albgzz/PES_DL/results/gen_vs_spline'+str(i+1)+'.npy')
    
scaling=  [4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]
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
for i in range(len(grid_size)):
    y = gen_vs_spline[0, i, :] - gen_vs_spline[1, i, :]
    plt.plot(x, y, label='Grid size: {}'.format(grid_size[i]))

plt.title('Train - Test RMSE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# # Create the second figure
# plt.figure()
# plt.plot(x, y2, label='cos(x)')
# plt.title('Figure 2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# Display the figures
plt.show()