# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x)

plt.plot(x, y)
plt.show()
