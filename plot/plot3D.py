import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

# Read data
cycle = "V"
N = 4096
filename = "SOL_" + cycle + str(N) + ".txt"
data = pd.read_csv("../src/" + filename, sep=',', header=None)

# Solutions from MG_solver_GPU.cu
z = data.values

# Create x, y coordinates
x = np.linspace(0, 1, z.shape[0], endpoint=True)
y = np.linspace(0, 1, z.shape[0], endpoint=True)
x, y = np.meshgrid(x, y)
y = y[::-1]   # flip the array, since the output file order is from y from top to bottom, x from left to right

# Analytic solution
ans = np.exp(x - y) * x * (1 - x) * y * (1 - y)

# Print out the error
print("( N, err, cycle ) = ( %d, %lf, %s )" % (N, np.sum(np.abs(ans - z)) / (N*N), cycle))

# Plot the error
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("| Analytic - Solution |\n %s-cycle, N = %d" % (cycle, N), fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
surf = ax.plot_surface(x, y, np.abs(ans - z), cmap=cm.seismic)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Plot the Analytic Solution
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("Analytic", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
surf = ax.plot_surface(x, y, ans, cmap=cm.seismic)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Plot the solution from MG Method
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title("MG Method Solution", fontsize=14)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
surf = ax.plot_surface(x, y, z, cmap=cm.seismic)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()