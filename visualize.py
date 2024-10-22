import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the fixed sentiment values for pos and neu
pos = 0.5  # Example positive sentiment
neu = 0.2  # Example neutral sentiment

# Generate a range of values for k and neg from 0 to 1
k_values = np.linspace(0, 1, 100)
neg_values = np.linspace(0, 1, 100)

# Create a meshgrid for k and neg
K, NEG = np.meshgrid(k_values, neg_values)

# Calculate the tendency for each combination of k and neg
TENDENCY = (K * NEG + (1 - K) * pos + (1 - np.abs(2 * K - 1)) * neu) / (K + (1 - K) + (1 - np.abs(2 * K - 1)))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(K, NEG, TENDENCY, cmap='viridis')

# Label the axes
ax.set_xlabel('k')
ax.set_ylabel('neg')
ax.set_zlabel('Tendency')

# Show the plot
plt.title('Tendency vs k and neg')
plt.show()
