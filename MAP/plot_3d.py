import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define the filename of your CSV file
filename = "./data/data_002_trajectories.csv"

# Read the data from the CSV file
data = pd.read_csv(filename)

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Loop through unique IDs and plot trajectories
unique_ids = data["id"].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ids)))  # Generate colormap for each ID

for i, id_value in enumerate(unique_ids):
    # Select data for the specific ID
    id_data = data[data["id"] == id_value]

    # Extract positions (x, y, z) for each frame
    x = id_data["x"]
    y = id_data["y"]
    z = id_data["z"]

    # Plot the trajectory with a different color for each ID
    ax.plot(x, y, z, color=colors[i], label=f"ID: {id_value}")

# Customize the plot (optional)
ax.set_xlabel("X-position")
ax.set_ylabel("Y-position")
ax.set_zlabel("Z-position")
ax.set_title("Trajectories of All IDs")

# Optionally, if there are too many IDs, you might want to limit the legend
if len(unique_ids) <= 10:
    ax.legend()
else:
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()
