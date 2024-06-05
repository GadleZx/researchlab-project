import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Define the filename of your CSV file
filename = "./data/data_002_trajectories.csv"

# Read the data from the CSV file
data = pd.read_csv(filename)

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Get unique IDs and generate a colormap for each ID
unique_ids = data["id"].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ids)))

# Initialize a dictionary to store the scatter plot objects
scatter_dict = {}

# Plot initialization
for i, id_value in enumerate(unique_ids):
    # Select data for the specific ID
    id_data = data[data["id"] == id_value]
    
    # Initialize scatter plot for each ID with the first frame's data
    initial_frame_data = id_data[id_data["frame_num"] == id_data["frame_num"].min()]
    scatter = ax.scatter(initial_frame_data["x"], initial_frame_data["y"], initial_frame_data["z"], c=np.array([colors[i]]), label=f"ID: {id_value}", s=100)
    scatter_dict[id_value] = scatter
    
# Customize the plot
ax.set_xlabel("X-position")
ax.set_ylabel("Y-position")
ax.set_zlabel("Z-position")
ax.set_title("Trajectories of All IDs")

# Calculate the range for each axis
x_min, x_max = data["x"].min(), data["x"].max()
y_min, y_max = data["y"].min(), data["y"].max()
z_min, z_max = data["z"].min(), data["z"].max()

# Set the limits for each axis
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Optionally, add a legend if there are not too many IDs
if len(unique_ids) <= 10:
    ax.legend()
else:
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Function to update the plot for each frame
def update(frame_num):
    for id_value in unique_ids:
        # Select data for the specific ID and frame
        id_data = data[(data["id"] == id_value) & (data["frame_num"] == frame_num)]
        if not id_data.empty:
            scatter = scatter_dict[id_value]
            scatter._offsets3d = (id_data["x"].values, id_data["y"].values, id_data["z"].values)

# Create the animation
frame_nums = sorted(data["frame_num"].unique())
ani = FuncAnimation(fig, update, frames=frame_nums, interval=100)

# Save the animation as a video file
writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save("trajectories_animation.mp4", writer=writer)

# Optionally, you can also show the animation
plt.show()
