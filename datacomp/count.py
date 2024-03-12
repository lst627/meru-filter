import numpy as np

# Specify the path to the .npy file
file_path = "/local1/siting/clip_score_l14_30_percent.npy"

# Load the .npy file
data = np.load(file_path)

# Print the contents of the .npy file
print(data.shape)

# Specify the path to the .npy file
file_path = "/local1/siting/clip_score_l14_30_percent_new.npy"

# Load the .npy file
data = np.load(file_path)

# Print the contents of the .npy file
print(data.shape)