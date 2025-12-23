import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import jaccard_score
import pandas as pd
from datetime import datetime  # Import datetime module
from scipy.ndimage import label

from model_unet import *
from test_data import create_dataset

np.random.seed(3)
torch.manual_seed(3)

# Set font to Times New Roman globally
plt.rcParams['font.family'] = 'serif'

# load data
valdata = create_dataset(datadir='./test2', apply_transforms=False)

batch_size = 1  # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load('segmentation.model', map_location=torch.device('cpu')))
model.eval()

# To store area in square kilometers over time and the corresponding dates
smoke_areas_km2 = []
image_dates = []

# Loop to process each batch
for i, batch in progress:
    x = batch['img'].float().to(device)
    idx = batch['idx']

    output = model(x)
    output = output.cpu()
    x = x.cpu()
    
    # Obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1
    
    # Derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1

    # Identify connected components (smoke patches)
    output_binary_numpy = output_binary[0][0].astype(np.uint8)  # Convert tensor to NumPy array
    labeled_array, num_features = label(output_binary_numpy)  # Label connected components
    
    # Get the coordinates of the central region (9 central pixels in a 3x3 block)
    height, width = output_binary[0][0].shape
    center_x, center_y = width // 2, height // 2  # Coordinates of the central pixel
    central_region = output_binary[0][0][center_y-1:center_y+2, center_x-1:center_x+2]  # 3x3 region
    
    # Check if any of the connected components (smoke patches) intersect with the central region
    central_patch = None
    for label_num in range(1, num_features + 1):  # Label starts at 1
        # Create a mask for each connected component
        component_mask = (labeled_array == label_num)
        
        # Check if the component intersects with the central region
        if np.any(component_mask[center_y-1:center_y+2, center_x-1:center_x+2]):  # Check overlap with 3x3 region
            central_patch = label_num
            break  # Keep the first patch found in the center region

    # If a central patch exists, keep it
    if central_patch is not None:
        # Keep only the pixels corresponding to the central patch
        output_binary[0][0] = (labeled_array == central_patch)

        # Visualize only the central patch (mask the rest of the image)
        central_smoke_area = np.zeros_like(output_binary[0][0])  # Create an empty image
        central_smoke_area[output_binary[0][0] == 1] = 1  # Set the central patch to 1

        # Plot the result showing only the central smoke area
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # RGB plot
        ax1.imshow(0.2 + 1.5 * (np.dstack([x[0][3], x[0][2], x[0][1]]) - np.min([x[0][3].numpy(),
                                                                                       x[0][2].numpy(),
                                                                                       x[0][1].numpy()])) / 
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]) - np.min([x[0][3].numpy(),
                                                       x[0][2].numpy(),
                                                       x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title('RGB', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # False color plot
        ax2.imshow(0.2 + (np.dstack([x[0][0], x[0][9], x[0][10]]) - np.min([x[0][0].numpy(),
                                                                              x[0][9].numpy(),
                                                                              x[0][10].numpy()])) / 
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]) - np.min([x[0][0].numpy(),
                                                       x[0][9].numpy(),
                                                       x[0][10].numpy()])),
                   origin='upper')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Segmentation ground-truth and central patch only
        ax3.imshow(central_smoke_area, cmap='Greens', alpha=0.5)  # Show only the central patch
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)
        f.tight_layout()
        plt.savefig((os.path.split(batch['imgfile'][0])[1]).replace('.tif', '_central_eval.png').replace(':', '_'), dpi=200)
        plt.close()

# Continue with sorting and saving data to Excel
# Sort the areas by date
sorted_data = sorted(zip(image_dates, smoke_areas_km2), key=lambda x: x[0])

# Unzip the sorted data back into sorted lists
sorted_dates, sorted_areas = zip(*sorted_data)

# Create and save the time series graph in square kilometers
plt.figure(figsize=(10, 6))
plt.plot(sorted_dates, sorted_areas, marker='o', color='b', linestyle='-', label="Smoke Area (km²)")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Smoke Area (km²)", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(loc='best')
plt.tight_layout()  # Ensure tight layout for the plot

# Save the plot as a PNG image
plt.savefig("smoke_area_over_time_km2_sorted.png", dpi=600)
plt.show()

# Save the sorted data to an Excel file
df = pd.DataFrame({
    'Date': sorted_dates,
    'Smoke Area (km²)': sorted_areas
})

# Save to Excel file with datetime formatted Date column
df.to_excel("smoke_area_over_time_sorted.xlsx", index=False)

print("Excel file saved as 'smoke_area_over_time_sorted.xlsx'")
