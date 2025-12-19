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

    # Derive smoke areas in pixels
    area_pred_pixels = np.sum(output_binary, axis=(1,2,3))  # Sum over the image dimensions (height, width, channels)
    
    # Convert to square meters (1 pixel = 10m * 10m = 100m²)
    area_pred_m2 = area_pred_pixels * 100  # 100 square meters per pixel
    
    # Convert to square kilometers (1 km² = 1,000,000 m²)
    area_pred_km2 = area_pred_m2 / 1000000  # Convert to square kilometers
    
    smoke_areas_km2.append(area_pred_km2[0])  # Append area for each image in square kilometers
    
    # Extract the date from the image filename (assuming the format 's2_patch_YYYYMMDD_scaled_120x120.tif')
    image_filename = batch['imgfile'][0]
    
    # Extract the date part (YYYYMMDD) from the filename
    date_str = os.path.splitext(os.path.basename(image_filename))[0].split('_')[2]  # Extract date (YYYYMMDD)
    
    # Convert to datetime object
    formatted_date = datetime.strptime(date_str, "%Y%m%d").date()  # Convert to datetime.date object
    image_dates.append(formatted_date)  # Append the formatted date to the list

    # For diagnostics (optional)
    if batch_size == 1:
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

        # Segmentation ground-truth and prediction
        ax3.imshow(pred[0][0], cmap='Greens', alpha=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)
        f.tight_layout()
        plt.savefig((os.path.split(batch['imgfile'][0])[1]).replace('.tif', '_eval.png').replace(':', '_'), dpi=200)
        plt.close()

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
