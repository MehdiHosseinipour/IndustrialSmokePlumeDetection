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

from model_unet import *
from test_data import create_dataset

np.random.seed(3)
torch.manual_seed(3)

# load data
valdata = create_dataset(datadir='./test2', apply_transforms=False)

batch_size = 1  # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load('segmentation.model', map_location=torch.device('cpu')))
model.eval()

# To store area in square meters over time
smoke_areas_m2 = []

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
    
    smoke_areas_m2.append(area_pred_m2[0])  # Append area for each image in square meters
    
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

        # false color plot
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

        # segmentation ground-truth and prediction
        ax3.imshow(pred[0][0], cmap='Greens', alpha=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)
        f.tight_layout()
        plt.savefig((os.path.split(batch['imgfile'][0])[1]).replace('.tif', '_eval.png').replace(':', '_'), dpi=200)
        plt.close()

# Create and save the time series graph in square meters
time_points = np.arange(len(smoke_areas_m2))  # Assuming images are processed in order over time
plt.figure(figsize=(10, 6))
plt.plot(time_points, smoke_areas_m2, marker='o', color='b', linestyle='-', label="Smoke Area (m²)")
plt.xlabel("Time (Image Index)", fontsize=12)
plt.ylabel("Smoke Area (m²)", fontsize=12)
plt.title("Smoke Area Over Time (in Square Meters)", fontsize=14)
plt.grid(True)
plt.legend(loc='best')
plt.savefig("smoke_area_over_time_m2.png", dpi=300)
plt.show()
