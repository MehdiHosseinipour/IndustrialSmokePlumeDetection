import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from model_unet import *
from test_data import create_dataset

np.random.seed(3)
torch.manual_seed(3)

# Load data (only images, no labels)
valdata = create_dataset(datadir='./test', apply_transforms=True)

batch_size = 1  # Set batch size to 1 for diagnostic images
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=False)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# Load model
model.load_state_dict(torch.load('segmentation.model', map_location=torch.device('cpu')))
model.eval()

# Run through test data (only images, no labels)
for i, batch in progress:
    x = batch['img'].float().to(device)
    idx = batch['idx']

    output = model(x)

    # Obtain binary prediction map (no label comparison)
    pred = np.zeros(output.shape)
    output = output.cpu()
    pred[output >= 0] = 1

    # Save output image results (if required)
    if batch_size == 1:
        f, ax = plt.subplots(1, 1, figsize=(6, 6))

        # RGB image
        ax.imshow(np.dstack([x[0][3], x[0][2], x[0][1]]), origin='upper')
        ax.set_title(f"Prediction for {os.path.split(batch['imgfile'][0])[1]}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Save plot
        plt.savefig(f"output_{os.path.split(batch['imgfile'][0])[1]}", dpi=200)
        plt.close()

print('Evaluation complete. Outputs saved.')
