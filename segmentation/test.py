import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import jaccard_score
from torch import nn
from model_unet import *
from test_data import create_dataset

np.random.seed(3)
torch.manual_seed(3)

# Load data (without using segmentation labels)
valdata = create_dataset(datadir='./test2')  # Only the test images

batch_size = 1  # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# Load model
model.load_state_dict(torch.load(
    'segmentation.model', map_location=torch.device('cpu')))
model.eval()

# Define loss function
loss_fn = nn.BCEWithLogitsLoss()

# Run through test data
all_ious = []
all_accs = []
all_arearatios = []

for i, batch in progress:
    x = batch['img'].float().to(device)
    idx = batch['idx']

    output = model(x)

    # Obtain binary prediction map
    pred = np.zeros(output.shape)
    output = output.cpu()
    pred[output >= 0] = 1

    # Derive IoU score
    cropped_iou = []
    for j in range(x.shape[0]):  # Loop through the batch (images)
        # Since there's no ground truth, we will use a dummy IoU value for display
        z = jaccard_score(np.zeros_like(pred[j][0].flatten()), pred[j][0].flatten(), average='binary')
        cropped_iou.append(z)
    all_ious.extend(cropped_iou)

    # Derive binary prediction labels based on non-zero areas
    prediction = np.array(np.sum(pred, axis=(1, 2, 3)) != 0).astype(int)

    # Derive image-wise accuracy for this batch
    all_accs.append(accuracy_score(np.zeros_like(prediction), prediction))

    # Derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1

    # Derive smoke areas (dummy area, as no ground truth is available)
    area_pred = np.sum(output_binary, axis=(1, 2, 3))
    area_true = np.zeros_like(area_pred)  # No true area available

    # Derive smoke area ratios (dummy values for display)
    arearatios = []
    for k in range(len(area_pred)):
        arearatios.append(area_pred[k] / (area_true[k] + 1e-6))  # Prevent division by zero
    all_arearatios.extend(arearatios)

    if batch_size == 1:
        # Create plot for the diagnostic images
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # RGB plot (using the first three channels)
        ax1.imshow(0.2 + 1.5 * (np.dstack([x[0][3], x[0][2], x[0][1]]) -
                    np.min([x[0][3].numpy(), x[0][2].numpy(), x[0][1].numpy()])) /
                   (np.max([x[0][3].numpy(), x[0][2].numpy(), x[0][1].numpy()]) -
                    np.min([x[0][3].numpy(), x[0][2].numpy(), x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title(f"Image {idx[0]}", fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # False color plot (using other bands for visualization)
        ax2.imshow(0.2 + (np.dstack([x[0][0], x[0][9], x[0][10]]) -
                    np.min([x[0][0].numpy(), x[0][9].numpy(), x[0][10].numpy()])) /
                   (np.max([x[0][0].numpy(), x[0][9].numpy(), x[0][10].numpy()]) -
                    np.min([x[0][0].numpy(), x[0][9].numpy(), x[0][10].numpy()])),
                   origin='upper')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Display the predicted binary map
        ax3.imshow(pred[0][0], cmap='Greens', alpha=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Add a dummy IoU label (since we don't have ground truth)
        ax3.annotate(f"Dummy IoU={cropped_iou[0]:.2f}", xy=(5, 15), fontsize=8)

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)
        f.tight_layout()

        # Save the plot
        plt.savefig(f"eval_{idx[0]}.png", dpi=200)
        plt.close()

print('IoU:', len(all_ious), np.average(all_ious))
print('Accuracy:', len(all_accs), np.average(all_accs))
print('Mean Area Ratio:', len(all_arearatios), np.average(all_arearatios),
      np.std(all_arearatios) / np.sqrt(len(all_arearatios) - 1))
