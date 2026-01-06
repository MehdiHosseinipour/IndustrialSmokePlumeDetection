import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import pandas as pd
from datetime import datetime
from scipy.ndimage import label

from model_unet import *
from test_data import create_dataset

np.random.seed(3)
torch.manual_seed(3)
plt.rcParams['font.family'] = 'serif'

# -------------------------
# CONFIG
# -------------------------
PATCH_SIZE_M = 1200            # meters (matches your GEE patch size)
AREA_MAX_KM2 = 0.72            # ONLY keep/save samples with area < 0.72 km²
OUT_IMG_DIR = "filtered_images"
OUT_EXCEL = "smoke_area_filtered_lt_0p72km2.xlsx"
OUT_PLOT = "smoke_area_filtered_lt_0p72km2.png"
# -------------------------

def parse_date_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r'(\d{8})', base)  # YYYYMMDD
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def area_km2_from_mask(binary_mask_2d, patch_size_m):
    h, w = binary_mask_2d.shape
    pixel_area_m2 = (patch_size_m / w) * (patch_size_m / h)
    return float(binary_mask_2d.sum()) * pixel_area_m2 / 1e6

# load data
valdata = create_dataset(datadir='./test2', apply_transforms=False)
all_dl = DataLoader(valdata, batch_size=1, shuffle=False)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load('segmentation.model', map_location=torch.device('cpu')))
model.eval()

# results (only for area < 0.72)
kept_dates = []
kept_areas = []
kept_files = []

os.makedirs(OUT_IMG_DIR, exist_ok=True)

for i, batch in progress:
    x = batch['img'].float().to(device)
    imgfile = batch['imgfile'][0]

    # ---- DATE ----
    img_date = parse_date_from_filename(imgfile)
    if img_date is None:
        print(f"Warning: No YYYYMMDD found in filename: {imgfile}. Skipping.")
        continue

    # ---- MODEL OUTPUT ----
    output = model(x).detach().cpu().numpy()      # [B,1,H,W] logits
    x_cpu = x.detach().cpu()

    # ---- BINARY MAP (logits threshold) ----
    output_binary_2d = (output[0, 0] > 0).astype(np.uint8)

    # ---- CONNECTED COMPONENTS ----
    labeled_array, num_features = label(output_binary_2d)

    h, w = output_binary_2d.shape
    cx, cy = w // 2, h // 2
    y0, y1 = max(cy - 1, 0), min(cy + 2, h)
    x0, x1 = max(cx - 1, 0), min(cx + 2, w)

    central_patch = None
    for label_num in range(1, num_features + 1):
        component_mask = (labeled_array == label_num)
        if np.any(component_mask[y0:y1, x0:x1]):
            central_patch = label_num
            break

    if central_patch is not None:
        central_smoke_area = (labeled_array == central_patch).astype(np.uint8)
    else:
        central_smoke_area = np.zeros_like(output_binary_2d, dtype=np.uint8)

    # ---- AREA ----
    area_km2 = area_km2_from_mask(central_smoke_area, PATCH_SIZE_M)

    # ---- KEEP ONLY area < 0.72 km² ----
    if area_km2 >= AREA_MAX_KM2:
        continue

    # store results
    kept_dates.append(img_date)
    kept_areas.append(area_km2)
    kept_files.append(os.path.basename(imgfile))

    # ---- SAVE DIAGNOSTIC IMAGE (ONLY for kept samples) ----
    f, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 6))  # Adjusted figsize to be larger

    # RGB (B4,B3,B2) -> indices depend on your dataset order
    ax1.imshow(
        0.2 + 1.5 * (np.dstack([x_cpu[0][3], x_cpu[0][2], x_cpu[0][1]]) - np.min([x_cpu[0][3].numpy(),
                                                                              x_cpu[0][2].numpy(),
                                                                              x_cpu[0][1].numpy()])) / 
        (np.max([x_cpu[0][3].numpy(),
                 x_cpu[0][2].numpy(),
                 x_cpu[0][1].numpy()]) - np.min([x_cpu[0][3].numpy(),
                                                 x_cpu[0][2].numpy(),
                                                 x_cpu[0][1].numpy()])),
        origin='upper'
    )
    ax1.set_title('RGB', fontsize=8)
    ax1.set_xticks([]); ax1.set_yticks([])

    # # False color (your previous indices, if you want to include it again)
    # Uncomment if needed
    # ax2.imshow(
    #     0.2 + (np.dstack([x_cpu[0][0], x_cpu[0][9], x_cpu[0][10]]) - np.min([x_cpu[0][0].numpy(),
    #                                                                         x_cpu[0][9].numpy(),
    #                                                                         x_cpu[0][10].numpy()])) / 
    #     (np.max([x_cpu[0][0].numpy(),
    #              x_cpu[0][9].numpy(),
    #              x_cpu[0][10].numpy()]) - np.min([x_cpu[0][0].numpy(),
    #                                               x_cpu[0][9].numpy(),
    #                                               x_cpu[0][10].numpy()])),
    #     origin='upper'
    # )
    # ax2.set_xticks([]); ax2.set_yticks([])

    ax3.imshow(central_smoke_area, cmap='Greens', alpha=0.7)
    ax3.set_title(f"Central patch | {area_km2:.4f} km²", fontsize=8)
    ax3.set_xticks([]); ax3.set_yticks([])

    f.tight_layout()

    out_png = os.path.basename(imgfile).replace('.tif', f'_central_lt_{AREA_MAX_KM2}km2.png').replace(':', '_')
    out_png_path = os.path.join(OUT_IMG_DIR, out_png)
    plt.savefig(out_png_path, dpi=1200)
    plt.close()

# ---- EXPORT FILTERED RESULTS TO EXCEL + PLOT ----
if len(kept_dates) == 0:
    raise ValueError(f"No samples found with area < {AREA_MAX_KM2} km². Nothing to export.")

df = pd.DataFrame({
    'Date': kept_dates,
    'Smoke Area (km²)': kept_areas,
    'Filename': kept_files
}).sort_values('Date').reset_index(drop=True)

df.to_excel(OUT_EXCEL, index=False)
print(f"Excel saved: {OUT_EXCEL}")
print(f"Saved diagnostic images to folder: {OUT_IMG_DIR}")

# Plot the smoke area over time
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Smoke Area (km²)'], marker='o', linestyle='-')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Smoke Area (km²)", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=600)
plt.show()
print(f"Plot saved: {OUT_PLOT}")
