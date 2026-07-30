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
from matplotlib.patches import Circle, Rectangle

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

# -------------------------
# LOAD DATA
# -------------------------
valdata = create_dataset(datadir='./test2', apply_transforms=False)
all_dl = DataLoader(valdata, batch_size=1, shuffle=False)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# -------------------------
# LOAD MODEL
# -------------------------
model.load_state_dict(
    torch.load('segmentation_best.model', map_location=torch.device('cpu'))
)
model.eval()

# -------------------------
# RESULTS STORAGE
# -------------------------
kept_dates = []
kept_areas = []
kept_files = []

os.makedirs(OUT_IMG_DIR, exist_ok=True)

# -------------------------
# MAIN LOOP
# -------------------------
for i, batch in progress:
    x = batch['img'].float().to(device)
    imgfile = batch['imgfile'][0]

    # ---- DATE ----
    img_date = parse_date_from_filename(imgfile)
    if img_date is None:
        print(f"Warning: No YYYYMMDD found in filename: {imgfile}. Skipping.")
        continue

    # ---- MODEL OUTPUT ----
    output = model(x).detach().cpu().numpy()      # [B,1,H,W]
    x_cpu = x.detach().cpu()

    # ---- BINARY MAP ----
    output_binary_2d = (output[0, 0] > 0).astype(np.uint8)
    
    # ---- USE FULL SMOKE MASK (NO CONNECTED COMPONENT FILTER) ----
    smoke_area_mask = output_binary_2d.copy()

    # ---- AREA ----
    area_km2 = area_km2_from_mask(smoke_area_mask, PATCH_SIZE_M)

    # ---- KEEP ONLY area < threshold ----
    if area_km2 >= AREA_MAX_KM2:
        continue

    kept_dates.append(img_date)
    kept_areas.append(area_km2)
    kept_files.append(os.path.basename(imgfile))

    # -------------------------
    # SAVE DIAGNOSTIC IMAGE
    # -------------------------
    f, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 6))

    # ---- RGB IMAGE ----
    ax1.imshow(
        0.2 + 1.5 * (
            np.dstack([x_cpu[0][3], x_cpu[0][2], x_cpu[0][1]]) -
            np.min([x_cpu[0][3].numpy(),
                    x_cpu[0][2].numpy(),
                    x_cpu[0][1].numpy()])
        ) / (
            np.max([x_cpu[0][3].numpy(),
                    x_cpu[0][2].numpy(),
                    x_cpu[0][1].numpy()]) -
            np.min([x_cpu[0][3].numpy(),
                    x_cpu[0][2].numpy(),
                    x_cpu[0][1].numpy()])
        ),
        origin='upper'
    )
    ax1.set_title('RGB', fontsize=8)
    ax1.set_xticks([]); ax1.set_yticks([])

    # ---- RED CENTER CIRCLE (FLARE) ----
    center_circle = Circle(
        (w // 2, h // 2),
        radius=min(h, w) * 0.03,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )
    ax1.add_patch(center_circle)

    # ---- FLARE LABEL ----
    ax1.text(
        0.5, -0.10,
        "Flare",
        transform=ax1.transAxes,
        ha='center',
        va='top',
        fontsize=9,
        color='red',
        fontweight='bold'
    )

    # ---- SMOKE MASK ----
    ax3.imshow(smoke_area_mask, cmap='Greens', alpha=0.7)
    ax3.set_title(f"Central patch | {area_km2:.4f} km²", fontsize=8)
    ax3.set_xticks([]); ax3.set_yticks([])

    # ---- LEGEND LABELS ----
    legend_y = -0.25
    box_size = 0.04

    # Black: Filtered Pixel
    ax3.add_patch(Rectangle(
        (0.05, legend_y), box_size, box_size,
        transform=ax3.transAxes,
        facecolor='black',
        edgecolor='black',
        clip_on=False
    ))
    ax3.text(0.11, legend_y + box_size / 2,
             "Filtered Pixel",
             transform=ax3.transAxes,
             va='center',
             fontsize=8)

    # Green: Smoke
    ax3.add_patch(Rectangle(
        (0.40, legend_y), box_size, box_size,
        transform=ax3.transAxes,
        facecolor='green',
        edgecolor='black',
        clip_on=False
    ))
    ax3.text(0.46, legend_y + box_size / 2,
             "Smoke",
             transform=ax3.transAxes,
             va='center',
             fontsize=8)

    # White: No Smoke
    ax3.add_patch(Rectangle(
        (0.65, legend_y), box_size, box_size,
        transform=ax3.transAxes,
        facecolor='white',
        edgecolor='black',
        clip_on=False
    ))
    ax3.text(0.71, legend_y + box_size / 2,
             "No Smoke",
             transform=ax3.transAxes,
             va='center',
             fontsize=8)

    f.tight_layout()

    out_png = os.path.basename(imgfile).replace(
        '.tif', f'_central_lt_{AREA_MAX_KM2}km2.png'
    ).replace(':', '_')

    plt.savefig(os.path.join(OUT_IMG_DIR, out_png), dpi=1200)
    plt.close()

# -------------------------
# EXPORT RESULTS
# -------------------------
if len(kept_dates) == 0:
    raise ValueError(
        f"No samples found with area < {AREA_MAX_KM2} km². Nothing to export."
    )

df = pd.DataFrame({
    'Date': kept_dates,
    'Smoke Area (km²)': kept_areas,
    'Filename': kept_files
}).sort_values('Date').reset_index(drop=True)

df.to_excel(OUT_EXCEL, index=False)
print(f"Excel saved: {OUT_EXCEL}")
print(f"Saved diagnostic images to folder: {OUT_IMG_DIR}")

# -------------------------
# TIME SERIES PLOT
# -------------------------
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
