import os
import numpy as np
from torch.utils.data import Dataset
import torch
import rasterio as rio

class SmokePlumeSegmentationDataset(Dataset):
    """Dataset class to handle images in the test directory without labels."""

    def __init__(self, datadir=None, transform=None):
        """Dataset class for images in the test directory.

        :param datadir: (str) image directory root, required
        :param transform: (`torchvision.transform` object) transformations to be applied, default: `None`
        """
        self.datadir = datadir
        self.transform = transform
        self.imgfiles = []

        # Only consider image files in the test directory
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if filename.endswith('.tif'):
                    self.imgfiles.append(os.path.join(root, filename))

    def __len__(self):
        """Returns length of dataset."""
        return len(self.imgfiles)

    def __getitem__(self, idx):
        """Read in image data and apply transformations."""
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]])

        # Force image shape to 120 x 120 pixels
        if imgdata.shape[1] != 120:
            newimgdata = np.empty((12, 120, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:,
                                                          :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:,
                                                          imgdata.shape[1]-1:, :]
            imgdata = newimgdata
        if imgdata.shape[2] != 120:
            newimgdata = np.empty((12, 120, 120))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:,
                                                          :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,
                                                          :, imgdata.shape[2]-1:]
            imgdata = newimgdata

        sample = {'idx': idx, 'img': imgdata, 'imgfile': self.imgfiles[idx]}

        # Apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses the same input parameters as PowerPlantDataset."""
    if apply_transforms:
        data_transforms = transforms.Compose([
            ToTensor(),
            Normalize(),
            Randomize(),
            RandomCrop()
        ])
    else:
        data_transforms = None

    data = SmokePlumeSegmentationDataset(*args, **kwargs, transform=data_transforms)

    return data
