import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
import torch
from torchvision import transforms

torch.manual_seed(3)
np.random.seed(3)

class SmokePlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, datadir=None, transform=None):
        """SmokePlumeSegmentation Dataset class."""
        self.datadir = datadir
        self.transform = transform

        # list of image files
        self.imgfiles = []

        # read in image file names
        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                self.imgfiles.append(os.path.join(root, filename))
                idx += 1

        # convert list into array
        self.imgfiles = np.array(self.imgfiles)

    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)

    def __getitem__(self, idx):
        """Read in image data and apply transformations."""
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1,2,3,4,5,6,7,8,9,10,12,13]])

        # force image shape to be 120 x 120 pixels
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

        sample = {'idx': idx,
                  'img': imgdata,
                  'imgfile': self.imgfiles[idx]}

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset."""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            Randomize(),
            RandomCrop(),
            ToTensor()
        ])
    else:
        data_transforms = None

    data = SmokePlumeSegmentationDataset(*args, **kwargs, transform=data_transforms)
    return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """
        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'imgfile': sample['imgfile']}

        return out


class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """
        sample['img'] = (sample['img'] - self.channel_means.reshape(
            sample['img'].shape[0], 1, 1)) / self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample


class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1, 2))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'imgfile': sample['imgfile']}


class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'imgfile': sample['imgfile']}


def display(self, idx):
    """Helper method to display a given example from the data set with
    index `idx`. Only RGB channels are displayed.

    :param idx: (int) image index to be displayed
    """
    sample = self[idx]
    imgdata = sample['img']

    # scale image data
    imgdata = (imgdata - np.min(imgdata)) / (np.max(imgdata) - np.min(imgdata))

    f, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.imshow(imgdata.transpose(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])

    return f
