# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# IndustrialSmokePlumeDetection is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IndustrialSmokePlumeDetection is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IndustrialSmokePlumeDetection.  If not,
# see <http://www.gnu.org/licenses/>.
#
# If you use this code for your own project, please cite the following
# conference contribution:
#   Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
#   "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
#   Tackling Climate Change with Machine Learning workshop at NeurIPS 2020.
#
#
# This file contains routines for the evaluation of the trained model
# based on the test data set

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np
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
valdata = create_dataset(datadir='./test2',apply_transforms=False
                         )

batch_size = 1 # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load(
    'segmentation.model', map_location=torch.device('cpu')))
model.eval()

all_arearatios = []
for i, batch in progress:
    x = batch['img'].float().to(device)
    idx = batch['idx']

    output = model(x)
    output=output.cpu()
    x=x.cpu()
    # obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1

    
    
    prediction = np.array(np.sum(pred,
                               axis=(1,2,3)) != 0).astype(int)

    

    # derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1

    # derive smoke areas
    area_pred = np.sum(output_binary, axis=(1,2,3))



    if batch_size == 1:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # RGB plot
        ax1.imshow(0.2+1.5*(np.dstack([x[0][3], x[0][2], x[0][1]])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]))/
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title('RGB',
                      fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # false color plot
        ax2.imshow(0.2+(np.dstack([x[0][0], x[0][9], x[0][10]])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]))/
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])-
                    np.min([x[0][0].numpy(),
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

        plt.savefig((os.path.split(batch['imgfile'][0])[1]).\
                    replace('.tif', '_eval.png').replace(':', '_'), dpi=200)
        plt.close()
