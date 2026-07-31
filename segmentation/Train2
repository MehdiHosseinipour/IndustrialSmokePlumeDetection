# ==========================================================
# Copyright (C) 2020 Michael Mommert
# IndustrialSmokePlumeDetection
#
# Modified:
# - Epoch = 300
# - Batch size = 16
# - Learning rate = 0.01
# - Momentum = 0.9
# - SGD Optimizer
# - BCE + Dice Loss
# - Threshold = 0.5
# - Early stopping
# - Best model based on Validation IoU
# ==========================================================


import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm.autonotebook import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score

from copy import deepcopy
import argparse


from model_unet import *
from data import create_dataset


print("Running on:", device)



# ==========================================================
# Dice Loss
# ==========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):

        super(DiceLoss, self).__init__()

        self.smooth = smooth



    def forward(self, inputs, targets):

        # Convert logits to probabilities
        inputs = torch.sigmoid(inputs)


        inputs = inputs.contiguous()
        targets = targets.contiguous()


        intersection = (
            inputs * targets
        ).sum(dim=(2,3))


        dice = (
            2. * intersection + self.smooth
        ) / (
            inputs.sum(dim=(2,3))
            +
            targets.sum(dim=(2,3))
            +
            self.smooth
        )


        return 1 - dice.mean()



# ==========================================================
# Combined BCE + Dice Loss
# ==========================================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()



def combined_loss(pred, target):

    bce = bce_loss(
        pred,
        target
    )


    dice = dice_loss(
        pred,
        target
    )


    return bce + dice



# ==========================================================
# IoU Calculation Function
# ==========================================================

def calculate_iou(y_true, y_pred):

    ious = []


    for i in range(len(y_true)):

        true = y_true[i].flatten()

        pred = y_pred[i].flatten()


        # Ignore empty images

        if (
            np.sum(true) != 0
            and
            np.sum(pred) != 0
        ):

            iou = jaccard_score(
                true,
                pred
            )

            ious.append(iou)



    if len(ious) == 0:

        return 0


    return np.mean(ious)




# ==========================================================
# Training Function
# ==========================================================

def train_model(
        model,
        epochs,
        opt,
        loss,
        batch_size):



    # ------------------------------------------------------
    # Best model tracking
    # ------------------------------------------------------

    best_val_iou = 0.0

    best_model_wts = None



    # ------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------

    patience = 30

    counter = 0



    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    data_train = create_dataset(

        datadir='./train',

        seglabeldir='./segmentation_labels',

        mult=1
    )



    data_val = create_dataset(

        datadir='./val',

        seglabeldir='./segmentation_labels',

        mult=1
    )



    # ------------------------------------------------------
    # Sampling
    # ------------------------------------------------------

    train_sampler = RandomSampler(

        data_train,

        replacement=True,

        num_samples=int(
            2 * len(data_train) / 3
        )
    )



    val_sampler = RandomSampler(

        data_val,

        replacement=True,

        num_samples=int(
            2 * len(data_val) / 3
        )
    )



    # ------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------

    train_dl = DataLoader(

        data_train,

        batch_size=batch_size,

        num_workers=6,

        pin_memory=True,

        sampler=train_sampler
    )



    val_dl = DataLoader(

        data_val,

        batch_size=batch_size,

        num_workers=6,

        pin_memory=True,

        sampler=val_sampler
    )




    # ======================================================
    # Epoch Loop
    # ======================================================


    for epoch in range(epochs):


        print(
            f"\nEpoch {epoch+1}/{epochs}"
        )


        # ==================================================
        # Training
        # ==================================================

        model.train()



        train_loss_total = 0

        train_acc_total = 0

        train_preds = []

        train_targets = []



        progress = tqdm(

            enumerate(train_dl),

            total=len(train_dl),

            desc="Training"
        )



        for i, batch in progress:


            x = batch['img'].float().to(device)

            y = batch['fpt'].float().to(device)



            # Forward

            output = model(x)



            # ------------------------------
            # Threshold = 0.5
            # ------------------------------

            output_prob = torch.sigmoid(output)



            output_binary = (

                output_prob

                .detach()

                .cpu()

                .numpy()

                >=0.5

            ).astype(int)



            y_numpy = (

                y.detach()

                .cpu()

                .numpy()

            )



            train_preds.extend(
                output_binary[:,0]
            )


            train_targets.extend(
                y_numpy
            )



            # Image level accuracy

            y_bin = np.array(

                np.sum(y_numpy,axis=(1,2))

                !=0

            ).astype(int)



            pred_bin = np.array(

                np.sum(
                    output_binary,
                    axis=(1,2,3)
                )

                !=0

            ).astype(int)



            train_acc_total += accuracy_score(

                y_bin,

                pred_bin

            )



            # Loss

            loss_epoch = loss(

                output,

                y.unsqueeze(dim=1)

            )


            train_loss_total += loss_epoch.item()



            # Backpropagation

            opt.zero_grad()


            loss_epoch.backward()


            opt.step()



            progress.set_description(

                "Train Loss %.4f"

                %
                (
                    train_loss_total/(i+1)
                )
            )
        # ==================================================
        # End of Training Epoch
        # ==================================================

        train_iou = calculate_iou(
            train_targets,
            train_preds
        )


        train_loss = (
            train_loss_total /
            (i + 1)
        )


        train_acc = (
            train_acc_total /
            (i + 1)
        )



        # TensorBoard logging

        writer.add_scalar(
            "training loss",
            train_loss,
            epoch
        )


        writer.add_scalar(
            "training IoU",
            train_iou,
            epoch
        )


        writer.add_scalar(
            "training accuracy",
            train_acc,
            epoch
        )



        torch.cuda.empty_cache()



        # ==================================================
        # Validation
        # ==================================================

        model.eval()


        val_loss_total = 0

        val_acc_total = 0


        val_preds = []

        val_targets = []



        with torch.no_grad():


            progress = tqdm(

                enumerate(val_dl),

                total=len(val_dl),

                desc="Validation"

            )



            for j, batch in progress:


                x = batch['img'].float().to(device)


                y = batch['fpt'].float().to(device)



                # Forward

                output = model(x)



                # Loss

                loss_epoch = loss(

                    output,

                    y.unsqueeze(dim=1)

                )


                val_loss_total += loss_epoch.item()



                # ------------------------------
                # Threshold = 0.5
                # ------------------------------

                output_prob = torch.sigmoid(output)



                output_binary = (

                    output_prob

                    .cpu()

                    .numpy()

                    >=0.5

                ).astype(int)



                y_numpy = (

                    y.cpu()

                    .numpy()

                )



                val_preds.extend(

                    output_binary[:,0]

                )


                val_targets.extend(

                    y_numpy

                )



                # Image classification accuracy

                y_bin = np.array(

                    np.sum(
                        y_numpy,
                        axis=(1,2)
                    )

                    !=0

                ).astype(int)



                pred_bin = np.array(

                    np.sum(
                        output_binary,
                        axis=(1,2,3)
                    )

                    !=0

                ).astype(int)



                val_acc_total += accuracy_score(

                    y_bin,

                    pred_bin

                )



                progress.set_description(

                    "Val Loss %.4f"

                    %
                    (
                        val_loss_total/(j+1)
                    )

                )



        # --------------------------------------------------
        # Validation Metrics
        # --------------------------------------------------


        val_loss = (

            val_loss_total /

            (j + 1)

        )


        val_iou = calculate_iou(

            val_targets,

            val_preds

        )


        val_acc = (

            val_acc_total /

            (j + 1)

        )



        # TensorBoard

        writer.add_scalar(

            "validation loss",

            val_loss,

            epoch

        )


        writer.add_scalar(

            "validation IoU",

            val_iou,

            epoch

        )


        writer.add_scalar(

            "validation accuracy",

            val_acc,

            epoch

        )



        # --------------------------------------------------
        # Print Epoch Results
        # --------------------------------------------------

        print(

            "\nEpoch {:03d}: "

            "Train Loss={:.4f} | "

            "Val Loss={:.4f} | "

            "Train IoU={:.4f} | "

            "Val IoU={:.4f} | "

            "Train Acc={:.4f} | "

            "Val Acc={:.4f}"

            .format(

                epoch + 1,

                train_loss,

                val_loss,

                train_iou,

                val_iou,

                train_acc,

                val_acc

            )

        )



        # ==================================================
        # Save Best Model Based on Validation IoU
        # ==================================================


        if val_iou > best_val_iou:


            print(

                "✓ Validation IoU improved - Saving model"

            )


            best_val_iou = val_iou



            # Important:
            # deepcopy prevents weight overwrite

            best_model_wts = deepcopy(

                model.state_dict()

            )


            counter = 0



        else:


            counter += 1


            print(

                f"No improvement "
                f"{counter}/{patience}"

            )



        # ==================================================
        # Save Epoch Checkpoint
        # ==================================================


        torch.save(

            model.state_dict(),

            'ep{:03d}_lr{:.3f}_bs{:02d}.model'

            .format(

                epoch + 1,

                args.lr,

                args.bs

            )

        )



        # ==================================================
        # Learning Rate Scheduler
        # ==================================================

        scheduler.step(

            val_loss

        )



        torch.cuda.empty_cache()



        # ==================================================
        # Early Stopping
        # ==================================================

        if counter >= patience:


            print(

                "\nEarly stopping activated!"

            )


            print(

                "Best Validation IoU:",

                best_val_iou

            )


            break



    # ======================================================
    # Save Final Best Model
    # ======================================================


    if best_model_wts is not None:


        torch.save(

            best_model_wts,

            'segmentation_best.model'

        )


        print(

            "\nBest model saved:"
            " segmentation_best.model"

        )



    return model
# ==========================================================
# Argument Parser
# ==========================================================


parser = argparse.ArgumentParser()



parser.add_argument(
    '-ep',
    type=int,
    default=300,
    help='Number of epochs'
)



parser.add_argument(
    '-bs',
    type=int,
    nargs='?',
    default=16,
    help='Batch size'
)



parser.add_argument(
    '-lr',
    type=float,
    nargs='?',
    default=0.01,
    help='Learning rate'
)



parser.add_argument(
    '-mo',
    type=float,
    nargs='?',
    default=0.9,
    help='Momentum'
)



args = parser.parse_args()



# ==========================================================
# TensorBoard Writer
# ==========================================================


writer = SummaryWriter(

    'runs/'
    +
    "ep{:03d}_lr{:.3f}_bs{:02d}_mo{:.1f}"

    .format(

        args.ep,

        args.lr,

        args.bs,

        args.mo

    )

)



# ==========================================================
# Loss Function
# ==========================================================


loss = combined_loss



# ==========================================================
# Optimizer
# SGD
# ==========================================================


opt = optim.SGD(

    model.parameters(),

    lr=args.lr,

    momentum=args.mo

)



# ==========================================================
# Learning Rate Scheduler
# ==========================================================


scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    opt,

    mode='min',

    factor=0.5,

    threshold=1e-4,

    patience=10,

    min_lr=1e-6

)



# ==========================================================
# Start Training
# ==========================================================


train_model(

    model,

    args.ep,

    opt,

    loss,

    args.bs

)



# Close TensorBoard

writer.close()
