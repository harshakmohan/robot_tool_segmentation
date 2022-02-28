import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import device
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)
from utils import DiceLoss2D
import os

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 538  # 1280 originally
IMAGE_WIDTH = 701  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

# New Path for entire UCL dataset
current_dir = os.path.abspath(os.getcwd())
top_data_dir = os.path.join(current_dir, 'data')
print('top_data_dir:', top_data_dir)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            print('loss = ', loss)
            print("loss.item() = ", loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():

    model = UNET(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = DiceLoss2D() # 2D Dice Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(data_dir=top_data_dir,
                                           batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS,
                                           pin_memory=PIN_MEMORY)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(type(train_loader))
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, folder='saved_images/', device=DEVICE)


if __name__ == '__main__':
    main()
