from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)
from model import UNET
import torch
import os

'''
Use this script to load checkpoints and play with them.
'''

current_dir = os.path.abspath(os.getcwd())
checkpoint_path = os.path.join(current_dir, 'checkpoints/my_checkpoint.pth.tar')
#checkpoint_path = '/home/harsha/PycharmProjects/robotic_surgery_tool_segmentation/checkpoints/my_checkpoint.pth.tar'
checkpoint = torch.load(checkpoint_path)

model = UNET()
