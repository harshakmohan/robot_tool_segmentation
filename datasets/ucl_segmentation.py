import numpy as np
import torch
import os
import os.path as osp
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate
import torch.utils.data as data
import torchvision.transforms as T
import random
import cv2
import scipy.io
from PIL import Image

def readKinematics(path):
    kinematics = scipy.io.loadmat(path)
    result = []
    for k in kinematics.keys():
        if "value" in k:
            result.append(kinematics[k][:,4::5])
    result = np.concatenate(result, axis=0).astype(np.float32)
    return result.T

class UCLSegmentation(data.Dataset):
    def __init__(self, folder_path, video_paths, series_length=2, image_transforms=None, gt_transforms=None, kinematics_transforms=None):
        self.folder_path = folder_path
        self.video_paths = [osp.join(folder_path, p) for p in video_paths]
        self.image_paths = []
        self.gt_paths = []
        self.kinematics = []
        self.image_transforms = image_transforms
        self.gt_transforms = gt_transforms
        self.kinematics_transforms = kinematics_transforms
        self.series_length = series_length
        for p in self.video_paths:
            kinematics = readKinematics(osp.join(p, "kinematic.mat"))
            self.kinematics.append(kinematics)
            images = os.listdir(osp.join(p, "images"))
            for i in range(300):
                if i < 10:
                    name = "00" + str(i) + ".png"
                elif i < 100:
                    name = "0" + str(i) + ".png"
                else:
                    name = str(i) + ".png"
                self.image_paths.append(osp.join(osp.join(p, "images"),name))
                self.gt_paths.append(osp.join(osp.join(p, "ground_truth"),name))
            gts = os.listdir(osp.join(p, "ground_truth"))
        self.kinematics = np.concatenate(self.kinematics, axis=0)
 #       print(len(self.image_paths))
 #       print(len(self.gt_paths))
 #       print(self.kinematics.shape)

    def __len__(self):
        return len(self.image_paths) - self.series_length + 1

    def __getitem__(self, idx: int):
        images = []
        gts = []
        kinematics_s = []
        for i in range(self.series_length):
            #print(self.image_paths[idx+i], self.gt_paths[idx+i])
            image = np.array(Image.open(self.image_paths[idx+i])).astype(np.float32)
            gt = (np.array(Image.open(self.gt_paths[idx+i]))/255)[:,:,0].astype(np.float32)
            kinematics = self.kinematics[idx+i]
            if self.image_transforms is None:
                image = T.ToTensor()(image)
            else:
                image = self.image_transforms(image)
            if self.gt_transforms is None:
                gt = T.ToTensor()(gt)
            else:
                gt = self.gt_transforms(gt)
            if self.kinematics_transforms is None:
                kinematics = torch.tensor(kinematics)
            else:
                kinematics = self.kinematics_transforms(kinematics)
            images.append(image)
            gts.append(gt)
            kinematics_s.append(kinematics)
        if self.series_length == 1:
            images = images[0]
            gts = gts[0]
            kinematics_s = kinematics_s[0]
        else:
            images = torch.stack(images)
            gts = torch.stack(gts)
            kinematics_s = torch.stack(kinematics_s)
        return images, gts, kinematics_s
