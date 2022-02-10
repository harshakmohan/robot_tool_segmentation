import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy


# def readKinematics(path):
#     kinematics = scipy.io.loadmat(path)
#     result = []
#     for k in kinematics.keys():
#         if "value" in k:
#             result.append(kinematics[k][:,4::5])
#     result = np.concatenate(result, axis=0).astype(np.float32)
#     return result.T

# TODO: Modify this to read the data in the directory format provided by UCL
# Take in a list of the Video_## folders that you want to use for training, validation, and testing.
class UCLSegmentation(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return image, mask


