import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import scipy
import torchvision.transforms.functional as TF


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

        #image = TF.pil_to_tensor(Image.open(img_path).convert('RGB'))
        #mask = TF.pil_to_tensor(Image.open(mask_path).convert('L'))

        image = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))/255.0).float()
        image = torch.permute(image, (2, 0, 1))
        mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L'), dtype=np.float32)/255.0).float()

        #print('image dim: ', image.size())
        #print('mask dim: ', mask.size())

        #mask[mask == 255.0] = 1.0

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return image, mask

class UCLSegmentationAll(Dataset):
    def __init__(self, folder_path, video_paths = ['Video_01', 'Video_02', 'Video_03'], train_list=['01', '02', '03'], val_list=['04']):
        self.folder_path = folder_path
        self.video_paths = [os.path.join(folder_path, p) for p in video_paths] # List of paths to the Video_## folders

        self.image_paths = []
        self.gt_paths = []

        for p in self.video_paths:

            for i in range(300):
                if i < 10:
                    name = "00" + str(i) + ".png"
                elif i < 100:
                    name = "0" + str(i) + ".png"
                else:
                    name = str(i) + ".png"
                self.image_paths.append(os.path.join(os.path.join(p, 'images'), name))
                self.gt_paths.append(os.path.join(os.path.join(p, 'images'), name))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))/255.0).float()
        image = torch.permute(image, (2, 0, 1))
        mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L'), dtype=np.float32)/255.0).float()

        return image, mask

