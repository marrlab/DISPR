#Written by Dominik Waibel
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
import torch
from skimage.transform import resize, rotate
import random
from skimage.io import imread, imsave
from skimage.transform import resize



def import_image(path_name):
    '''
    This function loads the image from the specified path
    NOTE: The alpha channel is removed (if existing) for consistency
    Args:
        path_name (str): path to image file
    return:
        image_data: numpy array containing the image data in at the given path.
    '''
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[: ,: ,0:3]
    return image_data

def augmentation(obj, img):
    if random.choice([True, False, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 2).copy()
        img = np.flip(img, len(np.shape(img)) - 2).copy()
    if random.choice([True, False, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 3).copy()
        img = np.flip(img, len(np.shape(img)) - 3).copy()
    return obj, img




"""
The data generator will open the 3D segmentation, 2D masks and 2D images for each fold from the directory given the filenames and return a tensor
The 2D mask and the 2D image will be multiplied pixel-wise to remove the background
"""

class SHAPRDataset(Dataset):
    def __init__(self, path, test_flag=True):
        self.test_flag = test_flag
        if self.test_flag==True:
            self.path = path + "/test/"
        else:
            self.path = path + "/train/"
        self.filenames = os.listdir(self.path + "/obj/")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = import_image(os.path.join(self.path, "mask", self.filenames[idx])) / 255.
        bf = import_image(os.path.join(self.path, "image", self.filenames[idx])) / 255.
        msk_bf = np.zeros((2, int(np.shape(img)[0]), int(np.shape(img)[1])))
        msk_bf[0, :, :] = img
        msk_bf[1, :, :] = bf * img
        mask_bf = msk_bf[:, np.newaxis, ...]
        e = np.concatenate((mask_bf, mask_bf), axis=1)
        e = np.concatenate((e, e), axis=1)
        e = np.concatenate((e, e), axis=1)
        e = np.concatenate((e, e), axis=1)
        e = np.concatenate((e, e), axis=1)
        mask_bf = np.concatenate((e, e), axis=1)
        if self.test_flag:
            return torch.from_numpy(mask_bf).float(), self.filenames[idx]
        else:
            obj = import_image(os.path.join(self.path, "obj", self.filenames[idx])) / 255.
            obj = obj[np.newaxis, :, :, :]
            #obj, mask_bf = augmentation(obj, mask_bf)
            return torch.from_numpy(mask_bf).float(), torch.from_numpy(obj).float()
