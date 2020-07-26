# Name: Sanjith Hebbar
# Date: 26-07-2020
# Description: Script containing Dataset classes for Polyp Segmentation

# Import Libraries
import cv2
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Dataset class for training and validation sets
class PolypDataset(Dataset):
    def __init__(self, data_file, mask_file, input_size):
        # Load Images
        with open(data_file) as f:
            self.image_paths = f.readlines()

        # Load Masks
        with open(mask_file) as f:
            self.mask_paths = f.readlines()

        self.image_paths = [x.strip() for x in self.image_paths]
        self.mask_paths = [x.strip() for x in self.mask_paths]
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        img = self.preprocess()(img)

        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2GRAY)
        mask = self.preprocess_mask()(mask)
        return (img, mask, idx)
    
    def preprocess(self):
        """
        Preprocessing function for images. 
        Transforms: 
        Conver to PIL image
        Resize to specified size.
        Convert to tensor
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size, 3),
            transforms.ToTensor()
            ])
    
    def preprocess_mask(self):
        """
        Preprocessing function for masks. 
        Transforms: 
        Conver to PIL image
        Resize to specified size.
        Convert to grayscale.
        Convert to tensor
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size, 3),
            transforms.Grayscale(),
            transforms.ToTensor()
            ])