import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class HairSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.classes = [0, 1, 2]

    def __getitem__(self, i):
        id = self.ids[i]
        
        image_path = os.path.join(self.images_dir, id + ".jpg") 
        mask_path = os.path.join(self.masks_dir, id + ".png")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, 0) 
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return torch.from_numpy(image), torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.ids)
