from torch.utils.data import Dataset
# import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from CNN_model.constants import *

class CustomImageLoadingDataset(Dataset):
    
    def __init__(self, csv, transform):
        self.csv = csv
        self.img_paths = self.csv.loc[:, 'img'].values
        self.img_labels = self.csv.loc[:, 'process'].values
        self.transform = transform
        
    def __len__(self):
        return self.csv.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        new_img_path = f"{IMGS_DIR}/{img_path}"

        label = self.img_labels[idx]
        
        img = plt.imread(new_img_path)
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
        img = torch.div(img, 255)
        
            
        return img, label, new_img_path