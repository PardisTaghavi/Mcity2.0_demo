#cityscapes.py


import random
import importlib
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from labels import labels
import matplotlib.pyplot as plt
import cv2
import numpy as np





#root_dir = /home/avalocal/mcity/data


'''

----data
    ----loop1
        ----images        : image_{}.png
        ----segmentations : segmentation_{}.png
        ----depths        : depth_{}.png
    ----loop2
        ----images
        ----segmentations
        ----depths
    ----loop3
        ----images
        ----segmentations
        ----depths

'''

def modify_label(label):
    '''
    label is a h x w numpy array
    '''
    # 0: road
    # 1: sidewalk
    # 2: building
    # 3: wall
    # 4: fence


class mcity(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
    
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        
         
        #root_dir + loop1/ loop2/ loop3
        self.dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.dirs.sort()
        #../data/loop1/ + ../data/loop2/ + ../data/loop3/


        #loop -> images, segmentationsPre, depthsPre
        self.image_files = []
        for d in self.dirs:
            image_files = [os.path.join(self.root_dir, d, 'images', f) for f in os.listdir(os.path.join(self.root_dir, d, 'images')) if f.endswith('.png')]
            image_files.sort()
            self.image_files.extend(image_files)

        
        self.label_files = []
        for d in self.dirs:
            label_files = [os.path.join(self.root_dir, d, 'segmentationsPre', f) for f in os.listdir(os.path.join(self.root_dir, d, 'segmentationsPre')) if f.endswith('.png')]
            label_files.sort()
            self.label_files.extend(label_files)

        self.depth_files = []
        for d in self.dirs:
            depth_files = [os.path.join(self.root_dir, d, 'depths', f) for f in os.listdir(os.path.join(self.root_dir, d, 'depths')) if f.endswith('.png')]
            depth_files.sort()
            self.depth_files.extend(depth_files)

    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, idx):
        # print(len(self.image_files), len(self.label_files), len(self.depth_files), "lengths") 9243



        image = cv2.cvtColor(cv2.imread(self.image_files[idx]), cv2.COLOR_BGR2RGB) 

        #seg map
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED) #bgr
        label = label[:, :, 2] #extract the red channel which is class
        label[label> 19] = 19 #unknown class

        #depth map
        depth = cv2.imread(self.depth_files[idx], cv2.COLOR_BGR2RGB)
        R, G, B = depth[:, :, 0].astype(np.float32), depth[:, :, 1].astype(np.float32), depth[:, :, 2].astype(np.float32)
        depth = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth = 100 * depth #depth in meters between 0 and 1000
        depth[depth > 100] = 100
        depth[depth < 0] = 0

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, masks=[label, depth])
            image = augmented['image']
            label, depth = augmented['masks']

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # C, H, W
        image = image / 255.0 # Normalize image
        label = torch.tensor(label)
        depth = torch.tensor(depth, dtype=torch.float32)
        
        return {'image': image, 'seg': label, 'depth': depth}
    
        

### Example usage for testing purposes
if __name__ == "__main__":           

    dataset = mcity(root_dir='/home/avalocal/mcity/data', transform=None)
    idx = 3800
    sample = dataset[idx]
    img, label, depth = sample['image'], sample['seg'], sample['depth']
    # 3, h, w | h, w | h, w
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img.permute(1, 2, 0))
    ax1.set_title('Image')
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(label)
    ax2.set_title('Label')
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(depth)
    ax3.set_title('Depth')
    plt.show()







    
        
        

