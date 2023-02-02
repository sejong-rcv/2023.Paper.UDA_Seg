import os
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image

class KP_dataset_test(data.Dataset):
    """
    rgb_img : BxCxHxW
    th_img : Bx1xHxW
    label : BxHxW
    """

    def __init__(self, data_dir):

        # assert (split in ['day', 'night']), 'split must be day | night'

        with open(os.path.join(data_dir , 'filenames_KP/val_night_th.txt'), 'r') as file:
            self.names = [name.strip() for idx, name in enumerate(file)] # set00_V000_lwir_I00456.png

        self.data_dir = data_dir
        self.image_folder = 'pseudo_KP/val_night'
        self.label_folder = 'labels/val_night'
        self.flag = '_th'

    def __getitem__(self, index):
        img_name = (self.names[index]).replace('_lwir', '', 1)
        img_name = img_name.split('.png')[0] # set00_V000_I00456

        image_path = os.path.join(self.data_dir, self.image_folder, img_name + self.flag + '.png')
        image = Image.open(image_path).convert('L')
        image = np.asarray(image, dtype=np.float32)  # HxWxC
        image = np.expand_dims(image, axis=2)##################

        image = image.transpose((2, 0, 1)) / 255  # [0,255]->[0,1] CxHxW
        image = torch.tensor(image)

        label_path = os.path.join(self.data_dir, self.label_folder, img_name + '.png')
        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.int64)

        label = torch.tensor(label)
        
        return image, label

    def __len__(self):
        return len(self.names)