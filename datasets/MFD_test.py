import os
import torch
from torch.utils import data
from torchvision import transforms, utils

import numpy as np
from PIL import Image

class MF_dataset_test(data.Dataset):
    """
    rgb_img : BxCxHxW
    th_img : Bx1xHxW
    label : BxHxW
    """

    def __init__(self, data_dir):

        # assert (split in ['day', 'night']), 'split must be day | night'

        with open(os.path.join(data_dir , 'test.txt'), 'r') as file:
            self.names = [name.strip() for idx, name in enumerate(file)]

        self.data_dir = data_dir
        self.image_folder = 'images'
        self.label_folder = 'labels'
        # # self.pseudo_folder = pseudo_folder
        # self.pseudo_folder = 'pseudo_all'
        # self.fake_folder = fake_folder
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        img_name = self.names[index]

        image_path = os.path.join(self.data_dir, self.image_folder, img_name + '.png')
        image = Image.open(image_path)
        image = np.asarray(image, dtype=np.float32)  # HxWxC


        image = image.transpose((2, 0, 1)) / 255  # [0,255]->[0,1] CxHxW
        image = torch.tensor(image)

        th_image = image[3]
        th_image = th_image.unsqueeze(0)
        ###################################

        label_path = os.path.join(self.data_dir, self.label_folder, img_name + '.png')
        label = Image.open(label_path)
        label = np.asarray(label, dtype=np.int64)

        label = torch.tensor(label)
        
        return th_image, label

    def __len__(self):
        return len(self.names)