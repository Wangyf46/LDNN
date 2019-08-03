import os
import sys
sys.path.insert(0, '/home/wangyf/codes/LDNN')
import ipdb
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.utils  import *

class DIV2K(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.args = args
        if args.period == 'train':
            self.img_dir = os.path.join(args.path, 'DIV2K_train_HR_aug')
            self.name_list = os.listdir(self.img_dir)
        else:
            self.img_dir = os.path.join(args.path, 'DIV2K_valid_HR_aug')
            self.name_list = os.listdir(self.img_dir)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        '''
         numpy image: H X W X C
         PIL image: C X H X W
         torch image: C X H X W
         img_PIL = transforms.ToPILImage()(img).convert('RGB')
        '''
        name = self.name_list[idx]
        img_file =  os.path.join(self.img_dir, name)
        img_RGB = cv2.imread(img_file)                               # numpy-RGB-uint8-[0-255]

        BL = LocalDimming(img_RGB, self.args)                        # numpy-float32-[0.0-255.0]
        LD = get_LD(BL, self.args.BMA)                               # numpy-float32-[0.0-255.0]

        img_transform = torch.from_numpy(img_RGB.transpose((2,0,1)).astype(np.float32) / 255.0) # torch.float32-[0.0-1.0]
        LD_transform = torch.from_numpy(LD)                                                     # torch.float32-[0.0-255.0]
        return  img_transform, LD_transform, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LDNN Train")
    parser.add_argument('--period', default='train')
    parser.add_argument('--path', default='/home/wangyf/datasets')
    parser.add_argument('--epochs', default=30)
    parser.add_argument('--bz', default=4)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--pre', default=False, help='load file model')
    parser.add_argument('--base_size', type=int, default=[1080, 1920])
    parser.add_argument('--block_size', type=int, default=[9, 16])
    parser.add_argument('--backlight', type=str, default='LUT')
    parser.add_argument('--BMA', type=int, default=4)
    args = parser.parse_args()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor()])
                                    # transforms.Normalize(mean, std)])
    listDataset = DIV2K(args, transform)

    train_loader = DataLoader(listDataset,
                              batch_size=args.bz,
                              shuffle=True)
    for i, (imgs, BLs, name) in enumerate(train_loader):
        ipdb.set_trace()
