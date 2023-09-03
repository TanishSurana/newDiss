# this is for optical flow refinement only

# todos: 

# load dataset
# train and save 
# metrics and save

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
#from sklearn.model_selection import train_test_split
import cv2

import os
import os.path
import pandas as pd

import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np

from glob import glob
from tqdm import tqdm


import sys
#sys.path.insert(0, 'vmd_code/code/')
sys.path.append('vmd_code/code')
from losses import lovasz_hinge, binary_xloss
from misc import AvgMeter


root = {
    'mask': 'small_mask',
    'optic': 'small_optical_features',
    'premask': 'small_vmd_mask'

}

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
# return image triple pairs in video and return single image


class CustomDataset_with_mask(data.Dataset):
    def __init__(self, root, traintest, joint_transform=None, img_transform=None, transform=None, initial_mask_percentage = 0.2):
        self.traintest = traintest
        self.mask_folder = root['mask']
        self.opic_folder = root['optic']
        self.premask_folder = root['premask']
        self.initial_mask_percentage = initial_mask_percentage
        
        self.num_video_frame = 0
        self.videoImg_list = self.generateImgFromVideo()
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.transform = transform
        print(len(self.videoImg_list), self.num_video_frame)

    def update_mask_percentage(self, current_epoch, total_epochs):
        max_epochs = total_epochs  # Adjust if needed
        decrease_rate = self.initial_mask_percentage / max_epochs
        updated_mask_percentage = max(0, self.initial_mask_percentage - current_epoch * 2*decrease_rate)
        return updated_mask_percentage

    def generate_random_mask(self, image_size, per):
        mask = torch.rand(image_size) > per
        return mask

    def apply_mask_to_image(self, image_tensor, mask):
        masked_image = image_tensor.clone()
        #print(masked_image.shape, mask.shape)
        masked_image[0, :, :].mul_(mask)
        return masked_image    
    
    def sortImg(self, img_list):
        
        img_int_list = [int(f.split('.')[0]) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]
    
    def generateImgFromVideo(self):

        imgs = []
 
        video_list = listdirs_only(os.path.join(self.mask_folder, self.traintest))

        for video in video_list:


            
            img_list = os.listdir(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            print(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            
            img_list = self.sortImg(img_list)
            print(img_list)
            prev = img_list[0].split('.')[0]
            
            for img in img_list:
                # need optic mask
                # need gt
                # need premask
                imgno = (img.split('.')[0])
                if int(imgno) == 1:
                    continue
                clubbed = [os.path.join(self.mask_folder, self.traintest, video, 'SegmentationClassPNG', img), 
                           os.path.join(self.opic_folder, self.traintest, video, prev+'.jpg'),
                           os.path.join(self.premask_folder, self.traintest, video, prev+'.png')]
                prev = imgno

                imgs.append(clubbed)
            
            self.num_video_frame += len(img_list)

        return imgs
    

    def __len__(self):
        return len(self.videoImg_list)
    
    def __getitem__(self, idx,  current_epoch=None, total_epochs=None):
        gt, optic, pre = self.videoImg_list[idx]  
        gtimage = Image.open(gt).convert("L")
        opticimage = Image.open(optic).convert("RGB")
        preimage = Image.open(pre).convert("L")

        if self.transform is not None:
            gtimage = self.transform(gtimage)
            opticimage = self.transform(opticimage)
            preimage = self.transform(preimage)
        

       
        if self.traintest == 'train':
            if current_epoch is not None and total_epochs is not None:
                mask_percentage = self.update_mask_percentage(current_epoch, total_epochs)
            else:
                mask_percentage = self.initial_mask_percentage

            mask = self.generate_random_mask(gtimage.shape[-2:], mask_percentage)
            gtimage = self.apply_mask_to_image(gtimage, mask)
            #print(mask_percentage, 'mask p')

        gtimage = gtimage.squeeze(0) # need to check if needed coz of mask
        
        return (opticimage, preimage), gtimage

class CustomDataset3(data.Dataset):
    def __init__(self, root, traintest, joint_transform=None, img_transform=None, transform=None, initial_mask_percentage = 0.2):
        self.traintest = traintest
        self.mask_folder = root['mask']
        self.opic_folder = root['optic']
        self.premask_folder = root['premask']
        self.initial_mask_percentage = initial_mask_percentage
        
        self.num_video_frame = 0
        self.videoImg_list = self.generateImgFromVideo()
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.transform = transform
        print(len(self.videoImg_list), self.num_video_frame)

    def update_mask_percentage(self, current_epoch, total_epochs):
        max_epochs = total_epochs  # Adjust if needed
        decrease_rate = self.initial_mask_percentage / max_epochs
        updated_mask_percentage = max(0, self.initial_mask_percentage - current_epoch * decrease_rate)
        return updated_mask_percentage

    def generate_random_mask(self, image_size, per):
        mask = torch.rand(image_size) > per
        return mask

    def apply_mask_to_image(self, image_tensor, mask):
        masked_image = image_tensor.clone()
        #print(masked_image.shape, mask.shape)
        masked_image[0, :, :].mul_(mask)
        return masked_image    
    
    def sortImg(self, img_list):
        
        img_int_list = [int(f.split('.')[0]) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]
    
    def generateImgFromVideo(self):

        imgs = []
 
        video_list = listdirs_only(os.path.join(self.mask_folder, self.traintest))

        for video in video_list:


            
            img_list = os.listdir(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            print(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            
            img_list = self.sortImg(img_list)
            print(img_list)
            prev = img_list[0].split('.')[0]
            
            for img in img_list:
                # need optic mask
                # need gt
                # need premask
                imgno = (img.split('.')[0])
                if int(imgno) == 1:
                    continue
                clubbed = [os.path.join(self.mask_folder, self.traintest, video, 'SegmentationClassPNG', img), 
                           os.path.join(self.opic_folder, self.traintest, video, prev+'.png'),
                           os.path.join(self.premask_folder, self.traintest, video, prev+'.png')]
                prev = imgno

                imgs.append(clubbed)
            
            self.num_video_frame += len(img_list)

        return imgs
    

    def __len__(self):
        return len(self.videoImg_list)
    
    def __getitem__(self, idx,  current_epoch=None, total_epochs=None):
        gt, optic, pre = self.videoImg_list[idx]  
        gtimage = Image.open(gt).convert("L")
        opticimage = Image.open(optic).convert("RGB")
        preimage = Image.open(pre).convert("L")

        if self.transform is not None:
            gtimage = self.transform(gtimage)
            opticimage = self.transform(opticimage)
            preimage = self.transform(preimage)
        


        gtimage = gtimage.squeeze(0) # need to check if needed coz of mask
        
        return (opticimage, preimage), gtimage

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union




class CustomDataset_sam(data.Dataset):
    def __init__(self, root, traintest, joint_transform=None, img_transform=None, transform=None, initial_mask_percentage = 0.2, max_colors = 5):
        self.traintest = traintest
        self.mask_folder = root['mask']
        self.opic_folder = root['optic']
        self.premask_folder = root['premask']
        self.initial_mask_percentage = initial_mask_percentage
        
        self.num_video_frame = 0
        self.videoImg_list = self.generateImgFromVideo()
        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'
        self.transform = transform
        self.max_colors = max_colors
        self.img_size = (416,416)
        print(len(self.videoImg_list), self.num_video_frame)

    def update_mask_percentage(self, current_epoch, total_epochs):
        max_epochs = total_epochs  # Adjust if needed
        decrease_rate = self.initial_mask_percentage / max_epochs
        updated_mask_percentage = max(0, self.initial_mask_percentage - current_epoch * decrease_rate)
        return updated_mask_percentage

    def generate_random_mask(self, image_size, per):
        mask = torch.rand(image_size) > per
        return mask

    def apply_mask_to_image(self, image_tensor, mask):
        masked_image = image_tensor.clone()
        #print(masked_image.shape, mask.shape)
        masked_image[0, :, :].mul_(mask)
        return masked_image    
    
    def sortImg(self, img_list):
        
        img_int_list = [int(f.split('.')[0]) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]
    
    def generateImgFromVideo(self):

        imgs = []
 
        video_list = listdirs_only(os.path.join(self.mask_folder, self.traintest))

        for video in video_list:


            
            img_list = os.listdir(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            print(os.path.join(self.mask_folder,self.traintest ,video, 'SegmentationClassPNG'))
            
            img_list = self.sortImg(img_list)
            print(img_list)
            prev = img_list[0].split('.')[0]
            
            for img in img_list:
                # need optic mask
                # need gt
                # need premask
                imgno = (img.split('.')[0])
                if int(imgno) == 1:
                    continue
                clubbed = [os.path.join(self.mask_folder, self.traintest, video, 'SegmentationClassPNG', img), 
                           os.path.join(self.opic_folder, self.traintest, video, prev+'.png'),
                           os.path.join(self.premask_folder, self.traintest, video, prev+'.png')]

                prev = imgno

                imgs.append(clubbed)
            
            self.num_video_frame += len(img_list)

        return imgs

    def __len__(self):
        return len(self.videoImg_list)
    
    def __getitem__(self, idx,  current_epoch=None, total_epochs=None):
        gt, optic, pre = self.videoImg_list[idx]  



    
        mask = cv2.imread(pre, 0) 
        multi_class = cv2.imread(optic)
        ground_truth = cv2.imread(gt, 0) 

        if self.transform is not None:
            mask = cv2.resize(mask, self.img_size) / 255.0
            multi_class = cv2.resize(multi_class, self.img_size)
            ground_truth = cv2.resize(ground_truth, self.img_size) / 255.0

        unique_colors = set(tuple(v) for m2d in multi_class for v in m2d)
        color_iou_scores = []

        for color in unique_colors:
            color_mask = ((multi_class == color) * 1).all(axis=-1)
            iou = compute_iou(color_mask, mask)
            color_iou_scores.append((color, iou))

        # Sort by IoU and take the top colors
        sorted_colors = [item[0] for item in sorted(color_iou_scores, key=lambda x: x[1], reverse=True)[:self.max_colors]]

        multi_class_channels = []
        for color in sorted_colors:
            channel = ((multi_class == color) * 1).all(axis=-1) * 255
            multi_class_channels.append(channel)

        # If there are fewer unique colors than MAX_COLORS, append all-zero channels
        while len(multi_class_channels) < self.max_colors:
            multi_class_channels.append(np.zeros_like(mask))

        combined = [mask] + multi_class_channels
        combined = np.stack(combined, axis=0) / 255.0

        return torch.tensor(combined, dtype=torch.float32), torch.tensor(ground_truth, dtype=torch.float32)
    
                
        gtimage = Image.open(gt).convert("L")
        opticimage = Image.open(optic).convert("L")
        multi_class = cv2.imread(optic)
        preimage = Image.open(pre).convert("L")

        if self.transform is not None:
            gtimage = self.transform(gtimage)
            opticimage = self.transform(opticimage)
            preimage = self.transform(preimage)
        


        gtimage = gtimage.squeeze(0) # need to check if needed coz of mask
        
        return (opticimage, preimage), gtimage
       
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Add more layers
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        encoded = self.encoder(combined_input)
        decoded = self.decoder(encoded)
        return decoded

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        encoded = self.encoder(combined_input)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_2d(nn.Module):
    def __init__(self):
        super(Autoencoder_2d, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        encoded = self.encoder(combined_input)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_2d_drop(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Autoencoder_2d_drop, self).__init__()
        
        # Dropout rate
        self.dropout = nn.Dropout(dropout_rate)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            self.dropout,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            self.dropout,
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        encoded = self.encoder(combined_input)
        decoded = self.decoder(encoded)
        return decoded



class UNet_old(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        # Create a combined input by concatenating along the channel dimension
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        
        # Apply the encoder
        encoded1 = self.encoder[0:4](combined_input)
        encoded2 = self.encoder[4:8](encoded1)
        encoded3 = self.encoder[8:12](encoded2)
        
        # Apply the decoder with skip connections
        decoded1 = self.decoder[0:2](encoded3)
        decoded2 = self.decoder[2:4](decoded1)
        decoded_output = self.decoder[4:](decoded2)
        
        return decoded_output


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input_image1, input_image2):
        # Create a combined input by concatenating along the channel dimension
        combined_input = torch.cat((input_image1, input_image2), dim=1)
        
        # Apply the encoder
        encoded1 = self.encoder[0:8](combined_input)
        encoded2 = self.encoder[8:16](encoded1)
        encoded3 = self.encoder[16:24](encoded2)
        encoded4 = self.encoder[24:](encoded3)
        
        # Apply the decoder with skip connections
        decoded1 = self.decoder[0:4](encoded4)
        decoded2 = self.decoder[4:8](decoded1)
        decoded3 = self.decoder[8:12](decoded2)
        decoded_output = self.decoder[12:](decoded3)
        
        return decoded_output

import torch.nn as nn
import torch.nn.functional as F

class Autoencoder_sam_multicolor(nn.Module):
    def __init__(self, max_colors=5, dropout_prob=0.5):
        super(Autoencoder_sam_multicolor, self).__init__()
        self.input_channels = 1 + max_colors  # 1 for binary mask, rest for multi-class channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Defines data transformations
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# root = {
#     'mask': 'small_mask',
#     'optic': 'small_optical_features',
#     'premask': 'small_vmd_mask'
# }
root = {
    'mask': 'fulldataset',
    'optic': 'sam_optic_mask',
    'premask': 'vmd_masks'
}


# Create dataset instance 
traindataset = CustomDataset_sam(root, traintest='train', transform=transform)


# Create DataLoader
batch_size = 32
train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=4)
testdataset = CustomDataset_sam(root, traintest='test', transform=transform)
val_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=4)


# # Create the autoencoder model
autoencoder = Autoencoder_sam_multicolor(max_colors = 5, dropout_prob = 0.3)
autoencoder.to(device)

# # Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary masks
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# # Assuming you have loaded your input images and segmentation masks
# # into input_images and segmentation_masks arrays

# # Split data into train and validation sets
# input_images_train, input_images_val, masks_train, masks_val = train_test_split(
#     input_images, masks, test_size=0.2, random_state=42
# )



# ... (previous code for model definition, data loading, etc.)

def iou_score(pred, target):
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def accuracy(pred, target):
    return (pred == target).sum().true_divide(pred.numel())


def mae(pred, target):
    return torch.abs(pred - target).mean()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary masks
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Lists to store metrics for each epoch
train_losses = []
train_ious = []
train_accs = []
train_maes = []
val_losses = []
val_ious = []
val_accs = []
val_maes = []
metrics_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Train IoU', 'Train Acc', 'Train MAE', 'Val Loss', 'Val IoU', 'Val Acc', 'Val MAE'])



# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    autoencoder.train()
    total_iou = 0.0
    total_acc = 0.0
    total_mae = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    loss_record = AvgMeter()
    
     
    for batch_combined, batch_ground_truth in progress_bar:
        train_loader.current_epoch = epoch
        train_loader.total_epochs = num_epochs
        batch_combined = batch_combined.to(device)
        batch_ground_truth = batch_ground_truth.to(device)
        batch_ground_truth = batch_ground_truth.unsqueeze(1)  # Add channel dimension for ground truth

        optimizer.zero_grad()
        
        input_image1 = batch_combined[:, 0:1, :, :]  # Binary mask image
        input_image2 = batch_combined[:, 1:, :, :]  # Multi-channel image with top colors

        combined_input = torch.cat([input_image1, input_image2], dim=1)
        reconstructions = autoencoder(combined_input)
        loss = lovasz_hinge(reconstructions, batch_ground_truth, per_image = False)
        loss.backward()
        optimizer.step()

        # Compute IoU, Accuracy, and MAE on GPU
        pred_masks = (reconstructions > 0.5).float()  # Convert to binary mask
        batch_iou = iou_score(pred_masks, batch_ground_truth)
        batch_acc = accuracy(pred_masks, batch_ground_truth)
        batch_mae = mae(reconstructions, batch_ground_truth)

        total_iou += batch_iou.item()
        total_acc += batch_acc.item()
        total_mae += batch_mae.item()

    autoencoder.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_acc = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for batch_combined, batch_ground_truth in val_loader:
            batch_combined = batch_combined.to(device)
            batch_ground_truth = batch_ground_truth.to(device)
            batch_ground_truth = batch_ground_truth.unsqueeze(1)  # Add channel dimension for masks

            input_image1 = batch_combined[:, 0:1, :, :]  # Binary mask image
            input_image2 = batch_combined[:, 1:, :, :]  # Multi-channel image with top colors
            
            combined_input = torch.cat([input_image1, input_image2], dim=1)
            reconstructions = autoencoder(combined_input)
            val_loss += criterion(reconstructions, batch_ground_truth).item()

            pred_masks = (reconstructions > 0.5).float()
            batch_iou = iou_score(pred_masks, batch_ground_truth)
            batch_acc = accuracy(pred_masks, batch_ground_truth)
            batch_mae = mae(reconstructions, batch_ground_truth)

            val_iou += batch_iou.item()
            val_acc += batch_acc.item()
            val_mae += batch_mae.item()

    train_iou = total_iou / len(train_loader)
    train_acc = total_acc / len(train_loader)
    train_mae = total_mae / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)

    # Save metrics for the current epoch
    train_losses.append(loss.item())
    train_ious.append(train_iou)
    train_accs.append(train_acc)
    train_maes.append(train_mae)
    val_losses.append(avg_val_loss)
    val_ious.append(avg_val_iou)
    val_accs.append(avg_val_acc)
    val_maes.append(avg_val_mae)

    # Save model checkpoint
    checkpoint_path = f"base_optic\\checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_ious': train_ious,
        'train_accs': train_accs,
        'train_maes': train_maes,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_accs': val_accs,
        'val_maes': val_maes
    }, checkpoint_path)
    
    epoch_metrics = {
    'Epoch': epoch+1,
    'Train Loss': loss.item(),
    'Train IoU': train_iou,
    'Train Acc': train_acc,
    'Train MAE': train_mae,
    'Val Loss': avg_val_loss,
    'Val IoU': avg_val_iou,
    'Val Acc': avg_val_acc,
    'Val MAE': avg_val_mae
    }
    metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {loss:.4f} - Train IoU: {train_iou:.4f} - Train Acc: {train_acc:.4f} - Train MAE: {train_mae:.4f} - Val Loss: {avg_val_loss:.4f} - Val IoU: {avg_val_iou:.4f} - Val Acc: {avg_val_acc:.4f} - Val MAE: {avg_val_mae:.4f}")

with pd.ExcelWriter('training_metrics.xlsx') as writer:
    metrics_df.to_excel(writer, index=False)




