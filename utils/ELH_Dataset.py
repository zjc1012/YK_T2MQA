import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image
import cv2
from transformers import SiglipProcessor

class ELH_Dataset(data.Dataset):
    def __init__(self, data_dir_2d, data_dir_2d_dev, datainfo_path,
                 crop_size=512, img_length_read=4, is_train=True):
        super(ELH_Dataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header=0, index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo['Image'] 
        self.ply_mos = dataInfo['Score']
        self.crop_size = crop_size
        self.data_dir_2d = data_dir_2d
        self.data_dir_2d_dev = data_dir_2d_dev
        self.img_length_read = img_length_read
        self.length = len(self.ply_name)
        self.is_train = is_train
        self.processor = SiglipProcessor.from_pretrained("google/siglip2-base-patch16-512", use_fast=False)

    def __len__(self):
        return self.length
    
    def crop_white_background(self, img):
        img_np = np.array(img)
        white_threshold = 245
        non_white_pixels = np.where(np.all(img_np < white_threshold, axis=2))
        
        if len(non_white_pixels[0]) == 0:
            return img
        y_min, y_max = non_white_pixels[0].min(), non_white_pixels[0].max()
        x_min, x_max = non_white_pixels[1].min(), non_white_pixels[1].max()
        
        object_rect = img_np[y_min:y_max+1, x_min:x_max+1]
        obj_h, obj_w = object_rect.shape[:2]
        target_size = max(obj_h, obj_w)
        pad_top = (target_size - obj_h) // 2
        pad_bottom = target_size - obj_h - pad_top
        pad_left = (target_size - obj_w) // 2
        pad_right = target_size - obj_w - pad_left
        
        padded_object = cv2.copyMakeBorder(
            object_rect,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        cropped_img = Image.fromarray(padded_object)
        
        return cropped_img

    def load_and_process_image(self, image_path):
        read_frame = Image.open(image_path).convert('RGB')
        read_frame = self.crop_white_background(read_frame)
        read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
        return read_frame
    
    def __getitem__(self, idx):
        img_name = self.ply_name.iloc[idx].strip('"')
        y_mos = self.ply_mos.iloc[idx] 
        y_label = torch.FloatTensor(np.array(y_mos))
        if "magic3d-refine-sd_a" in img_name and "magic3d-refine-sd_a/a_" not in img_name:
            img_name = img_name.replace("magic3d-refine-sd_a/", "magic3d-refine-sd_a/a_")
        projected_imgs = [] 

        for i in range(self.img_length_read):
            image_name = os.path.join(self.data_dir_2d, img_name, f"{i}.png")
            if not os.path.exists(image_name):
                image_name = os.path.join(self.data_dir_2d_dev, img_name, f"{i}.png")
            processed_image_name = self.load_and_process_image(image_name)
            projected_imgs.append(processed_image_name)

        inputs = self.processor(images=projected_imgs, return_tensors="pt")
        processed_projected_img = inputs['pixel_values']
        return img_name, processed_projected_img, y_label