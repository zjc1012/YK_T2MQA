import argparse
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from models.ELHnet import ELHnet
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import torch
import random
from torch.utils import data
import cv2
from transformers import SiglipProcessor

class ELH_Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir_2d, data_dir_2d_dev, datainfo_path,
                 crop_size=512, img_length_read=4, is_train=True):
        super(ELH_Inference_Dataset, self).__init__()
        
        dataInfo = pd.read_csv(datainfo_path, header=0, index_col=False, encoding="utf-8-sig")
        self.image_paths = dataInfo['Image'] 
        self.crop_size = crop_size
        self.data_dir_2d = data_dir_2d
        self.data_dir_2d_dev = data_dir_2d_dev
        self.img_length_read = img_length_read
        self.length = len(self.image_paths)
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
        img_name = self.image_paths.iloc[idx].strip('"')
        img_name1 = img_name
        projected_imgs = [] 
        
        if "magic3d-refine-sd_a" in img_name and "magic3d-refine-sd_a/a_" not in img_name:
            img_name = img_name.replace("magic3d-refine-sd_a/", "magic3d-refine-sd_a/a_")

        for i in range(self.img_length_read):
            image_name = os.path.join(self.data_dir_2d, img_name, f"{i}.png")
            if os.path.exists(image_name):
                processed_image_name = self.load_and_process_image(image_name)
                projected_imgs.append(processed_image_name)
            else:
                print(f"Image not found: {image_name}")

        inputs = self.processor(images=projected_imgs, return_tensors="pt")
        processed_projected_img = inputs['pixel_values']
        y_label = torch.FloatTensor([0.0])
        return img_name1, processed_projected_img, y_label

def set_rand_seed(seed=1998):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with multiple checkpoints")
    parser.add_argument('--batch_size', help='Batch size for inference', default=16, type=int)
    parser.add_argument('--data_dir_2d', required=True, type=str, help='path to the 2d images')
    parser.add_argument('--data_dir_2d_dev', required=True, type=str, help='path to the other 2d images')
    parser.add_argument('--img_length_read', default=6, type=int, help='number of the using images')
    parser.add_argument('--ckpt_dir', required=True, type=str, help='directory containing all pth checkpoint files')
    parser.add_argument('--input_csv', required=True, type=str, help='path to input csv file (Image,Score)')
    parser.add_argument('--output_dir', required=True, type=str, help='directory to save output csv files')
    parser.add_argument('--crop_size', default=512, type=int, help='image crop size')
    
    args = parser.parse_args()
    return args

def load_model(checkpoint_path, device, args, score_list, quality_classes):
    model = ELHnet(device, args, score_list, quality_classes)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model

def create_inference_dataset(args, input_csv_path):
    dataset = ELH_Inference_Dataset(
        data_dir_2d=args.data_dir_2d,
        data_dir_2d_dev=args.data_dir_2d_dev,
        datainfo_path=input_csv_path,
        crop_size=args.crop_size,
        img_length_read=args.img_length_read,
        is_train=False  # 推理模式
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return dataset, dataloader

def inference_single_model(model, dataloader, device, input_df):
    model.eval()
    predictions = []
    image_paths = []
    
    with torch.no_grad():
        for batch_idx, (img_names, imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            quality_score = model(img_names, imgs)
            preds = quality_score.cpu().numpy().flatten()
            predictions.extend(preds)
            image_paths.extend(img_names)
    
    pred_df = pd.DataFrame({
        'Image': image_paths,
        'Score': predictions
    })
    
    input_df['Image'] = input_df['Image'].astype(str).str.strip('"')
    pred_df['Image'] = pred_df['Image'].astype(str).str.strip('"')
    input_df = input_df.drop('Score', axis=1)  # 删除原有的空Score列
    result_df = input_df.merge(pred_df, on='Image', how='left')
    
    return result_df

def save_results(result_df, output_path):
    result_df['Score'] = pd.to_numeric(result_df['Score'], errors='coerce')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {output_path}")
    
    return result_df

def aggregate_results(output_dir, input_csv_path, output_filename='output.csv'):
    result_files = [f for f in os.listdir(output_dir) 
                   if f.endswith('.csv') and not f == output_filename]
    if not result_files:
        print("No result files found for aggregation!")
        return
    
    input_df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    input_df['Image'] = input_df['Image'].astype(str).str.strip('"')
    input_df = input_df.drop('Score', axis=1)
    for file in result_files:
        file_path = os.path.join(output_dir, file)
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['Image'] = df['Image'].astype(str).str.strip('"')
        col_name = file.replace('.csv', '')
        input_df = input_df.merge(df[['Image', 'Score']], on='Image', how='left')
        input_df = input_df.rename(columns={'Score': col_name})
    score_columns = [f.replace('.csv', '') for f in result_files]
    input_df['Score'] = input_df[score_columns].mean(axis=1)
    
    final_agg_df = input_df[['Image', 'Score']].copy()
    output_path = os.path.join(output_dir, output_filename)
    final_agg_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Aggregated results saved to: {output_path}")
    
    return final_agg_df

def main():
    args = parse_args()
    set_rand_seed(1998)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    score_list = [0.0, 1.25, 2.5, 3.75, 5.0]
    quality_classes = ['bad', 'poor', 'fair', 'good', 'perfect']
    input_df = pd.read_csv(args.input_csv, encoding='utf-8-sig')
    print(f"Loaded input dataset with {len(input_df)} samples")
    
    dataset, dataloader = create_inference_dataset(args, args.input_csv)
    
    ckpt_files = [f for f in os.listdir(args.ckpt_dir) if f.endswith('.pth')]
    if not ckpt_files:
        print(f"No .pth files found in {args.ckpt_dir}")
        return
    
    print(f"Found {len(ckpt_files)} checkpoint files for inference")
    
    all_results = []
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
        print(f"\nProcessing checkpoint: {ckpt_file}")
        
        try:
            model = load_model(ckpt_path, device, args, score_list, quality_classes)
            result_df = inference_single_model(model, dataloader, device, input_df)
            output_filename = ckpt_file.replace('.pth', '.csv')
            output_path = os.path.join(args.output_dir, output_filename)
            
            final_df = save_results(result_df, output_path)
            final_df['model_name'] = ckpt_file.replace('.pth', '')
            all_results.append(final_df)
                
        except Exception as e:
            print(f"Error processing {ckpt_file}: {str(e)}")
            continue
    if all_results:
        print("\nAggregating results from all models...")
        aggregate_results(args.output_dir, args.input_csv)
    else:
        print("No results to aggregate!")

if __name__ == '__main__':
    main()