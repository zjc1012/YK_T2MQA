import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import transformers
from transformers import AutoModel, SiglipProcessor
import os

class DynamicPromptGenerator(nn.Module):
    def __init__(self, quality_classes):
        super().__init__()
        self.quality_classes = quality_classes
        self.num_quality_classes = len(quality_classes)
        
    def forward(self, img_names):
        all_prompts = []
        for img_name in img_names:
            for quality in self.quality_classes:
                prompt = f"The visual quality of this image is {quality}."
                all_prompts.append(prompt)
        return all_prompts

class ELHnet(nn.Module):
    def __init__(self, device, args, score_list, quality_classes):
        super(ELHnet, self).__init__()
        self.device = device
        self.score_list = score_list
        self.quality_classes = quality_classes
        self.num_quality_classes = len(quality_classes)
        self.img_length_read = args.img_length_read 

        self.dynamic_prompt_generator = DynamicPromptGenerator(quality_classes)
        self.ckpt = "google/siglip2-base-patch16-512"
        
        self.model = AutoModel.from_pretrained(self.ckpt, device_map=None)
        for name, param in self.model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.processor = SiglipProcessor.from_pretrained(self.ckpt, use_fast=False)
        self.register_buffer("logit_scale", torch.tensor(self.model.logit_scale.item()))

    def _compute_quality_score(self, imgs, current_device):
        local_batch_size, num_views_split, channels, image_height, image_width = imgs.shape
        
        imgs_reshaped = imgs.reshape(-1, channels, image_height, image_width).to(current_device)
        image_features_output = self.model.get_image_features(imgs_reshaped)
        image_embeddings = image_features_output.pooler_output if hasattr(image_features_output, 'pooler_output') else image_features_output
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        _, C = image_embeddings.shape
        image_f = image_embeddings.reshape(local_batch_size, num_views_split, C).mean(dim=1, keepdim=False)

        return image_f, C
    
    def forward(self, img_names, texture_imgs):
        current_device = texture_imgs.device
        local_batch_size, num_views, channels, image_height, image_width = texture_imgs.shape
        
        if torch.cuda.is_available():
            gpu_idx = current_device.index if current_device.index is not None else 0
        else:
            gpu_idx = 0
        start_idx = gpu_idx * local_batch_size
        end_idx = start_idx + local_batch_size
        img_names = img_names[start_idx:end_idx]
        
        global_image_f, C = self._compute_quality_score(texture_imgs, current_device)
        global_prompts = self.dynamic_prompt_generator(img_names)
        global_tokenized = self.processor(
            text=global_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(current_device) 
        
        with torch.no_grad():
            global_text_f = self.model.get_text_features(**global_tokenized)
        if hasattr(global_text_f, 'pooler_output'):
            global_text_f = global_text_f.pooler_output
        else:
            global_text_f = global_text_f
            
        global_text_f = global_text_f / global_text_f.norm(dim=-1, keepdim=True)
        global_text_f = global_text_f.reshape(local_batch_size, self.num_quality_classes, C).to(current_device)

        logit_scale = self.logit_scale.exp().to(current_device)
        global_logits = logit_scale * torch.bmm(
            global_image_f.unsqueeze(1),
            global_text_f.permute(0, 2, 1)
        ).squeeze(1)
        global_pred = F.softmax(global_logits, dim=1).to(current_device)
        bin_tensor = torch.tensor(self.score_list, device=current_device).reshape(1, self.num_quality_classes)
        global_score = (global_pred * bin_tensor).sum(1, keepdim=True).to(current_device)

        return global_score


