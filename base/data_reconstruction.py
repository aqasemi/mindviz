import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class ReconstructionDataset(Dataset):
    """
    Dataset for SDXL fine-tuning. It pairs ground truth images with their pre-generated EEG embeddings.
    """
    def __init__(self, config, embeddings, split='train'):
        """
        Args:
            config: The main configuration object.
            embeddings (dict): A dictionary loaded from a .pt file, containing 'embeddings' and 'img_paths'.
            split (str): 'train' or 'test'.
        """
        self.config = config
        self.split = split
        self.image_dir = config['image_dir']
        
        self.eeg_embeds = embeddings['embeddings']
        self.img_paths = embeddings['img_paths']

        # Pre-process images for SDXL (1024x1024)
        self.transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.eeg_embeds)

    def __getitem__(self, idx):
        eeg_embed = self.eeg_embeds[idx]
        img_path = self.img_paths[idx]
        
        full_img_path = os.path.join(self.image_dir, img_path)
        try:
            image = Image.open(full_img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            print(f"Could not load image {full_img_path}: {e}. Skipping.")
            # Return a different sample if the current one is broken
            return self.__getitem__((idx + 1) % len(self))

        if self.split == 'train':
            return {
                "pixel_values": pixel_values,
                "conditioning_embeds": eeg_embed,
            }
        else: # test split
            return {
                "conditioning_embeds": eeg_embed,
                "ground_truth_image": pixel_values,
                "image_path": img_path
            }