import os

import torch
import torch.nn as nn
import sys
sys.path.append("../")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import transformers
print(transformers.__version__)

import numpy as np

test_embeds = np.load("/home/qasemiaa/mjo/upb/exp/eeg_intra-subject_ubp_EEGProjectLayer_ViT-H-14/sub-08_seed29/test_embeddings.npz", allow_pickle=True)
test_eeg_embeds = torch.tensor(test_embeds['eeg_embeddings'], dtype=torch.float32).to(device)
test_image_embeds = torch.tensor(test_embeds['img_embeddings'], dtype=torch.float32).to(device)

# train_embeds = np.load("/home/qasemiaa/mjo/upb/exp/eeg_intra-subject_ubp_EEGProjectLayer_ViT-H-14/sub-08_seed29/train_embeddings.npz", allow_pickle=True)
# train_eeg_embeds = torch.tensor(train_embeds['eeg_embeddings'], dtype=torch.float32).to(device)
# train_image_embeds = torch.tensor(train_embeds['img_embeddings'], dtype=torch.float32).to(device)


ret_test_embeds = np.load("/home/qasemiaa/mjo/upb/exp/eeg_intra-subject_ubp_EEGProjectLayer_RN50/sub-08_seed0/test_embeddings.npz", allow_pickle=True)
ret_test_img_embeds = torch.tensor(ret_test_embeds['img_embeddings'], dtype=torch.float32)
ret_test_eeg_embeds = torch.tensor(ret_test_embeds['eeg_embeddings'], dtype=torch.float32)

def top_k(eeg_z, img_z, k=5):
    eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
    similarity = (eeg_z @ img_z.T)
    top_kvalues, top_k_indices = similarity.topk(k, dim=-1)
    return top_k_indices


import torch
from torch import nn
import matplotlib.pyplot as plt


# from reconstruction.diffusion_prior import *
from custom_pipeline_low_level import Generator4Embeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from torchvision import transforms


def get_ssim(rec, img):
    recon_gray = rgb2gray(rec.resize((500, 500)))
    img_gray = rgb2gray(img.resize((500, 500)))
    return ssim(recon_gray, img_gray, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)

    

train = False # doesn't work for train; gotta use 'indicies' values to order the images
classes = None
pictures = None

def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []
    
    if train:
        text_directory = "/ibex/user/qasemiaa/datasets/things_eeg/image_set/training_images"
    else:
        text_directory = "/ibex/user/qasemiaa/datasets/things_eeg/image_set/test_images"
    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        new_description = f"{description}"
        texts.append(new_description)

    if train:
        img_directory = "/ibex/user/qasemiaa/datasets/things_eeg/image_set/training_images"
    else:
        img_directory ="/ibex/user/qasemiaa/datasets/things_eeg/image_set/test_images"
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images
texts, images = load_data()

generator = Generator4Embeds(num_inference_steps=4, device=device)

from PIL import Image
from pathlib import Path
import os

# Assuming generator.generate returns a PIL Image
sub = "08"
directory = f"generated_imgs/{sub}"
for k in range(40): 
    print(f"--- Processing Sample {k} ---")
    num_generated = 2
    
    eeg_embeds = test_eeg_embeds[k:k+1]
    top_g = top_k(ret_test_eeg_embeds[k:k+1], ret_test_img_embeds, 2)[0].tolist()
    fig, axes = plt.subplots(1, num_generated + 1, figsize=(15, 5))
    fig.suptitle(f'Results for Sample {k}', fontsize=16)


    for j in range(num_generated):
        image_class_name = ' '.join(Path(images[top_g[j]]).name.split("_")[:-1])
        image = generator.generate(eeg_embeds.unsqueeze(0).to(dtype=torch.float16), text_prompt=image_class_name)

        image_dir = f'{directory}/image_{k}'
        os.makedirs(image_dir, exist_ok=True)
        image.save(f'{image_dir}/{j}.png')
        
        axes[j].imshow(image)
        axes[j].set_title(f'Image {j+1} ') # Class: {image_class_name}')
        axes[j].axis('off') 

    # Save the ground truth image in the same folder
    gt_path = images[k]
    gt_image = Image.open(gt_path).convert("RGB") 
    gt_save_path = f'{directory}/image_{k}/gt.png'
    gt_image.save(gt_save_path)
    
    # Plot the ground truth image in the last subplot
    axes[num_generated].imshow(gt_image)
    axes[num_generated].set_title('Ground Truth')
    axes[num_generated].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()




