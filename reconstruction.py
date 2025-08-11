import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from diffusers import AutoencoderKL
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

# User imports
from base.data_eeg import load_eeg_data
from base.utils import update_config, set_seed

# --- Regression Model (Directly maps EEG -> Image Latent) ---
class EEGToLatentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        r_config = config.regression_model
        
        self.eeg_dim = r_config.eeg_dim
        self.latent_dim = r_config.latent_channels * r_config.latent_size * r_config.latent_size
        
        # 1. Project EEG embedding to the model's working dimension
        self.input_proj = nn.Linear(self.eeg_dim, r_config.hidden_dim)
        
        # 2. Transformer Encoder Block
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=r_config.hidden_dim,
            nhead=8,  # Number of attention heads
            dim_feedforward=r_config.hidden_dim * 4, # Standard practice
            dropout=r_config.dropout,
            activation='gelu',
            batch_first=True # IMPORTANT for our data shape
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=r_config.num_layers
        )
        
        # 3. Final MLP head to project to latent space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(r_config.hidden_dim),
            nn.Linear(r_config.hidden_dim, self.latent_dim)
        )

    def forward(self, x):
        # x shape: (batch, eeg_dim)
        # We need a sequence length for the transformer, so we'll add a dummy dimension.
        x = x.unsqueeze(1) # -> (batch, 1, eeg_dim)
        
        x = self.input_proj(x) # -> (batch, 1, hidden_dim)
        x = self.transformer_encoder(x) # -> (batch, 1, hidden_dim)
        
        # Squeeze out the sequence dimension before the final projection
        x = x.squeeze(1) # -> (batch, hidden_dim)
        
        x = self.output_proj(x) # -> (batch, latent_dim)
        return x

# --- Util Function (same as before) ---
@torch.no_grad()
def prepare_latents(config, vae, original_dataloader, save_path):
    # This function is unchanged and works as before
    if os.path.exists(save_path):
        print(f"Loading existing VAE latents from {save_path}")
        return torch.load(save_path)

    print(f"Preparing VAE latents and saving to {save_path}")
    device = vae.device
    dtype = vae.dtype
    
    transform = transforms.Compose([
        transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(1024),
        transforms.ToTensor(),
    ])

    image_paths = [os.path.join(config.image_dir, original_dataloader.dataset[i]['img_path']) for i in range(len(original_dataloader.dataset))]

    class ImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self): return len(self.paths)
        def __getitem__(self, i): return self.transform(Image.open(self.paths[i]).convert("RGB"))

    image_loader = DataLoader(ImageDataset(image_paths, transform), batch_size=config.regression_train.batch_size // 4, num_workers=8)

    all_latents = []
    for image_batch in tqdm(image_loader, desc="Encoding images to latents"):
        image_batch = image_batch.to(device) * 2.0 - 1.0
        latents = vae.encode(image_batch.to(dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        all_latents.append(latents.cpu())

    latents_tensor = torch.cat(all_latents, dim=0)
    torch.save(latents_tensor, save_path)
    print("Latents prepared.")
    return latents_tensor


def main():
    parser = argparse.ArgumentParser()
    # (Same parser setup as before)
    parser.add_argument("--main_config", required=True)
    parser.add_argument("--recon_config", default="configs/eeg/reconstruction.yaml")
    parser.add_argument("--subjects", default='sub-08')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--dataset", default="eeg")
    parser.add_argument("--exp_setting", default='intra-subject')
    parser.add_argument("--brain_backbone", default='EEGProjectLayer')
    parser.add_argument("--vision_backbone", default='RN50')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--c", type=int, default=6)
    args = parser.parse_args()
    set_seed(args.seed)
    
    torch.set_float32_matmul_precision('high')

    # --- Configs and Device ---
    main_config = OmegaConf.load(args.main_config)
    recon_config = OmegaConf.load(args.recon_config)
    config = OmegaConf.merge(main_config, recon_config)
    config = update_config(args, config)
    config['data']['subjects'] = [args.subjects]
    pretrain_map = {'RN50': {'z_dim': 1024}}
    config['z_dim'] = pretrain_map[args.vision_backbone]['z_dim']
    config.regression_model.eeg_dim = config.z_dim
    
    print("--- Combined Config ---")
    print(OmegaConf.to_yaml(config))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.regression_train.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # --- Load VAE ---
    vae = AutoencoderKL.from_pretrained(config.sdxl_model_name, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # --- Prepare Data & Latents ---
    train_loader_orig, _, test_loader_orig = load_eeg_data(config)
    
    train_npz = np.load(config.train_embeddings_path)
    train_eeg_embeds = torch.from_numpy(train_npz['eeg_embeddings'][train_npz['indices']])
    test_npz = np.load(config.test_embeddings_path)
    test_eeg_embeds = torch.from_numpy(test_npz['eeg_embeddings'][test_npz['indices']])

    # Normalize EEG embeddings based on training set statistics
    eeg_mean, eeg_std = train_eeg_embeds.mean(dim=0), train_eeg_embeds.std(dim=0)
    train_eeg_embeds = (train_eeg_embeds - eeg_mean) / eeg_std
    test_eeg_embeds = (test_eeg_embeds - eeg_mean) / eeg_std

    train_image_latents = prepare_latents(config, vae, train_loader_orig, config.train_latents_path)
    test_image_latents = prepare_latents(config, vae, test_loader_orig, config.test_latents_path)
    
    # Normalize latents (important for regression stability)
    latent_mean, latent_std = train_image_latents.mean(), train_image_latents.std()
    train_image_latents = (train_image_latents - latent_mean) / latent_std
    test_image_latents = (test_image_latents - latent_mean) / latent_std

    # We need to save the norm stats to use them in generation
    norm_stats = {'eeg_mean': eeg_mean, 'eeg_std': eeg_std, 'latent_mean': latent_mean, 'latent_std': latent_std}

    # --- Setup Model and Logger ---
    model = EEGToLatentTransformer(config).to(device)
    log_name = OmegaConf.select(config, 'name')
    logger_path = os.path.join(config.save_dir, log_name, f"{args.subjects}_seed{config.seed}")
    os.makedirs(logger_path, exist_ok=True)
    torch.save(norm_stats, os.path.join(logger_path, "norm_stats.pth"))
    
    # --- Training ---
    if args.train:
        print("\n--- Starting EEG to Latent Regression Training ---")
        r_config = config.regression_train
        
        train_dataset = TensorDataset(train_eeg_embeds, train_image_latents)
        train_loader = DataLoader(train_dataset, batch_size=r_config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=r_config.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=r_config.scheduler_t_max, eta_min=1e-6)
        criterion = nn.MSELoss()
        
        for epoch in range(r_config.epoch):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{r_config.epoch}")
            for step, (eeg_batch, latents_batch) in enumerate(progress_bar):
                eeg_batch = eeg_batch.to(device)
                latents_batch = latents_batch.to(device).view(eeg_batch.size(0), -1)

                with torch.autocast(device_type="cuda", dtype=dtype):
                    predicted_latents = model(eeg_batch)
                    loss = criterion(predicted_latents, latents_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            scheduler.step()
        
        torch.save(model.state_dict(), os.path.join(logger_path, "regressor.pth"))
        print(f"Training finished. Model saved to {logger_path}")

    # --- Generation ---
    if args.generate:
        print("\n--- Starting Image Generation from Regression ---")
        model.load_state_dict(torch.load(os.path.join(logger_path, "regressor.pth")))
        model.eval()

        norm_stats = torch.load(os.path.join(logger_path, "norm_stats.pth"))
        
        r_config = config.regression_model
        output_dir = os.path.join(logger_path, "reconstructions")
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(config.regression_generate.num_images_to_generate), desc="Generating images"):
            eeg_embed = test_eeg_embeds[i:i+1].to(device)
            
            with torch.no_grad():
                predicted_latent_flat = model(eeg_embed)
                
            # Reshape, denormalize, and decode
            predicted_latent = predicted_latent_flat.view(1, r_config.latent_channels, r_config.latent_size, r_config.latent_size)
            predicted_latent = (predicted_latent * norm_stats['latent_std']) + norm_stats['latent_mean'] # Denormalize
            predicted_latent = predicted_latent / vae.config.scaling_factor
            
            with torch.no_grad():
                image = vae.decode(predicted_latent.to(dtype)).sample
            
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).to(torch.float32).numpy()
            image = (image * 255).round().astype("uint8")[0]
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(output_dir, f"{i:04d}_recon.png"))
            
            # Also save the ground truth for comparison
            gt_latent = test_image_latents[i:i+1]
            gt_latent = (gt_latent * norm_stats['latent_std']) + norm_stats['latent_mean'] # Denormalize
            gt_latent = gt_latent / vae.config.scaling_factor
            with torch.no_grad():
                gt_image = vae.decode(gt_latent.to(device, dtype=dtype)).sample
            gt_image = (gt_image / 2 + 0.5).clamp(0, 1)
            gt_image = gt_image.cpu().permute(0, 2, 3, 1).to(torch.float32).numpy()
            gt_image = (gt_image * 255).round().astype("uint8")[0]
            pil_gt = Image.fromarray(gt_image)
            pil_gt.save(os.path.join(output_dir, f"{i:04d}_ground_truth.png"))

if __name__ == "__main__":
    main()