import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset
from diffusers.optimization import get_cosine_schedule_with_warmup

class DiffusionPriorUNet(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            cond_dim=1024,
            hidden_dim=[1024, 512, 256], # Simplified for stability
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.time_proj = Timesteps(time_embed_dim, True, 0)

        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(time_embed_dim, hd) for hd in hidden_dim[:-1]]
        )
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hd) for hd in hidden_dim[:-1]]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                nn.LayerNorm(hidden_dim[i+1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers-1)]
        )

        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(time_embed_dim, hd) for hd in reversed(hidden_dim[:-1])]
        )
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hd) for hd in reversed(hidden_dim[:-1])]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                nn.LayerNorm(hidden_dim[i-1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers-1, 0, -1)]
        )

        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, t, c=None):
        t_emb_orig = self.time_proj(t)
        x = self.input_layer(x)
        hidden_activations = []
        
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t_emb_orig)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t_emb_orig)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x = x + hidden_activations.pop()
            
        x = self.output_layer(x)
        return x

class EmbeddingDataset(Dataset):
    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {"c_embedding": self.c_embeddings[idx], "h_embedding": self.h_embeddings[idx]}

class Pipe:
    def __init__(self, diffusion_prior, scheduler, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        self.scheduler = scheduler
        self.device = device
        
    def train(self, dataloader, num_epochs=100, learning_rate=1e-4):
        self.diffusion_prior.train()
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.diffusion_prior.parameters(), lr=learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            loss_sum = 0
            for batch in progress_bar:
                c_embeds = batch['c_embedding'].to(self.device)
                h_embeds = batch['h_embedding'].to(self.device)
                
                if torch.rand(1) < 0.1: # Classifier-free guidance training
                    c_embeds = torch.zeros_like(c_embeds)

                noise = torch.randn_like(h_embeds)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (h_embeds.shape[0],), device=self.device)
                perturbed_h_embeds = self.scheduler.add_noise(h_embeds, noise, timesteps)
                
                noise_pred = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                loss = criterion(noise_pred, noise)
                            
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = loss_sum / len(dataloader)
            print(f'Epoch {epoch+1} finished. Average Loss: {avg_loss:.5f}')

    @torch.no_grad()
    def generate(self, c_embeds, num_inference_steps=50, guidance_scale=5.0, generator=None):
        self.diffusion_prior.eval()
        N = c_embeds.shape[0]
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)
        c_embeds_uncond = torch.zeros_like(c_embeds)
        
        for t in tqdm(timesteps, desc="Generating Embeddings"):
            t_batch = t.repeat(N)
            
            noise_pred_cond = self.diffusion_prior(h_t, t_batch, c_embeds)
            noise_pred_uncond = self.diffusion_prior(h_t, t_batch, c_embeds_uncond)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            h_t = self.scheduler.step(noise_pred, t, h_t, generator=generator).prev_sample
        
        return h_t