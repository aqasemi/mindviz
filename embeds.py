import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
from scipy.stats import norm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader # Added for creating a non-shuffled dataloader

##import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device

device = get_device('auto')
print(f"Using device: {device}")

# --- NEW UTILITY FUNCTION ---
def generate_and_save_embeddings_from_model(model, dataloader, save_path, device):
    """
    Generates embeddings for a dataset using a trained model and saves them.
    """
    model.to(device)
    model.eval()

    all_eeg_embeddings = []
    all_img_embeddings = []
    all_indices = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"\rGenerating batch {i+1}/{len(dataloader)}", end="")
            # Move batch to the correct device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get embeddings from the model's forward pass
            eeg_z, img_z, _ = model(batch)
            
            all_eeg_embeddings.append(eeg_z.cpu())
            all_img_embeddings.append(img_z.cpu())
            all_indices.append(batch['idx'].cpu())
            
    print("\nGeneration complete.")

    # Concatenate all batches
    all_eeg_embeddings = torch.cat(all_eeg_embeddings, dim=0).numpy()
    all_img_embeddings = torch.cat(all_img_embeddings, dim=0).numpy()
    all_indices = torch.cat(all_indices, dim=0).numpy()

    # Sort embeddings by index to ensure a consistent order
    sort_order = np.argsort(all_indices)
    all_eeg_embeddings = all_eeg_embeddings[sort_order]
    all_img_embeddings = all_img_embeddings[sort_order]
    all_indices = all_indices[sort_order]

    # Save to a compressed numpy file
    np.savez(save_path,
             eeg_embeddings=all_eeg_embeddings,
             img_embeddings=all_img_embeddings,
             indices=all_indices)
    print(f"Embeddings saved to {save_path}")


def load_model(config, train_loader, test_loader, log_dir):
    """ Modified to accept and set the log_dir for saving files. """
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    pl_model.log_dir = log_dir # Set the log_dir attribute for later use
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader, model_type='RN50'):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []
    
        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3
        
        self.mAP_total = 0
        self.match_similarities = []

        # --- MODIFIED ---
        # Add a placeholder for the logging directory
        self.log_dir = None
        # Add lists to store test outputs for later saving
        self.test_eeg_embeddings = []
        self.test_img_embeddings = []
        self.test_indices = []
        

    def forward(self, batch, sample_posterior=False):
        idx = batch['idx'].cpu().detach().numpy() 
        eeg = batch['eeg']
        img = batch['img']
        img_z = batch['img_features']
        
        eeg_z = self.brain(eeg)
        img_z = img_z/img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma
            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]
            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)
            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim
            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2
            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label
            loss = total_loss
        else:
            loss = total_loss
        return eeg_z, img_z, loss
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch,sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True,logger=True, sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss= self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss
    
    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        
        # --- MODIFIED: Store embeddings and indices for saving later ---
        self.test_eeg_embeddings.append(eeg_z.cpu().detach())
        self.test_img_embeddings.append(img_z.cpu().detach())
        self.test_indices.append(batch['idx'].cpu().detach())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z/eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        #compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero(as_tuple=True)[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        
        return loss
        
    def on_test_epoch_end(self):
        # --- MODIFIED: Save test embeddings at the beginning of the hook ---
        if self.test_eeg_embeddings:  # Only run if test_step was executed
            # Concatenate local results first
            local_eeg = torch.cat(self.test_eeg_embeddings, dim=0)
            local_img = torch.cat(self.test_img_embeddings, dim=0)
            local_idx = torch.cat(self.test_indices, dim=0)

            # Gather results from all processes (works for DDP and single-GPU)
            gathered_eeg = self.all_gather(local_eeg)
            gathered_img = self.all_gather(local_img)
            gathered_idx = self.all_gather(local_idx)
            
            # Save on the main process
            if self.trainer.is_global_zero:
                print("\nAggregating and saving test embeddings...")
                # Concatenate results from all ranks. `gathered_` is a list of tensors.
                all_eeg_embeddings = torch.cat([t for t in gathered_eeg], dim=0).cpu().numpy()
                all_img_embeddings = torch.cat([t for t in gathered_img], dim=0).cpu().numpy()
                all_indices = torch.cat([t for t in gathered_idx], dim=0).cpu().numpy()

                # Sort embeddings by index
                sort_order = np.argsort(all_indices)
                all_eeg_embeddings = all_eeg_embeddings[sort_order]
                all_img_embeddings = all_img_embeddings[sort_order]
                all_indices = all_indices[sort_order]

                # Save to file
                save_path = os.path.join(self.log_dir, 'test_embeddings.npz')
                np.savez(save_path,
                         eeg_embeddings=all_eeg_embeddings,
                         img_embeddings=all_img_embeddings,
                         indices=all_indices)
                print(f"Test embeddings saved to {save_path}")

        # Clear lists on all ranks for any subsequent test runs
        self.test_eeg_embeddings.clear()
        self.test_img_embeddings.clear()
        self.test_indices.clear()

        # --- Original on_test_epoch_end logic continues below ---
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes.clear()
        self.all_true_labels.clear()

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(), 'test_top5_acc': top_k_accuracy.item(), 'mAP': self.mAP, 'similarity': self.match_similarities}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        return [optimizer]
    
def main():
    parser = argparse.ArgumentParser()
    # ... (rest of argparse setup is unchanged)
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name=config['name'], version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}")
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(opt.config, os.path.join(log_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)

    print(f"train num: {len(train_loader.dataset)},val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader, log_dir) # Pass log_dir

    checkpoint_callback = ModelCheckpoint(save_last=True)

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(monitor='val_top1_acc', min_delta=0.001, patience=5, verbose=False, mode='max')
    else:
        early_stop_callback = EarlyStopping(monitor='train_loss', min_delta=0.001, patience=5, verbose=False, mode='min')

    trainer = Trainer(log_every_n_steps=10, strategy='ddp_find_unused_parameters_false', callbacks=[early_stop_callback, checkpoint_callback], max_epochs=config['train']['epoch'], devices=[device], accelerator='cuda', logger=logger)

    ckpt_path = 'last'
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    # --- MODIFIED: Generate and save training embeddings after training is complete ---
    if trainer.is_global_zero:
        print("\nGenerating training embeddings with final model...")
        # Create a new dataloader from the training dataset with shuffle=False for ordered generation
        train_dataset = train_loader.dataset
        train_loader_for_generation = DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=False, # Important for ordered output
            num_workers=train_loader.num_workers
        )
        train_save_path = os.path.join(log_dir, 'train_embeddings.npz')
        generate_and_save_embeddings_from_model(pl_model, train_loader_for_generation, train_save_path, device)

    # Use a barrier to make sure all processes wait for rank 0 to finish saving before testing
    if torch.distributed.is_initialized() and trainer.world_size > 1:
        torch.distributed.barrier()

    # --- Test the model (this will also generate and save test embeddings) ---
    if config['exp_setting'] == 'inter-subject':
        test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    else:
        test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    if trainer.is_global_zero:
        with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

if __name__ == "__main__":
    main()