import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from pipelines.sdxl.projector import EEGToIPAdapterProjection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", required=True, help="Path to train_embeddings.npz from your encoder run")
    parser.add_argument("--val_embeddings", required=False, help="Optional path to validation/test embeddings.npz")
    parser.add_argument("--eeg_key", default="eeg_embeddings")
    parser.add_argument("--img_key", default="img_embeddings")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", default="exp/projector")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load npz
    train = torch.load(args.train_embeddings, map_location="cpu") if args.train_embeddings.endswith('.pt') else None
    if train is None:
        import numpy as np
        train = np.load(args.train_embeddings)
        train_eeg = torch.from_numpy(train[args.eeg_key]).float()
        train_img = torch.from_numpy(train[args.img_key]).float()
    else:
        train_eeg = train[args.eeg_key].float()
        train_img = train[args.img_key].float()

    # Normalize to unit sphere (cosine space)
    train_eeg = train_eeg / train_eeg.norm(dim=-1, keepdim=True)
    train_img = train_img / train_img.norm(dim=-1, keepdim=True)

    eeg_dim = train_eeg.shape[-1]
    ip_dim = train_img.shape[-1]

    model = EEGToIPAdapterProjection(
        eeg_dim=eeg_dim,
        ip_adapter_dim=ip_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # cosine similarity loss as 1 - cosine similarity
    def cosine_loss(pred, target):
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    dataset = TensorDataset(train_eeg, train_img)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for eeg_batch, img_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            eeg_batch = eeg_batch.to(args.device)
            img_batch = img_batch.to(args.device)

            pred = model(eeg_batch)
            loss = cosine_loss(pred, img_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * eeg_batch.size(0)
        avg = running / len(dataset)
        print(f"Epoch {epoch+1}: loss={avg:.4f}")

        torch.save({
            'model_state': model.state_dict(),
            'config': vars(args),
        }, os.path.join(args.save_dir, 'last.pt'))

    torch.save({
        'model_state': model.state_dict(),
        'config': vars(args),
    }, os.path.join(args.save_dir, 'final.pt'))


if __name__ == "__main__":
    main()
