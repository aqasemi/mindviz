import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, CosineAnnealingLR, SequentialLR
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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "onecycle", "cosine"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Linear warmup ratio for cosine schedule")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast)")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", default="exp/projector")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine similarity loss as 1 - cosine similarity
    def cosine_loss(pred, target):
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)
        return 1.0 - (pred * target).sum(dim=-1).mean()

    dataset = TensorDataset(train_eeg, train_img)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Optional validation
    val_loader = None
    if args.val_embeddings:
        val = torch.load(args.val_embeddings, map_location="cpu") if args.val_embeddings.endswith('.pt') else None
        if val is None:
            import numpy as np
            val = np.load(args.val_embeddings)
            val_eeg = torch.from_numpy(val[args.eeg_key]).float()
            val_img = torch.from_numpy(val[args.img_key]).float()
        else:
            val_eeg = val[args.eeg_key].float()
            val_img = val[args.img_key].float()
        val_eeg = val_eeg / val_eeg.norm(dim=-1, keepdim=True)
        val_img = val_img / val_img.norm(dim=-1, keepdim=True)
        val_loader = DataLoader(TensorDataset(val_eeg, val_img), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Scheduler setup
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    elif args.scheduler == "cosine":
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        main_steps = max(1, total_steps - warmup_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=main_steps)
            ],
            milestones=[warmup_steps]
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (args.device.startswith('cuda') or args.device.startswith('cuda:')))

    best_val = float('inf')
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for eeg_batch, img_batch in progress:
            eeg_batch = eeg_batch.to(args.device, non_blocking=True)
            img_batch = img_batch.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.autocast(device_type=("cuda" if args.device.startswith('cuda') else "cpu"), dtype=torch.bfloat16):
                    pred = model(eeg_batch)
                    loss = cosine_loss(pred, img_batch)
                scaler.scale(loss).backward()
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(eeg_batch)
                loss = cosine_loss(pred, img_batch)
                loss.backward()
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running += loss.item() * eeg_batch.size(0)
            progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg = running / len(dataset)
        msg = f"Epoch {epoch+1}: train_loss={avg:.4f}"

        # Validation
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for ve, vi in val_loader:
                    ve = ve.to(args.device, non_blocking=True)
                    vi = vi.to(args.device, non_blocking=True)
                    pred = model(ve)
                    vloss = cosine_loss(pred, vi)
                    val_running += vloss.item() * ve.size(0)
            val_avg = val_running / len(val_loader.dataset)
            msg += f", val_loss={val_avg:.4f}"
            # Save best
            if val_avg < best_val:
                best_val = val_avg
                torch.save({
                    'model_state': model.state_dict(),
                    'config': vars(args),
                    'best_val_loss': best_val,
                    'epoch': epoch + 1,
                }, os.path.join(args.save_dir, 'best.pt'))
            model.train()

        print(msg)

        # Always save last
        torch.save({
            'model_state': model.state_dict(),
            'config': vars(args),
            'epoch': epoch + 1,
        }, os.path.join(args.save_dir, 'last.pt'))

    torch.save({
        'model_state': model.state_dict(),
        'config': vars(args),
    }, os.path.join(args.save_dir, 'final.pt'))


if __name__ == "__main__":
    main()
