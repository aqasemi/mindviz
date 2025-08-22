import os
import re
import argparse
import torch
from PIL import Image

from pipelines.sdxl.generator import SDXLGenerator
from pipelines.sdxl.projector import EEGToIPAdapterProjection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npz (test set)")
    parser.add_argument("--weights", required=True, help="Path to trained projector .pt")
    parser.add_argument("--out", default="exp/sdxl_recon")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--images_root", type=str, default=None, help="Root directory containing ground-truth images (expects relative img_paths in npz)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load embeddings npz
    import numpy as np
    data = np.load(args.embeddings)
    eeg = torch.from_numpy(data['eeg_embeddings']).float()
    # If available, use `img_paths` to load ground-truth; otherwise fallback to none
    img_paths = data.get('img_paths') if isinstance(data, np.lib.npyio.NpzFile) else None
    # Optional: use img embeddings for debugging quality
    # img = torch.from_numpy(data['img_embeddings']).float()

    # Prepare numbered GT list if requested
    numbered_gt_paths = None
    if args.images_root is not None and os.path.isdir(args.images_root):
        exts = {'.png', '.jpg', '.jpeg', '.webp'}

        def natural_key(name: str):
            m = re.search(r"(\d+)", os.path.basename(name))
            return int(m.group(1)) if m else name

        candidates = [
            os.path.join(args.images_root, f)
            for f in os.listdir(args.images_root)
            if os.path.splitext(f.lower())[1] in exts
        ]
        numbered_gt_paths = sorted(candidates, key=natural_key) if candidates else None

    # Load projector
    ckpt = torch.load(args.weights, map_location='cpu')
    cfg = ckpt['config']
    eeg_dim = eeg.shape[-1]
    model = EEGToIPAdapterProjection(
        eeg_dim=eeg_dim,
        ip_adapter_dim=cfg.get('ip_adapter_dim', 1024),
        hidden_dim=cfg.get('hidden_dim', 2048),
        num_layers=cfg.get('layers', 4),
        dropout=cfg.get('dropout', 0.1),
    ).to(args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Setup SDXL generator
    gen = SDXLGenerator(device=args.device)

    with torch.no_grad():
        for i in range(min(args.num_images, eeg.size(0))):
            eeg_i = eeg[i:i+1].to(args.device)
            ip_embed = model(eeg_i)
            img_ip = gen.generate(ip_embed, prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance)[0]

            # Baseline without IP-Adapter
            img_noip = gen.generate_no_adapter(prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance)[0]

            # Ground-truth image, if paths are provided
            gt_img = None
            try:
                if img_paths is not None:
                    rel = img_paths[i] if isinstance(img_paths, np.ndarray) else None
                    if rel is not None:
                        if args.images_root is not None:
                            full = os.path.join(args.images_root, rel)
                            if os.path.exists(full):
                                gt_img = Image.open(full).convert('RGB')
                        else:
                            # Fallback: try common defaults
                            proj_root = os.getcwd()
                            candidates = [
                                os.path.join(proj_root, 'data', 'things-eeg', 'image_set_resize'),
                                os.path.join(proj_root, 'data', 'things-meg', 'Image_set_Resize'),
                                os.path.join(proj_root, 'data', 'things-meg', 'image_set_resize'),
                            ]
                            for base in candidates:
                                full = os.path.join(base, rel)
                                if os.path.exists(full):
                                    gt_img = Image.open(full).convert('RGB')
                                    break
                elif numbered_gt_paths is not None and i < len(numbered_gt_paths):
                    gt_img = Image.open(numbered_gt_paths[i]).convert('RGB')
            except Exception:
                gt_img = None

            # Compose strip: [GT | no-IP | IP]
            tiles = []
            if gt_img is not None:
                tiles.append(gt_img)
            tiles.append(img_noip)
            tiles.append(img_ip)

            # Ensure same size
            w, h = tiles[-1].size
            tiles = [t.resize((w, h), Image.LANCZOS) for t in tiles]

            # Concatenate horizontally
            total_w = w * len(tiles)
            out_img = Image.new('RGB', (total_w, h))
            x = 0
            for t in tiles:
                out_img.paste(t, (x, 0))
                x += w

            out_img.save(os.path.join(args.out, f"{i:04d}.png"))


if __name__ == "__main__":
    main()
