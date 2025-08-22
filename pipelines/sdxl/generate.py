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
    test_directory = "/ibex/user/qasemiaa/datasets/things_eeg/image_set/test_images"
    get_imgs = lambda folder_path: [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_paths = [d for d in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, d))]
    img_paths.sort()
    img_paths = [i + "/" + get_imgs(os.path.join(test_directory, i))[0] for i in img_paths]
    rl_img = torch.from_numpy(data['img_embeddings']).float()

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
            img_noip = gen.generate(eeg_i, prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance)[0]
            gt_img = Image.open(os.path.join(args.images_root, test_directory, img_paths[i])).convert('RGB')

            # Compose strip: [GT | no-EEGIP | IP]
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
