#!/usr/bin/env python3
"""Minimal multi-seed U-Net training and probability-map export scaffold.

This script is intentionally lightweight. It expects a manifest CSV with columns:
  id, split, image_path, binary_mask_path
where split is train/val/test. It trains one U-Net per seed and exports test probability maps.
For a production paper, tune hyperparameters on validation only and record the final config.
"""
from pathlib import Path
import argparse, random, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='runs/unet_multiseed')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2024])
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    # Import torch lazily so metric-only users do not need torch.
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from skimage import io, transform

    class NucleiDataset(Dataset):
        def __init__(self, frame, image_size):
            self.frame = frame.reset_index(drop=True); self.image_size = image_size
        def __len__(self): return len(self.frame)
        def __getitem__(self, idx):
            r = self.frame.iloc[idx]
            img = io.imread(r.image_path)
            if img.ndim == 2: img = np.stack([img]*3, axis=-1)
            if img.shape[-1] > 3: img = img[..., :3]
            mask = io.imread(r.binary_mask_path)
            if mask.ndim == 3: mask = mask[..., 0]
            img = transform.resize(img, (self.image_size, self.image_size), preserve_range=True, anti_aliasing=True).astype(np.float32) / 255.0
            mask = transform.resize(mask > 0, (self.image_size, self.image_size), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
            img = np.transpose(img, (2,0,1))
            return torch.from_numpy(img), torch.from_numpy(mask[None]), str(r.id)

    class DoubleConv(nn.Module):
        def __init__(self, c1, c2):
            super().__init__(); self.net = nn.Sequential(nn.Conv2d(c1,c2,3,padding=1), nn.ReLU(inplace=True), nn.Conv2d(c2,c2,3,padding=1), nn.ReLU(inplace=True))
        def forward(self,x): return self.net(x)
    class UNetSmall(nn.Module):
        def __init__(self):
            super().__init__(); self.d1=DoubleConv(3,32); self.p1=nn.MaxPool2d(2); self.d2=DoubleConv(32,64); self.p2=nn.MaxPool2d(2); self.mid=DoubleConv(64,128); self.u2=nn.ConvTranspose2d(128,64,2,2); self.c2=DoubleConv(128,64); self.u1=nn.ConvTranspose2d(64,32,2,2); self.c1=DoubleConv(64,32); self.out=nn.Conv2d(32,1,1)
        def forward(self,x):
            x1=self.d1(x); x2=self.d2(self.p1(x1)); xm=self.mid(self.p2(x2)); x=self.u2(xm); x=self.c2(torch.cat([x,x2],1)); x=self.u1(x); x=self.c1(torch.cat([x,x1],1)); return self.out(x)
    def dice_loss(logits, y, eps=1e-6):
        p = torch.sigmoid(logits); inter=(p*y).sum(dim=(1,2,3)); den=p.sum(dim=(1,2,3))+y.sum(dim=(1,2,3)); return 1 - ((2*inter+eps)/(den+eps)).mean()

    df = pd.read_csv(args.manifest)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = UNetSmall().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss()
        train = NucleiDataset(df[df.split=='train'], args.image_size); val = NucleiDataset(df[df.split=='val'], args.image_size); test = NucleiDataset(df[df.split=='test'], args.image_size)
        tr_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            model.train(); losses=[]
            for x,y,_ in tr_loader:
                x=x.to(device); y=y.to(device); opt.zero_grad(); logits=model(x); loss=bce(logits,y)+dice_loss(logits,y); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            print(f'seed={seed} epoch={epoch+1}/{args.epochs} loss={np.mean(losses):.4f}')
        seed_dir = out/f'seed_{seed}'; (seed_dir/'prob_maps').mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'seed': seed, 'args': vars(args)}, seed_dir/'checkpoint.pt')
        model.eval(); loader=DataLoader(test, batch_size=1, shuffle=False)
        with torch.no_grad():
            for x,_,ids in loader:
                prob = torch.sigmoid(model(x.to(device))).cpu().numpy()[0,0]
                np.save(seed_dir/'prob_maps'/f'{ids[0]}.npy', prob)
        print(f'Wrote probability maps to {seed_dir / "prob_maps"}')

if __name__ == '__main__':
    main()
