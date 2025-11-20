import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_dTm_3D")))
# Import the geometric encoder from your model.py
from model import PretrainEncoder, ATOM_CA

###############################################################################
# GeoDTm dataset with ESMFold pLDDT from pickles
###############################################################################

class GeoDTmDataset(Dataset):
    """
    Loads WT/Mut features (esm2.pt, fixed_embedding.pt, pair.pt, coordinate.pt)
    and ESMFold plddt scores from pickle (*.pkl).
    Places plddt vector as last column into fixed_embedding (or concatenates).
    """
    def __init__(self, csv_or_df, features_dir: str):
        super().__init__()
        self.df = pd.read_csv(csv_or_df) if isinstance(csv_or_df, str) else csv_or_df.reset_index(drop=True)
        self.features_dir = features_dir
        assert "name" in self.df.columns, "CSV must contain a 'name' column."
        assert "dTm" in self.df.columns, "CSV must contain a 'dTm' column."

    def _load_feature_dict(self, sample_id: str, variant: str):
        folder = os.path.join(self.features_dir, sample_id, variant)
        out = {}
        # Load ESM2 embedding
        out["dynamic_embedding"] = torch.load(os.path.join(folder, "esm2.pt")).float()
        # Load physicochemical fixed embedding (this defines the "true" sequence length)
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()
        L_fixed = fixed.shape[0]

        # Load pLDDT from ESMFold pickle
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
            plddt_raw = torch.tensor(pkl["plddt"], dtype=torch.float32)
            # ---- PATCH: Remove batch dim if present ----
            if plddt_raw.ndim == 2 and plddt_raw.shape[0] == 1:
                plddt_raw = plddt_raw[0]
        L_plddt = plddt_raw.shape[0]

        # --- Fix pLDDT length to match fixed_embedding length ---
        if L_plddt != L_fixed:
            if L_plddt > L_fixed:
                print(f"[GeoDTmDataset] Cropping pLDDT from {L_plddt} to {L_fixed} for {folder}", flush=True)
                plddt_raw = plddt_raw[:L_fixed]
            else:
                print(f"[GeoDTmDataset] WARNING: pLDDT shorter ({L_plddt}) than fixed_embedding ({L_fixed}) for {folder}. Padding.", flush=True)
                pad = plddt_raw[-1].repeat(L_fixed - L_plddt)
                plddt_raw = torch.cat([plddt_raw, pad], dim=0)

        # Normalize to [0, 1]
        plddt = plddt_raw / 100.0

        # ---- PATCH: Now shapes are [L, 7] and [L, 1], safe to concatenate
        out["fixed_embedding"] = torch.cat([fixed, plddt.unsqueeze(-1)], dim=-1)
        # Load pair features
        out["pair"] = torch.load(os.path.join(folder, "pair.pt")).float()
        # Load atom mask (from coordinate.pt)
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        out["atom_mask"] = coord_data["pos14_mask"].all(dim=-1).float()
        return out

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["name"])
        target = float(row["dTm"])
        wt_data = self._load_feature_dict(sample_id, "wt_data")
        mut_data = self._load_feature_dict(sample_id, "mut_data")
        target = torch.tensor(target, dtype=torch.float32)
        return wt_data, mut_data, target


###############################################################################
# Model and Training Logic (copied from train_geodtm_no_plddt.py)
###############################################################################

class GeoDTmModel(nn.Module):
    def __init__(self, node_dim: int, n_head: int, pair_dim: int, num_layer: int):
        super().__init__()
        self.encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        mask = mask_1d.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data):
        plddt = torch.sign(torch.relu(data["fixed_embedding"][:, :, -1] - 0.7)).bool()
        atom_mask = torch.stack(
            (data["atom_mask"].bool(), plddt.unsqueeze(-1).repeat(1, 1, data["atom_mask"].shape[-1])),
            dim=0,
        ).all(dim=0)
        node_feat = self.encoder(data["dynamic_embedding"], data["pair"], atom_mask)
        res_mask = atom_mask[:, :, ATOM_CA]
        pooled = self._masked_mean(node_feat, res_mask)
        return pooled

    def forward(self, wt_data, mut_data):
        z_wt = self.encode(wt_data)
        z_mut = self.encode(mut_data)
        delta = z_mut - z_wt
        out = self.head(delta).squeeze(-1)
        return out

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    x = x.unsqueeze(0)
    diff = x.T - x
    P = torch.sigmoid(diff * regularization_strength)
    ranks = 1 + P.sum(dim=1)
    return ranks

def spearman_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_r = soft_rank(pred)
    targ_r = soft_rank(target)
    pred_r = pred_r - pred_r.mean()
    targ_r = targ_r - targ_r.mean()
    pred_r = pred_r / (pred_r.norm(p=2) + 1e-8)
    targ_r = targ_r / (targ_r.norm(p=2) + 1e-8)
    rho = (pred_r * targ_r).sum()
    return 1.0 - rho

def dtm_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    loss_spear = spearman_loss(pred, target)
    loss_mse = F.mse_loss(pred, target)
    return alpha * loss_spear + (1.0 - alpha) * loss_mse

def move_batch_to_device(batch, device):
    wt_data, mut_data, target = batch
    for d in (wt_data, mut_data):
        for k in d:
            if isinstance(d[k], torch.Tensor):
                d[k] = d[k].to(device)
    target = target.to(device)
    return wt_data, mut_data, target

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cuda"),
    alpha_loss: float = 0.5,
) -> tuple:
    is_train = optimizer is not None
    model.train(is_train)
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        wt_data, mut_data, target = move_batch_to_device(batch, device)
        pred = model(wt_data, mut_data)
        loss = dtm_loss(pred, target, alpha=alpha_loss)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = target.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = F.mse_loss(all_preds, all_targets).item()
    pred_rank = torch.argsort(torch.argsort(all_preds))
    targ_rank = torch.argsort(torch.argsort(all_targets))
    pred_rank = pred_rank.float() - pred_rank.float().mean()
    targ_rank = targ_rank.float() - targ_rank.float().mean()
    pred_rank /= (pred_rank.norm(p=2) + 1e-8)
    targ_rank /= (targ_rank.norm(p=2) + 1e-8)
    rho = (pred_rank * targ_rank).sum().item()
    return total_loss / max(n_samples, 1), mse, rho

def load_pretrained_encoder(
    model: GeoDTmModel,
    geofitness_ckpt: str,
    device: torch.device,
):
    print(f"Loading pretrained GeoFitness from {geofitness_ckpt}", flush=True)
    ckpt = torch.load(geofitness_ckpt, map_location=device)
    if isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    else:
        state = ckpt
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("pretrain_encoder."):
            new_k = k[len("pretrain_encoder.") :]
            encoder_state[new_k] = v
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print("Loaded encoder from GeoFitness. Missing:", missing, "Unexpected:", unexpected, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S4346.csv",
                        help="Training CSV (ΔTm training data)")
    parser.add_argument("--test_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S571.csv",
                        help="Test CSV (ΔTm benchmark)")
    parser.add_argument("--features_dir", type=str, default="/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/",
                        help="Directory containing per-sample folders e.g. <features_dir>/<sample_id>/wt_data/esm2.pt")
    parser.add_argument("--geofitness_ckpt", type=str, default=None, help="Path to pretrained GeoFitness .pt (optional)")
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs_frozen", type=int, default=5,
                        help="Number of epochs with encoder frozen (pretraining stage)")
    parser.add_argument("--epochs_finetune", type=int, default=50,
                        help="Number of epochs with encoder unfrozen (fine-tuning)")
    parser.add_argument("--alpha_loss", type=float, default=0.5,
                        help="Weight for Spearman vs MSE in conjugated loss")
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="geodtm_models")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(args.train_csv)
    # Column that identifies the *protein*, not the individual mutant
    full_df['protein'] = full_df['name'].apply(lambda x: x.split('_')[1])
    protein_col = "protein"
    assert protein_col in full_df.columns, f"{protein_col} not in train CSV"

    val_frac = 0.1
    proteins = full_df[protein_col].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(proteins)

    n_val_prot = max(1, int(math.ceil(len(proteins) * val_frac)))
    val_proteins = set(proteins[:n_val_prot])
    train_proteins = set(proteins[n_val_prot:])

    train_df = full_df[full_df[protein_col].isin(train_proteins)].reset_index(drop=True)
    val_df   = full_df[full_df[protein_col].isin(val_proteins)].reset_index(drop=True)

    print(f"Protein-disjoint split:")
    print(f"  Train proteins: {len(train_proteins)}, samples: {len(train_df)}")
    print(f"  Val proteins:   {len(val_proteins)}, samples: {len(val_df)}", flush=True)

    train_ds = GeoDTmDataset(train_df, args.features_dir)
    val_ds   = GeoDTmDataset(val_df,   args.features_dir)
    test_ds  = GeoDTmDataset(args.test_csv, args.features_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = GeoDTmModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
    ).to(device)

    if args.geofitness_ckpt and os.path.isfile(args.geofitness_ckpt):
        load_pretrained_encoder(model, args.geofitness_ckpt, device=device)
    else:
        print("No geofitness_ckpt provided. Training encoder from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "geodtm_best.pt")
    early_counter = 0

    print("Stage 1: Freezing encoder for rapid head optimization (GeoDTm, as in paper).", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False

    for epoch in range(1, args.epochs_frozen + 1):
        train_loss, train_mse, train_rho = run_epoch(
            model, train_loader, optimizer, device, args.alpha_loss
        )
        val_loss, val_mse, val_rho = run_epoch(
            model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss
        )
        scheduler.step(val_loss)

        print(
            f"[Frozen] Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | "
            f"Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}"
        , flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Early stopping (frozen stage).", flush=True)
                break

    print("Stage 2: Unfreezing encoder for joint fine-tuning.", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = True

    early_counter = 0
    for epoch in range(1, args.epochs_finetune + 1):
        train_loss, train_mse, train_rho = run_epoch(
            model, train_loader, optimizer, device, args.alpha_loss
        )
        val_loss, val_mse, val_rho = run_epoch(
            model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss
        )
        scheduler.step(val_loss)

        print(
            f"[Finetune] Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | "
            f"Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}"
        , flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Early stopping (fine-tune stage).", flush=True)
                break

    print(f"Loading best model from {best_path} for test evaluation (S571).", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_mse, test_rho = run_epoch(
        model, test_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss
    )
    print(
        f"Test (S571) | Loss {test_loss:.4f} | MSE {test_mse:.4f} | Spearman {test_rho:.3f}"
    , flush=True)

if __name__ == "__main__":
    main()