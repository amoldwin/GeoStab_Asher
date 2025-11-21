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
# Assumes this script is in the same project where the original GeoStab model.py lives
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_dTm_3D")))

from model import PretrainModel  # "true" GeoDTm / ddG/dTm model


###############################################################################
# GeoDTm Dataset using true GeoStab PretrainModel
###############################################################################

class GeoDTmTrueDataset(Dataset):
    """
    Dataset for GeoDTm using the full GeoStab PretrainModel.

    Expects per-sample folders like:
        <features_dir>/<sample_id>/wt_data/esm2.pt
        <features_dir>/<sample_id>/wt_data/fixed_embedding.pt  (7-d physchem)
        <features_dir>/<sample_id>/wt_data/pair.pt             (L x L x 7)
        <features_dir>/<sample_id>/wt_data/coordinate.pt       (pos14, pos14_mask)
        <features_dir>/<sample_id>/wt_data/wt_esmf.pkl         (ESMFold output with 'plddt')

        and similarly for mut_data/ (mut_esmf.pkl)

    We construct:
      - dynamic_embedding: [L, 1280] (esm2)
      - fixed_embedding:   [L, 9]    = [7 physchem, pH, pLDDT]
      - pair:              [L, L, 7]
      - atom_mask:         [L, 14] from coordinate["pos14_mask"]
      - mut_pos:           [L] from mut_info.csv ("mut_pos"), 1 at mutation site
      - target:            scalar ΔTm from CSV
    """

    def __init__(self, csv_or_df, features_dir: str):
        super().__init__()
        if isinstance(csv_or_df, pd.DataFrame):
            self.df = csv_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_or_df)

        self.features_dir = features_dir

        assert "name" in self.df.columns, "CSV must contain a 'name' column."
        assert "dTm" in self.df.columns, "CSV must contain a 'dTm' column."

        # detect pH column
        ph_candidates = [c for c in self.df.columns if c.lower() == "ph"]
        self.ph_col = ph_candidates[0] if ph_candidates else None
        if self.ph_col is None:
            print("[Warning] No pH column found in CSV; defaulting to pH=7.0")

    def _load_feature_dict(self, row, variant: str):
        sample_id = str(row["name"])
        folder = os.path.join(self.features_dir, sample_id, variant)
        out = {}

        # 1) ESM2 embeddings
        dyn = torch.load(os.path.join(folder, "esm2.pt")).float()  # [L, 1280]
        L_dyn = dyn.shape[0]

        # 2) 7-d physchem fixed features
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()  # [L, 7]
        if fixed.dim() == 1:
            fixed = fixed.unsqueeze(-1)
        L_fixed = fixed.shape[0]
        assert L_fixed == L_dyn, f"Length mismatch: esm2 {L_dyn} vs fixed_embedding {L_fixed}"
        L = L_fixed

        # 3) pH feature (scalar -> [L,1])
        if self.ph_col is not None:
            ph_val = float(row[self.ph_col])
        else:
            ph_val = 7.0
        # clamp like the paper (0–11)
        ph_val = max(0.0, min(11.0, ph_val))
        ph_feat = torch.full((L, 1), ph_val, dtype=torch.float32)

        # 4) pLDDT from ESMFold pickle (wt_esmf.pkl / mut_esmf.pkl)
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        plddt_raw = torch.tensor(pkl["plddt"], dtype=torch.float32)

        # Normalize shapes:
        #  (L,)           -> already per-residue pLDDT
        #  (L, 37)        -> logits over 37 bins -> expected value in [0,100]
        #  (1, L)/(L, 1)  -> just squeeze to (L,)
        if plddt_raw.dim() == 1:
            pass
        elif plddt_raw.dim() == 2:
            if plddt_raw.shape[1] == 37:
                Lp, K = plddt_raw.shape
                bins = torch.arange(K, dtype=plddt_raw.dtype)  # 0..36
                probs = plddt_raw.softmax(-1)                  # (L,37)
                plddt_raw = (probs * bins).sum(-1) * (100.0 / 36.0)  # (L,)
            elif plddt_raw.shape[0] == 1 or plddt_raw.shape[1] == 1:
                plddt_raw = plddt_raw.view(-1)
            else:
                raise ValueError(f"Unexpected 2D pLDDT shape {plddt_raw.shape} in {pkl_path}")
        else:
            raise ValueError(
                f"Unexpected pLDDT tensor rank {plddt_raw.dim()} with shape {plddt_raw.shape} in {pkl_path}"
            )

        # length-align pLDDT to fixed_embedding length
        L_plddt = plddt_raw.shape[0]
        if L_plddt > L:
            plddt_raw = plddt_raw[:L]
        elif L_plddt < L:
            print(
                f"[GeoDTmTrueDataset] WARNING: pLDDT shorter ({L_plddt}) than "
                f"fixed_embedding ({L}) in {folder}. Padding.",
                flush=True,
            )
            if L_plddt > 0:
                pad_val = plddt_raw[-1]
            else:
                pad_val = torch.tensor(0.0, dtype=plddt_raw.dtype)
            pad = pad_val.repeat(L - L_plddt)
            plddt_raw = torch.cat([plddt_raw, pad], dim=0)

        # normalize pLDDT to [0,1]
        plddt = plddt_raw / 100.0  # (L,)

        # 5) final fixed_embedding: [7 physchem, pH, pLDDT] -> [L, 9]
        fixed_full = torch.cat([fixed, ph_feat, plddt.unsqueeze(-1)], dim=-1)

        # 6) pair features
        pair = torch.load(os.path.join(folder, "pair.pt")).float()

        # 7) atom mask from coordinate.pt -> [L,14]
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        atom_mask = coord_data["pos14_mask"].all(dim=-1).float()

        # 8) mutation position mask [L]
        info_path = os.path.join(self.features_dir, sample_id, "mut_info.csv")
        mut_pos_mask = torch.zeros(L, dtype=torch.float32)
        if os.path.exists(info_path):
            info = pd.read_csv(info_path, index_col=0)
            if "mut_pos" in info.columns:
                mut_pos = info.loc["test", "mut_pos"]
                try:
                    mut_pos = int(mut_pos)
                    if 0 <= mut_pos < L:
                        mut_pos_mask[mut_pos] = 1.0
                except Exception:
                    pass

        out["dynamic_embedding"] = dyn
        out["fixed_embedding"] = fixed_full
        out["pair"] = pair
        out["atom_mask"] = atom_mask
        out["mut_pos"] = mut_pos_mask
        return out

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = float(row["dTm"])
        wt_data = self._load_feature_dict(row, "wt_data")
        mut_data = self._load_feature_dict(row, "mut_data")
        target = torch.tensor(target, dtype=torch.float32)
        return wt_data, mut_data, target

###############################################################################
# Loss & metrics (reuse your dtm_loss + Spearman machinery)
###############################################################################

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Stable soft rank that works even when x is a scalar or has batch_size=1.
    """
    # Flatten to a 1D vector [N]
    x = x.reshape(-1)

    # Pairwise differences: [N, N]
    diff = x.unsqueeze(0).T - x.unsqueeze(0)

    # Smooth pairwise comparison matrix
    P = torch.sigmoid(diff * regularization_strength)  # [N, N]

    # Soft rank: 1 + sum_j sigma(x_i - x_j)
    ranks = 1 + P.sum(dim=1)  # [N]

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
    # Ensure 1D vectors
    pred = pred.reshape(-1)
    target = target.reshape(-1)

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
        pred = pred.reshape(-1)
        target = target.reshape(-1)

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

    # Simple Spearman via double argsort
    pred_rank = torch.argsort(torch.argsort(all_preds))
    targ_rank = torch.argsort(torch.argsort(all_targets))
    pred_rank = pred_rank.float() - pred_rank.float().mean()
    targ_rank = targ_rank.float() - targ_rank.float().mean()
    pred_rank /= (pred_rank.norm(p=2) + 1e-8)
    targ_rank /= (targ_rank.norm(p=2) + 1e-8)
    rho = (pred_rank * targ_rank).sum().item()

    return total_loss / max(n_samples, 1), mse, rho


###############################################################################
# Loading GeoFitness pretraining (optional)
###############################################################################

def load_pretrained_geofitness(
    model: PretrainModel,
    geofitness_ckpt: str,
    device: torch.device,
):
    """
    Load a GeoFitness checkpoint into the PretrainModel.

    This is flexible: it tries:
      - ckpt["model"], if present
      - otherwise assumes ckpt is directly a state_dict
    and loads with strict=False so it can ignore missing heads, etc.
    """
    print(f"Loading pretrained GeoFitness from {geofitness_ckpt}", flush=True)
    ckpt = torch.load(geofitness_ckpt, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded GeoFitness. Missing:", missing, "Unexpected:", unexpected, flush=True)


###############################################################################
# Main training logic
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S4346.csv",
        help="Training CSV (ΔTm training data)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S571.csv",
        help="Test CSV (ΔTm benchmark)",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/",
        help="Directory with per-sample folders <features_dir>/<sample_id>/wt_data/...",
    )
    parser.add_argument(
        "--geofitness_ckpt",
        type=str,
        default=None,
        help="Path to pretrained GeoFitness .pt (optional)",
    )
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--dms_node_dim", type=int, default=32)
    parser.add_argument("--dms_pair_dim", type=int, default=32)
    parser.add_argument("--dms_n_head", type=int, default=4)
    parser.add_argument("--dms_num_layer", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)  # variable length, so keep 1
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--epochs_frozen",
        type=int,
        default=5,
        help="Epochs with PretrainEncoder frozen",
    )
    parser.add_argument(
        "--epochs_finetune",
        type=int,
        default=50,
        help="Epochs with full model trainable",
    )
    parser.add_argument(
        "--alpha_loss",
        type=float,
        default=0.5,
        help="Weight for Spearman vs MSE in combined loss",
    )
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="geodtm_models_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Protein-disjoint split (as before) ---
    full_df = pd.read_csv(args.train_csv)
    full_df["protein"] = full_df["name"].apply(lambda x: x.split("_")[1])
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
    val_df = full_df[full_df[protein_col].isin(val_proteins)].reset_index(drop=True)

    print("Protein-disjoint split:")
    print(f"  Train proteins: {len(train_proteins)}, samples: {len(train_df)}")
    print(f"  Val proteins:   {len(val_proteins)}, samples: {len(val_df)}", flush=True)

    train_ds = GeoDTmTrueDataset(train_df, args.features_dir)
    val_ds = GeoDTmTrueDataset(val_df, args.features_dir)
    test_ds = GeoDTmTrueDataset(args.test_csv, args.features_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- True GeoDTm model ---
    model = PretrainModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
        dms_node_dim=args.dms_node_dim,
        dms_num_layer=args.dms_num_layer,
        dms_n_head=args.dms_n_head,
        dms_pair_dim=args.dms_pair_dim,
    ).to(device)

    if args.geofitness_ckpt and os.path.isfile(args.geofitness_ckpt):
        load_pretrained_geofitness(model, args.geofitness_ckpt, device=device)
    else:
        print("No geofitness_ckpt provided. Training from scratch.", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "geodtm_best.pt")
    early_counter = 0

    # --- Stage 1: freeze DMS pretrain encoder (pretrain_encoder) ---
    print("Stage 1: Freezing pretrain_encoder (DMS module).", flush=True)
    for name, p in model.named_parameters():
        if name.startswith("pretrain_encoder."):
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
            f"Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Early stopping (frozen stage).", flush=True)
                break

    # --- Stage 2: unfreeze everything for joint fine-tuning ---
    print("Stage 2: Unfreezing full model for joint fine-tuning.", flush=True)
    for p in model.parameters():
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
            f"Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}",
            flush=True,
        )

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
        f"Test (S571) | Loss {test_loss:.4f} | MSE {test_mse:.4f} | Spearman {test_rho:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
