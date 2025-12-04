# -*- coding: utf-8 -*-
# Radius ablation script — fixed to make ablations effective and robust.
# Key fixes:
#  - Zero ablated fixed features explicitly in the dataset.
#  - Deterministic fixed_dim = 7 + int(use_pH) + int(use_plddt).
#  - Inject fixed features into node embedding via fixed_proj + input_proj so
#    zeroed fixed inputs are actually removed from model computation.
#  - Safeguards for pLDDT padding and clear shape checks.

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
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_dTm_3D")))
from model import PretrainEncoder, ATOM_CA

###############################################################################
# Radius-based Ablation Dataset
###############################################################################

class GeoDTmRadiusAblationDataset(Dataset):
    def __init__(
        self, csv_or_df, features_dir, radius=0,
        use_fixed_embedding=True, use_dynamic_embedding=True,
        use_pair=True, use_atom_mask=True, use_pH=True, use_plddt=True
    ):
        super().__init__()
        if isinstance(csv_or_df, pd.DataFrame):
            self.df = csv_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_or_df)
        self.features_dir = features_dir
        self.radius = radius
        self.use_fixed_embedding = use_fixed_embedding
        self.use_dynamic_embedding = use_dynamic_embedding
        self.use_pair = use_pair
        self.use_atom_mask = use_atom_mask
        self.use_pH = use_pH
        self.use_plddt = use_plddt
        ph_candidates = [c for c in self.df.columns if c.lower() == "ph"]
        self.ph_col = ph_candidates[0] if ph_candidates else None

    def _extract_window(self, tensor, mut_pos):
        """
        Crop tensor [L,...] to [window,...] w.r.t mutation position mut_pos and self.radius.
        If at sequence edge, window is truncated; no padding.
        """
        L = tensor.shape[0]
        start = max(0, mut_pos - self.radius)
        end = min(L, mut_pos + self.radius + 1)
        # For pair features: [L, L, ...]
        if tensor.dim() >= 2 and tensor.shape[0] == L and tensor.shape[1] == L:
            return tensor[start:end, start:end]
        else:
            return tensor[start:end]

    def _windowed_feature_dict(self, row, variant: str):
        sample_id = str(row["name"])
        folder = os.path.join(self.features_dir, sample_id, variant)
        info_path = os.path.join(self.features_dir, sample_id, "mut_info.csv")
        mut_pos = None
        # Always get mut_pos from mut_info.csv if present
        if os.path.exists(info_path):
            info = pd.read_csv(info_path, index_col=0)
            if "mut_pos" in info.columns:
                try:
                    mut_pos = int(info.loc["test", "mut_pos"])
                except Exception:
                    mut_pos = None
        if mut_pos is None:
            raise ValueError(f"Could not determine mut_pos for {sample_id}")

        # Loading raw features
        d_emb = torch.load(os.path.join(folder, "esm2.pt")).float()
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()
        if fixed.dim() == 1:
            fixed = fixed.unsqueeze(-1)

        pair = torch.load(os.path.join(folder, "pair.pt")).float()
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        atom_mask = coord_data["pos14_mask"].all(dim=-1)

        # pH feature
        ph_val = 7.0
        if self.use_pH and self.ph_col is not None:
            ph_val = float(row[self.ph_col])
            ph_val = max(0.0, min(11.0, ph_val))
        ph_feat = torch.full((d_emb.shape[0], 1), ph_val, dtype=torch.float32)

        # pLDDT feature
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        plddt = torch.tensor(pkl["plddt"], dtype=torch.float32)
        if plddt.dim() != 1:
            plddt = plddt.view(-1)
        # Protect against empty plddt vector
        if plddt.numel() == 0:
            plddt = torch.zeros((d_emb.shape[0],), dtype=torch.float32)
        # shape alignment
        if plddt.shape[0] > d_emb.shape[0]:
            plddt = plddt[:d_emb.shape[0]]
        elif plddt.shape[0] < d_emb.shape[0]:
            pad_val = plddt[-1] if plddt.numel() > 0 else torch.tensor(0.0, dtype=plddt.dtype)
            pad = pad_val.repeat(d_emb.shape[0] - plddt.shape[0])
            plddt = torch.cat([plddt, pad], dim=0)
        plddt = plddt / 100.0
        plddt_feat = plddt.unsqueeze(-1)

        # Respect ablation flags: zero out ablated components explicitly.
        if not self.use_fixed_embedding:
            fixed = torch.zeros_like(fixed)
        if not self.use_pH:
            ph_feat = torch.zeros_like(ph_feat)
        if not self.use_plddt:
            plddt_feat = torch.zeros_like(plddt_feat)

        # Merge features: always same column order and width: [7 physchem] + [pH?] + [pLDDT?]
        fixed_full = torch.cat([fixed, ph_feat, plddt_feat], dim=-1)

        # Windowed crop each feature
        d_emb = self._extract_window(d_emb if self.use_dynamic_embedding else torch.zeros_like(d_emb), mut_pos)
        fixed_full = self._extract_window(fixed_full, mut_pos)
        pair = self._extract_window(pair if self.use_pair else torch.zeros_like(pair), mut_pos)
        atom_mask = self._extract_window(atom_mask if self.use_atom_mask else torch.ones_like(atom_mask, dtype=torch.bool), mut_pos)

        # Windowed mutation position [winlen]
        mask_len = d_emb.shape[0]
        mut_pos_mask = torch.zeros(mask_len, dtype=torch.float32)
        mut_pos_center = mut_pos - max(0, mut_pos - self.radius)
        if 0 <= mut_pos_center < mask_len:
            mut_pos_mask[mut_pos_center] = 1.0

        feature_dict = dict(
            dynamic_embedding=d_emb,
            fixed_embedding=fixed_full,
            pair=pair,
            atom_mask=atom_mask,
            mut_pos=mut_pos_mask
        )
        return feature_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = float(row["dTm"])
        wt_data = self._windowed_feature_dict(row, "wt_data")
        mut_data = self._windowed_feature_dict(row, "mut_data")
        return wt_data, mut_data, torch.tensor(target, dtype=torch.float32)

###############################################################################
# Model (updated to ingest fixed features)
###############################################################################

class GeoDTmAblationModel(nn.Module):
    def __init__(self, node_dim, n_head, pair_dim, num_layer, fixed_dim):
        super().__init__()
        # Reuse the PretrainEncoder blocks (same architecture) but we will
        # manually prepare the initial node embedding to include fixed features
        self.encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1),
        )
        self.fixed_dim = fixed_dim
        self.node_dim = node_dim

        # Project concatenated [node_dim_from_esm2, fixed_dim] -> node_dim
        if fixed_dim > 0:
            # fixed_proj consumes the fixed vector (per-residue) and outputs node_dim
            self.fixed_proj = nn.Sequential(
                nn.Linear(fixed_dim, node_dim),
                nn.LeakyReLU(),
                nn.Linear(node_dim, node_dim)
            )
            # input_proj maps (node_dim + node_dim) -> node_dim after fixed_proj
            self.input_proj = nn.Linear(node_dim + node_dim, node_dim)
        else:
            # If no fixed features are used, no proj is necessary; keep identity-like projection
            self.input_proj = nn.Identity()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        mask = mask_1d.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data):
        # Dynamic embedding: ensure batch dim
        dyn_emb = data["dynamic_embedding"]
        if dyn_emb.dim() == 2:
            dyn_emb = dyn_emb.unsqueeze(0)  # [1, L, 1280]
        # Pair: [1, L, L, 7] or [N, L, L, 7]
        pair = data["pair"]
        if pair.dim() == 3:
            pair = pair.unsqueeze(0)
        # Atom mask: [1, L, 14]
        atom_mask = data["atom_mask"]
        if atom_mask.dim() == 2:
            atom_mask = atom_mask.unsqueeze(0)

        # 1) esm2 -> node_dim
        dyn_node = self.encoder.esm2_transform(dyn_emb)  # [N, L, node_dim]

        # 2) fixed features (if present)
        fixed_full = data.get("fixed_embedding", None)
        if fixed_full is None:
            fixed_proj = torch.zeros_like(dyn_node)
        else:
            if fixed_full.dim() == 2:
                fixed_full = fixed_full.unsqueeze(0)  # [1, L, N_fixed]
            # Sanity check
            if fixed_full.shape[-1] != self.fixed_dim:
                raise RuntimeError(
                    f"fixed_full last-dim ({fixed_full.shape[-1]}) != model.fixed_dim ({self.fixed_dim}). "
                    "Construct the model with fixed_dim = 7 + int(use_pH) + int(use_plddt)."
                )
            if self.fixed_dim > 0:
                fixed_proj = self.fixed_proj(fixed_full)  # [N, L, node_dim]
            else:
                fixed_proj = torch.zeros_like(dyn_node)

        # 3) combine dynamic node and fixed projection
        if isinstance(self.input_proj, nn.Identity):
            embedding = dyn_node
        else:
            embedding = self.input_proj(torch.cat([dyn_node, fixed_proj], dim=-1))  # [N, L, node_dim]

        # 4) pair encoding and run blocks
        pair_enc = self.encoder.pair_encoder(pair)
        for block in self.encoder.blocks:
            embedding, pair_enc = block(embedding, pair_enc, atom_mask[:, :, ATOM_CA])

        # embedding [N, L, node_dim]
        res_mask = atom_mask[:, :, ATOM_CA]
        pooled = self._masked_mean(embedding, res_mask)
        return pooled

    def forward(self, wt_data, mut_data):
        z_wt = self.encode(wt_data)
        z_mut = self.encode(mut_data)
        delta = z_mut - z_wt
        out = self.head(delta).squeeze(-1)
        return out

###############################################################################
# Training Utils - same as before
###############################################################################

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    x = x.reshape(-1)
    diff = x.unsqueeze(0).T - x.unsqueeze(0)
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

###############################################################################
# Main Radius Ablation Script
###############################################################################

def radius_suffix(args):
    return f"radius{args.radius:02d}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S4346.csv")
    parser.add_argument("--test_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S571.csv")
    parser.add_argument("--features_dir", type=str, default="/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/")
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=50)
    parser.add_argument("--alpha_loss", type=float, default=0.5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="geodtm_models_ablation_radius")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--job_id", type=str, default="radius_job")
    parser.add_argument("--radius", type=int, default=0, help="Radius (window size) for ablation. Use 0 for mut-site only, 1 for mut±1, etc.")
    # Feature switches
    parser.add_argument("--use_fixed_embedding", action="store_true", default=True)
    parser.add_argument("--no_fixed_embedding", action="store_false", dest="use_fixed_embedding")
    parser.add_argument("--use_dynamic_embedding", action="store_true", default=True)
    parser.add_argument("--no_dynamic_embedding", action="store_false", dest="use_dynamic_embedding")
    parser.add_argument("--use_pair", action="store_true", default=True)
    parser.add_argument("--no_pair", action="store_false", dest="use_pair")
    parser.add_argument("--use_atom_mask", action="store_true", default=True)
    parser.add_argument("--no_atom_mask", action="store_false", dest="use_atom_mask")
    parser.add_argument("--use_pH", action="store_true", default=True)
    parser.add_argument("--no_pH", action="store_false", dest="use_pH")
    parser.add_argument("--use_plddt", action="store_true", default=True)
    parser.add_argument("--no_plddt", action="store_false", dest="use_plddt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()
    print(f"using seed:{args.seed}", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed or 0)
    torch.cuda.manual_seed_all(args.seed or 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = radius_suffix(args)
    best_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_best_{suffix}_{args.seed}.pt")
    test_csv_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_test_predictions_{suffix}_{args.seed}.csv")

    # --- Data ---
    full_df = pd.read_csv(args.train_csv)
    full_df['protein'] = full_df['name'].apply(lambda x: x.split('_')[1])
    protein_col = "protein"

    proteins = full_df[protein_col].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(proteins)
    val_frac = 0.1
    n_val_prot = max(1, int(math.ceil(len(proteins) * val_frac)))
    val_proteins = set(proteins[:n_val_prot])
    train_proteins = set(proteins[n_val_prot:])
    train_df = full_df[full_df[protein_col].isin(train_proteins)].reset_index(drop=True)
    val_df   = full_df[full_df[protein_col].isin(val_proteins)].reset_index(drop=True)
    print(f"Protein-disjoint split: Train proteins/samples: {len(train_proteins)}/{len(train_df)}, Val: {len(val_proteins)}/{len(val_df)}", flush=True)

    # Datasets
    train_ds = GeoDTmRadiusAblationDataset(
        train_df, args.features_dir, radius=args.radius,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    val_ds = GeoDTmRadiusAblationDataset(
        val_df, args.features_dir, radius=args.radius,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    test_ds = GeoDTmRadiusAblationDataset(
        args.test_csv, args.features_dir, radius=args.radius,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # --- Compute fixed feature length deterministically ---
    # fixed_full = [7 physchem] + [pH?] + [pLDDT?] always (we zero content for ablations)
    fixed_dim = 7 + int(bool(args.use_pH)) + int(bool(args.use_plddt))
    print(f"[Info] Using fixed_dim = {fixed_dim} (7 physchem + pH:{int(args.use_pH)} + pLDDT:{int(args.use_plddt)})", flush=True)

    # --- Model ---
    model = GeoDTmAblationModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
        fixed_dim=fixed_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, verbose=True,
    )

    # Freeze encoder (pretrain) then finetune
    print("Stage 1: Freezing encoder for head optimization.", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    best_val_loss = float("inf")
    early_counter = 0
    for epoch in range(1, args.epochs_frozen + 1):
        train_loss, train_mse, train_rho = run_epoch(model, train_loader, optimizer, device, args.alpha_loss)
        val_loss, val_mse, val_rho = run_epoch(model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss)
        scheduler.step(val_loss)
        print(f"[Frozen] Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}", flush=True)
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
        train_loss, train_mse, train_rho = run_epoch(model, train_loader, optimizer, device, args.alpha_loss)
        val_loss, val_mse, val_rho = run_epoch(model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss)
        scheduler.step(val_loss)
        print(f"[Finetune] Epoch {epoch:03d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val MSE {val_mse:.4f} | Val Spearman {val_rho:.3f}", flush=True)
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

    # Test set and output CSV
    print("Generating test-set predictions and saving CSV...", flush=True)
    model.eval()
    test_names = []
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            wt_data, mut_data, target = move_batch_to_device(batch, device)
            pred = model(wt_data, mut_data)
            sample_name = test_ds.df.iloc[i]["name"]
            test_names.append(sample_name)
            test_preds.append(float(pred.cpu().item()))
            test_targets.append(float(target.cpu().item()))
    pd.DataFrame({
        "name": test_names,
        "model_score": test_preds,
        "true_label": test_targets,
    }).to_csv(test_csv_path, index=False)
    print(f"Saved test predictions to: {test_csv_path}", flush=True)

if __name__ == "__main__":
    main()