# -*- coding: utf-8 -*-
# NOTE: This file is a lightly modified copy of the repository version.
# Fixes:
#  - Make fixed_dim deterministic and consistent with how dataset constructs fixed_full:
#      fixed_full = [7 physchem dims] + [pH?] + [pLDDT?]
#    (previous detection logic could pick up a different interpretation of flags
#     and cause a shape mismatch at runtime).
#  - Construct model after datasets so optimizer includes all parameters.
#  - Add explicit shape checks and clear error messages for easier debugging.
#  - Keep previous ablation behavior: dataset zeroes ablated features; model injects fixed features.
#  - Add --delta_struct: replace mutant structure features with Δ(structure) = mutant − WT
#    (pair features and optional pLDDT column), preserving masks and non-structure features.
#  - Add SpearmanQueue to accumulate Spearman signal across steps so Spearman loss
#    remains meaningful with batch_size=1.

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
# Ablation Dataset
###############################################################################

class GeoDTmAblationDataset(Dataset):
    def __init__(
        self, csv_or_df, features_dir, use_fixed_embedding=True, use_dynamic_embedding=True,
        use_pair=True, use_atom_mask=True, use_pH=True, use_plddt=True
    ):
        super().__init__()
        if isinstance(csv_or_df, pd.DataFrame):
            self.df = csv_or_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_or_df)
        self.features_dir = features_dir
        self.use_fixed_embedding = use_fixed_embedding
        self.use_dynamic_embedding = use_dynamic_embedding
        self.use_pair = use_pair
        self.use_atom_mask = use_atom_mask
        self.use_pH = use_pH
        self.use_plddt = use_plddt
        # Infer pH column
        ph_candidates = [c for c in self.df.columns if c.lower() == "ph"]
        self.ph_col = ph_candidates[0] if ph_candidates else None

    def _load_feature_dict(self, row, variant: str):
        sample_id = str(row["name"])
        folder = os.path.join(self.features_dir, sample_id, variant)
        L = None

        feature_dict = {}

        # Dynamic embedding (ESM2)
        d_emb = torch.load(os.path.join(folder, "esm2.pt")).float()
        L = d_emb.shape[0]
        feature_dict["dynamic_embedding"] = d_emb if self.use_dynamic_embedding else torch.zeros_like(d_emb)
        
        # Fixed embedding (7 physchem dims)
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()
        fixed = fixed if self.use_fixed_embedding else torch.zeros_like(fixed)
        if fixed.dim() == 1:
            fixed = fixed.unsqueeze(-1)
        feature_dict["fixed_embedding"] = fixed

        # Pair features
        pair = torch.load(os.path.join(folder, "pair.pt")).float()
        feature_dict["pair"] = pair if self.use_pair else torch.zeros_like(pair)

        # Atom mask from coordinate.pt
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        atom_mask = coord_data["pos14_mask"].all(dim=-1)  # already bool
        feature_dict["atom_mask"] = atom_mask if self.use_atom_mask else torch.ones_like(atom_mask, dtype=torch.bool)

        # pH feature (expand to [L, 1])
        ph_val = 7.0
        if self.use_pH and self.ph_col is not None:
            ph_val = float(row[self.ph_col])
            ph_val = max(0.0, min(11.0, ph_val))
        ph_feat = torch.full((L, 1), ph_val, dtype=torch.float32)
        feature_dict["pH"] = ph_feat if self.use_pH else torch.zeros_like(ph_feat)

        # pLDDT feature (expand to [L, 1])
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        plddt = torch.tensor(pkl["plddt"], dtype=torch.float32)
        if plddt.dim() != 1:
            plddt = plddt.view(-1)
        # shape alignment
        L_plddt = plddt.shape[0]
        if L_plddt > L:
            plddt = plddt[:L]
        elif L_plddt < L:
            pad_val = plddt[-1] if L_plddt > 0 else torch.tensor(0.0, dtype=plddt.dtype)
            pad = pad_val.repeat(L - L_plddt)
            plddt = torch.cat([plddt, pad], dim=0)
        plddt = plddt / 100.0  # Normalize to [0,1]
        plddt_feat = plddt.unsqueeze(-1)
        feature_dict["plddt"] = plddt_feat if self.use_plddt else torch.zeros_like(plddt_feat)

        # Merge fixed_embedding, pH, pLDDT: [L, N_fixed]
        feature_dict["fixed_full"] = torch.cat(
            [feature_dict["fixed_embedding"], feature_dict["pH"], feature_dict["plddt"]], dim=-1
        )

        # Mutation position mask (optional, always present in features for compatibility)
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
        feature_dict["mut_pos"] = mut_pos_mask

        return feature_dict

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = float(row["dTm"])
        wt_data = self._load_feature_dict(row, "wt_data")
        mut_data = self._load_feature_dict(row, "mut_data")
        return wt_data, mut_data, torch.tensor(target, dtype=torch.float32)

###############################################################################
# Model
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
        # We will use the encoder's esm2_transform (same weights as PretrainEncoder)
        dyn_node = self.encoder.esm2_transform(dyn_emb)  # [N, L, node_dim]

        # 2) fixed features (if present)
        fixed_full = data.get("fixed_full", None)
        if fixed_full is None:
            # tolerate older datasets that don't provide fixed_full
            fixed_proj = torch.zeros_like(dyn_node)
        else:
            if fixed_full.dim() == 2:
                fixed_full = fixed_full.unsqueeze(0)  # [1, L, N_fixed]
            # Sanity check: dataset vs model bookkeeping
            sample_fixed_dim = fixed_full.shape[-1]
            if sample_fixed_dim != self.fixed_dim:
                # Instead of hard failing, give a clear diagnostic and adapt if possible.
                raise RuntimeError(
                    f"fixed_full last-dim ({sample_fixed_dim}) != model.fixed_dim ({self.fixed_dim}). "
                    "This indicates a mismatch between the dataset's fixed vector length and the model construction. "
                    "The model must be created with a fixed_dim equal to the length of the dataset's fixed_full "
                    "(7 physchem + optional pH + optional pLDDT). "
                    "Recommended fix: create the model AFTER datasets and set fixed_dim = 7 + int(use_pH) + int(use_plddt)."
                )
            # Project fixed features into node_dim
            if self.fixed_dim > 0:
                fixed_proj = self.fixed_proj(fixed_full)  # [N, L, node_dim]
            else:
                fixed_proj = torch.zeros_like(dyn_node)

        # 3) combine dynamic node and fixed projection
        if isinstance(self.input_proj, nn.Identity):
            embedding = dyn_node
        else:
            # concat along feature dim and project
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
# Training Utils
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

class SpearmanQueue:
    """A fixed-size FIFO queue of past (pred, target) scalars/vectors to accumulate Spearman signal."""
    def __init__(self, capacity: int = 256, device: torch.device | None = None):
        self.capacity = max(0, int(capacity))
        self.device = device
        self.preds = torch.empty(0, device=device)
        self.targets = torch.empty(0, device=device)

    def size(self) -> int:
        return int(self.preds.numel())

    def add(self, pred: torch.Tensor, target: torch.Tensor):
        # Store detached history to avoid backprop through past steps
        pred = pred.detach().reshape(-1)
        target = target.detach().reshape(-1)
        if self.device is not None:
            pred = pred.to(self.device)
            target = target.to(self.device)
        self.preds = torch.cat([self.preds, pred], dim=0)
        self.targets = torch.cat([self.targets, target], dim=0)
        # Trim to capacity (FIFO)
        excess = self.preds.numel() - self.capacity
        if excess > 0:
            self.preds = self.preds[excess:]
            self.targets = self.targets[excess:]

    def spearman_with_current(self, curr_pred: torch.Tensor, curr_target: torch.Tensor) -> torch.Tensor:
        # Combine history (constants) + current sample (with grad)
        curr_pred = curr_pred.reshape(-1)
        curr_target = curr_target.reshape(-1)
        all_pred = torch.cat([self.preds, curr_pred], dim=0)
        all_target = torch.cat([self.targets, curr_target], dim=0)
        if all_pred.numel() < 2:
            # Not enough samples to form a correlation; return neutral contribution
            return curr_pred.new_tensor(0.0)
        return spearman_loss(all_pred, all_target)

def move_batch_to_device(batch, device):
    wt_data, mut_data, target = batch
    for d in (wt_data, mut_data):
        for k in d:
            if isinstance(d[k], torch.Tensor):
                d[k] = d[k].to(device)
    target = target.to(device)
    return wt_data, mut_data, target

def apply_delta_struct(wt_data, mut_data, use_pair: bool, use_plddt: bool):
    """
    Replace mutant structure features with Δ(structure) = mutant − WT.
    - pair: mut_pair = mut_pair - wt_pair (if enabled)
    - fixed_full: last column is pLDDT when enabled -> mutate last column to ΔpLDDT
      Physchem and pH columns remain unchanged.
    Atom masks are preserved to remain valid attention masks.
    """
    # Δ pair features
    if use_pair and ("pair" in wt_data) and ("pair" in mut_data):
        if wt_data["pair"].shape == mut_data["pair"].shape:
            mut_data["pair"] = mut_data["pair"] - wt_data["pair"]

    # Δ pLDDT if present (always last column in fixed_full when use_plddt=True)
    if use_plddt and ("fixed_full" in wt_data) and ("fixed_full" in mut_data):
        if wt_data["fixed_full"].shape[-1] >= 1 and mut_data["fixed_full"].shape == wt_data["fixed_full"].shape:
            mut_ff = mut_data["fixed_full"].clone()
            wt_ff = wt_data["fixed_full"]
            mut_ff[..., -1] = mut_ff[..., -1] - wt_ff[..., -1]
            mut_data["fixed_full"] = mut_ff

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cuda"),
    alpha_loss: float = 0.5,
    delta_struct: bool = False,
    use_pair: bool = True,
    use_plddt: bool = True,
    spearman_queue: SpearmanQueue | None = None,
) -> tuple:
    is_train = optimizer is not None
    model.train(is_train)

    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        wt_data, mut_data, target = move_batch_to_device(batch, device)

        # Apply Δ(structure) to mutant channel if requested
        if delta_struct:
            apply_delta_struct(wt_data, mut_data, use_pair=use_pair, use_plddt=use_plddt)

        pred = model(wt_data, mut_data)
        pred_vec = pred.reshape(-1)
        targ_vec = target.reshape(-1)

        # Accumulated Spearman if a queue is provided; otherwise per-batch Spearman.
        if spearman_queue is not None:
            loss_spear = spearman_queue.spearman_with_current(pred_vec, targ_vec)
        else:
            loss_spear = spearman_loss(pred_vec, targ_vec)

        loss_mse = F.mse_loss(pred_vec, targ_vec)
        loss = alpha_loss * loss_spear + (1.0 - alpha_loss) * loss_mse

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Push current sample into the queue AFTER update
            if spearman_queue is not None:
                spearman_queue.add(pred_vec, targ_vec)

        bs = targ_vec.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = F.mse_loss(all_preds, all_targets).item()
    # Simple Spearman via double argsort (for reporting)
    pred_rank = torch.argsort(torch.argsort(all_preds))
    targ_rank = torch.argsort(torch.argsort(all_targets))
    pred_rank = pred_rank.float() - pred_rank.float().mean()
    targ_rank = targ_rank.float() - targ_rank.float().mean()
    pred_rank /= (pred_rank.norm(p=2) + 1e-8)
    targ_rank /= (targ_rank.norm(p=2) + 1e-8)
    rho = (pred_rank * targ_rank).sum().item()
    return total_loss / max(n_samples, 1), mse, rho

###############################################################################
# Main Ablation Script
###############################################################################

def ablation_suffix(args):
    ablated = []
    if not args.use_fixed_embedding: ablated.append("no_fixed")
    if not args.use_dynamic_embedding: ablated.append("no_demb")
    if not args.use_pair: ablated.append("no_pair")
    if not args.use_atom_mask: ablated.append("no_atommask")
    if not args.use_pH: ablated.append("no_pH")
    if not args.use_plddt: ablated.append("no_plddt")
    if args.delta_struct: ablated.append("delta_struct")
    return "_".join(ablated) if ablated else "all_inputs"

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
    parser.add_argument("--epochs_frozen", type=int, default=0)
    parser.add_argument("--epochs_finetune", type=int, default=50)
    parser.add_argument("--alpha_loss", type=float, default=0.5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="geodtm_models_ablation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--job_id", type=str, default="ablation_job")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Ablation switches
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

    # New switch: replace mutant structure inputs with Δ(structure)
    parser.add_argument("--delta_struct", action="store_true", help="Feed Δ(structure) = mutant − WT for mutant structure features (pair, pLDDT).")

    # Accumulated Spearman queue size (for batch_size=1)
    parser.add_argument("--spearman_queue", type=int, default=256, help="FIFO queue size to accumulate Spearman loss across steps.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed or 0)
    torch.cuda.manual_seed_all(args.seed or 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = ablation_suffix(args)
    best_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_best_{suffix}_{args.seed}.pt")
    test_csv_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_test_predictions_{suffix}_{args.seed}.csv")

    # --- Data ---
    # Load full training CSV as DataFrame
    full_df = pd.read_csv(args.train_csv)

    # Tag each entry with its protein name using name format
    full_df['protein'] = full_df['name'].apply(lambda x: x.split('_')[1])
    protein_col = "protein"  # <-- change if needed

    assert protein_col in full_df.columns, f"{protein_col} not in train CSV"

    # Get unique proteins and split them into train / val sets
    val_frac = 0.1
    proteins = full_df[protein_col].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(proteins)

    n_val_prot = max(1, int(math.ceil(len(proteins) * val_frac)))
    val_proteins = set(proteins[:n_val_prot])
    train_proteins = set(proteins[n_val_prot:])

    train_df = full_df[full_df[protein_col].isin(train_proteins)].reset_index(drop=True)
    val_df   = full_df[full_df[protein_col].isin(val_proteins)].reset_index(drop=True)

    print(f"Protein-disjoint split:")
    print(f"  Train proteins: {len(train_proteins)}, samples: {len(train_df)}")
    print(f"  Val proteins:   {len(val_proteins)}, samples: {len(val_df)}", flush=True)

    # Build datasets from DataFrames
    train_ds = GeoDTmAblationDataset(
        train_df, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    val_ds = GeoDTmAblationDataset(
        val_df, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    test_ds = GeoDTmAblationDataset(
        args.test_csv, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- Compute fixed feature length deterministically ---
    # The dataset always constructs fixed_full as: [7 physchem] + [pH?] + [pLDDT?]
    fixed_dim = 0
    fixed_dim += 7  # physchem fixed_embedding.pt always provides 7 dims on disk
    if args.use_pH: fixed_dim += 1
    if args.use_plddt: fixed_dim += 1

    print(f"[Info] Using fixed_dim = {fixed_dim} (7 physchem + pH:{int(args.use_pH)} + pLDDT:{int(args.use_plddt)})", flush=True)
    if args.delta_struct:
        print("[Info] delta_struct enabled: mutant 'pair' replaced with Δpair, mutant pLDDT replaced with ΔpLDDT.", flush=True)

    # --- Model (create AFTER datasets so fixed_dim matches actual data) ---
    model = GeoDTmAblationModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
        fixed_dim=fixed_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Spearman accumulation queue (training only)
    train_queue = SpearmanQueue(capacity=args.spearman_queue, device=device)

    print("Stage 1: Freezing encoder for rapid head optimization (GeoDTm ablation).", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False

    best_val_loss = float("inf")
    early_counter = 0

    for epoch in range(1, args.epochs_frozen + 1):
        train_loss, train_mse, train_rho = run_epoch(
            model, train_loader, optimizer, device, args.alpha_loss,
            delta_struct=args.delta_struct, use_pair=args.use_pair, use_plddt=args.use_plddt,
            spearman_queue=train_queue
        )
        val_loss, val_mse, val_rho = run_epoch(
            model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss,
            delta_struct=args.delta_struct, use_pair=args.use_pair, use_plddt=args.use_plddt,
            spearman_queue=None
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

    # Optionally reset or keep queue; here we keep it so correlation builds up.
    # train_queue = SpearmanQueue(capacity=args.spearman_queue, device=device)

    early_counter = 0
    for epoch in range(1, args.epochs_finetune + 1):
        train_loss, train_mse, train_rho = run_epoch(
            model, train_loader, optimizer, device, args.alpha_loss,
            delta_struct=args.delta_struct, use_pair=args.use_pair, use_plddt=args.use_plddt,
            spearman_queue=train_queue
        )
        val_loss, val_mse, val_rho = run_epoch(
            model, val_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss,
            delta_struct=args.delta_struct, use_pair=args.use_pair, use_plddt=args.use_plddt,
            spearman_queue=None
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

    # Test set and output CSV
    print("Generating test-set predictions and saving CSV...", flush=True)
    model.eval()
    test_names = []
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            wt_data, mut_data, target = move_batch_to_device(batch, device)
            if args.delta_struct:
                apply_delta_struct(wt_data, mut_data, use_pair=args.use_pair, use_plddt=args.use_plddt)
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