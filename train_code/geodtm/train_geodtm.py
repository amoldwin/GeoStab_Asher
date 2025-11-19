import os
import math
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Import the geometric encoder from your existing model.py
from model import PretrainEncoder, ATOM_CA  # ATOM_CA defined in your model.py

###############################################################################
# GeoDTm model: shared encoder for WT / Mutant + anti-symmetric ΔTm head
###############################################################################

class GeoDTmModel(nn.Module):
    """
    GeoDTm-style model:

    - Shared geometric encoder for wild-type and mutant.
    - Masked mean pooling to get per-protein embeddings.
    - ΔTm = MLP( z_mut - z_wt ).
    """
    def __init__(self, node_dim: int, n_head: int, pair_dim: int, num_layer: int):
        super().__init__()

        # Geometric encoder (same architecture as GeoFitness encoder)
        self.encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)

        # Small MLP head on top of encoder pooled difference
        self.head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        """
        x: [N, L, D]
        mask_1d: [N, L] boolean
        returns: [N, D]
        """
        mask = mask_1d.unsqueeze(-1)  # [N, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single protein (batch) into [N, L, node_dim] then pool to [N, node_dim].
        Uses the same pLDDT gating logic as PretrainModel from model.py.
        """
        # data keys: "dynamic_embedding", "fixed_embedding", "pair", "atom_mask"
        # Add pLDDT gate (last channel of fixed_embedding)
        plddt = torch.sign(torch.relu(data["fixed_embedding"][:, :, -1] - 0.7)).bool()
        atom_mask = torch.stack(
            (data["atom_mask"], plddt.unsqueeze(-1).repeat(1, 1, data["atom_mask"].shape[-1])),
            dim=0,
        ).all(dim=0)

        # Encoder produces [N, L, node_dim]
        node_feat = self.encoder(
            data["dynamic_embedding"],
            data["pair"],
            atom_mask,
        )

        # Mask for residues based on CA atom
        res_mask = atom_mask[:, :, ATOM_CA]  # [N, L] boolean
        pooled = self._masked_mean(node_feat, res_mask)  # [N, node_dim]
        return pooled

    def forward(
        self,
        wt_data: Dict[str, torch.Tensor],
        mut_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward for a batch of samples.

        Returns:
            pred_dtm: [N] predicted ΔTm (mut - wt, in °C)
        """
        z_wt = self.encode(wt_data)   # [N, D]
        z_mut = self.encode(mut_data) # [N, D]

        delta = z_mut - z_wt          # anti-symmetric by construction
        out = self.head(delta).squeeze(-1)  # [N]
        return out


###############################################################################
# Dataset for ΔTm (S4346 / S571)
###############################################################################

class GeoDTmDataset(Dataset):
    """
    Simple dataset that:

    - Reads rows from S4346.csv or S571.csv
    - For each row, loads WT and mutant features from .pt files
      at paths based on sample_id.

    Expected CSV columns:
        sample_id: unique identifier used to locate feature files
        dtm:       experimental ΔTm value (float, °C)

    Feature files:
        <features_dir>/<sample_id>_wt.pt
        <features_dir>/<sample_id>_mut.pt

    Each .pt should be a dict with keys:
        "dynamic_embedding", "fixed_embedding", "pair", "atom_mask"
    """
    def __init__(self, csv_path: str, features_dir: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir

        # adapt these if your column names differ
        if "name" not in self.df.columns:
            raise ValueError("CSV must contain a 'name' column.")
        if "dtm" not in self.df.columns:
            raise ValueError("CSV must contain a 'dtm' column with ΔTm values.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["name"])
        target = float(row["dtm"])

        wt_path = os.path.join(self.features_dir, f"{sample_id}_wt.pt")
        mut_path = os.path.join(self.features_dir, f"{sample_id}_mut.pt")

        wt_data = torch.load(wt_path)
        mut_data = torch.load(mut_path)

        # convert all tensors to float32, etc. (assuming they are already tensors)
        for k in wt_data:
            if isinstance(wt_data[k], torch.Tensor):
                wt_data[k] = wt_data[k].float()
        for k in mut_data:
            if isinstance(mut_data[k], torch.Tensor):
                mut_data[k] = mut_data[k].float()

        target = torch.tensor(target, dtype=torch.float32)
        return wt_data, mut_data, target


###############################################################################
# Soft Spearman loss (approx) + MSE
###############################################################################

def soft_rank(x: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """
    Simple differentiable approximate rank using a pairwise sigmoid kernel.
    Not exactly Blondel et al.'s algorithm, but a common soft-rank surrogate.

    x: [N]
    returns: [N] approximate ranks in [1, N]
    """
    x = x.unsqueeze(0)  # [1, N]
    diff = x.T - x      # [N, N]
    P = torch.sigmoid(diff * regularization_strength)
    # expected rank ~ 1 + sum_j P_ij
    ranks = 1 + P.sum(dim=1)
    return ranks


def spearman_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Approximate Spearman correlation as a loss = 1 - ρ.

    pred, target: [N]
    """
    # center both
    pred_r = soft_rank(pred)
    targ_r = soft_rank(target)

    pred_r = pred_r - pred_r.mean()
    targ_r = targ_r - targ_r.mean()

    pred_r = pred_r / (pred_r.norm(p=2) + 1e-8)
    targ_r = targ_r / (targ_r.norm(p=2) + 1e-8)

    rho = (pred_r * targ_r).sum()
    return 1.0 - rho  # want to maximize ρ → minimize 1-ρ


def dtm_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Conjugated loss: L = α * (1 - Spearman) + (1 - α) * MSE

    Paper: use soft Spearman + MSE for ∆∆G / ∆Tm training. :contentReference[oaicite:6]{index=6}
    """
    loss_spear = spearman_loss(pred, target)
    loss_mse = F.mse_loss(pred, target)
    return alpha * loss_spear + (1.0 - alpha) * loss_mse


###############################################################################
# Training / evaluation loops
###############################################################################

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
) -> Tuple[float, float, float]:
    """
    One epoch over loader.

    Returns:
        mean_loss, mse, spearman (on full epoch).
    """
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

    # Compute epoch-level MSE and Spearman for logging
    mse = F.mse_loss(all_preds, all_targets).item()

    # Spearman via (non-differentiable) rank
    pred_rank = torch.argsort(torch.argsort(all_preds))
    targ_rank = torch.argsort(torch.argsort(all_targets))
    pred_rank = pred_rank.float() - pred_rank.float().mean()
    targ_rank = targ_rank.float() - targ_rank.float().mean()
    pred_rank /= (pred_rank.norm(p=2) + 1e-8)
    targ_rank /= (targ_rank.norm(p=2) + 1e-8)
    rho = (pred_rank * targ_rank).sum().item()

    return total_loss / max(n_samples, 1), mse, rho


###############################################################################
# Main training routine
###############################################################################

def load_pretrained_encoder(
    model: GeoDTmModel,
    geofitness_ckpt: str,
    device: torch.device,
):
    """
    Load encoder weights from a pretrained GeoFitness checkpoint.

    Assumes checkpoint was saved with PretrainModel from model.py, whose
    state_dict contains 'pretrain_encoder.*'. :contentReference[oaicite:7]{index=7}
    """
    print(f"Loading pretrained GeoFitness from {geofitness_ckpt}")
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
    print("Loaded encoder from GeoFitness. Missing:", missing, "Unexpected:", unexpected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S4346.csv",
                        help="Training CSV (ΔTm training data)")
    parser.add_argument("--test_csv", type=str, default="/projects/ashehu/amoldwin/datasets/protein_melting_temps/S571.csv",
                        help="Test CSV (ΔTm benchmark)")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing <sample_id>_wt.pt and <sample_id>_mut.pt")
    parser.add_argument("--geofitness_ckpt", type=str, required=True,
                        help="Path to pretrained GeoFitness .pt")
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
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

    # Load datasets
    full_train = GeoDTmDataset(args.train_csv, args.features_dir)
    # Simple split S4346 → train/val internally (e.g. 90/10)
    val_frac = 0.1
    n_total = len(full_train)
    n_val = max(1, int(math.ceil(n_total * val_frac)))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    test_ds = GeoDTmDataset(args.test_csv, args.features_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = GeoDTmModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
    ).to(device)

    # Load pretrained encoder
    load_pretrained_encoder(model, args.geofitness_ckpt, device=device)

    # Optimizer and scheduler
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

    # -----------------------------
    # Stage 1: encoder frozen
    # -----------------------------
    print("Stage 1: Freezing encoder for rapid head optimization (GeoDTm, as in paper).")
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
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Early stopping (frozen stage).")
                break

    # -----------------------------
    # Stage 2: fine-tune all params
    # -----------------------------
    print("Stage 2: Unfreezing encoder for joint fine-tuning.")
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
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            early_counter += 1
            if early_counter >= args.early_stop:
                print("Early stopping (fine-tune stage).")
                break

    # -----------------------------
    # Final test on S571
    # -----------------------------
    print(f"Loading best model from {best_path} for test evaluation (S571).")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_mse, test_rho = run_epoch(
        model, test_loader, optimizer=None, device=device, alpha_loss=args.alpha_loss
    )
    print(
        f"Test (S571) | Loss {test_loss:.4f} | MSE {test_mse:.4f} | Spearman {test_rho:.3f}"
    )


if __name__ == "__main__":
    main()
