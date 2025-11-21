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
        
        # Fixed embedding
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
        atom_mask = coord_data["pos14_mask"].all(dim=-1).float()
        feature_dict["atom_mask"] = atom_mask if self.use_atom_mask else torch.ones_like(atom_mask)

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
        plddt = plddt / 100.0  # Normalize
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
        self.encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1),
        )
        self.fixed_dim = fixed_dim
        self.node_dim = node_dim

        # Adjust embedding of fixed features for ablation
        self.fixed_proj = nn.Sequential(
            nn.Linear(fixed_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, node_dim)
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        mask = mask_1d.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data):
        # If ablating dynamic_embedding, it's zeros. If ablating fixed, it's zeros.
        # Project both to node_dim if needed, then sum, or concatenate then project.
        dyn_emb = data["dynamic_embedding"].unsqueeze(0)    # [1, L, 1280]
        fixed_full = data["fixed_full"].unsqueeze(0)         # [1, L, fixed_dim]

        # Concatenate and project to node_dim
        x_in = torch.cat([dyn_emb, fixed_full], dim=-1)      # [1, L, 1280 + fixed_dim]
        x_in = self.input_proj(x_in)                         # [1, L, node_dim]

        pair = data["pair"].unsqueeze(0)
        atom_mask = data["atom_mask"].unsqueeze(0)
        node_feat = self.encoder(x_in, pair, atom_mask)
        res_mask = atom_mask[:, :, ATOM_CA]
        pooled = self._masked_mean(node_feat, res_mask)
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
    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=50)
    parser.add_argument("--alpha_loss", type=float, default=0.5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="geodtm_models_ablation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--job_id", type=str, default="ablation_job")

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

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = ablation_suffix(args)
    best_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_best_{suffix}.pt")
    test_csv_path = os.path.join(args.out_dir, f"{args.job_id}_geodtm_test_predictions_{suffix}.csv")

    # Dim of fixed_full: 7+1+1 if all present, fewer if ablated
    fixed_dim = 0
    if args.use_fixed_embedding: fixed_dim += 7
    if args.use_pH: fixed_dim += 1
    if args.use_plddt: fixed_dim += 1

    # --- Data ---
    full_train = GeoDTmAblationDataset(
        args.train_csv,
        args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    val_frac = 0.1
    n_total = len(full_train)
    n_val = max(1, int(math.ceil(n_total * val_frac)))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    test_ds = GeoDTmAblationDataset(
        args.test_csv,
        args.features_dir,
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
        optimizer,
        factor=0.5,
        patience=5,
        verbose=True,
    )

    print("Stage 1: Freezing encoder for rapid head optimization (GeoDTm ablation).", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False

    best_val_loss = float("inf")
    early_counter = 0

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