import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_dTm_3D")))
from model import PretrainEncoder, ATOM_CA

PATHO_CLASS_MAP = {'Rare': 0, 'Common': 0, 'Pathogenic': 1, 'Likely-pathogenic': 1}

###############################################################################
# Dataset
###############################################################################

class GeoPathoAblationDataset(Dataset):
    def __init__(self, csv_path, features_dir,
                 use_fixed_embedding=True, use_dynamic_embedding=True,
                 use_pair=True, use_atom_mask=True, use_pH=True, use_plddt=True):
        super().__init__()
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir
        self.use_fixed_embedding = use_fixed_embedding
        self.use_dynamic_embedding = use_dynamic_embedding
        self.use_pair = use_pair
        self.use_atom_mask = use_atom_mask
        self.use_pH = use_pH
        self.use_plddt = use_plddt
        ph_candidates = [c for c in self.df.columns if c.lower() == "ph"]
        self.ph_col = ph_candidates[0] if ph_candidates else None

    def _load_feature_dict(self, row, variant: str):
        sample_id = str(row["prot_variant"])
        folder = os.path.join(self.features_dir, sample_id, variant)
        # Load ESM2 embedding to infer length L
        d_emb = torch.load(os.path.join(folder, "esm2.pt")).float()
        L = d_emb.shape[0]

        # Dynamic embedding (may be zeroed for ablation)
        d_emb = d_emb if self.use_dynamic_embedding else torch.zeros_like(d_emb)

        # Fixed 7-d physchem
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()
        if fixed.dim() == 1:
            fixed = fixed.unsqueeze(-1)
        fixed = fixed if self.use_fixed_embedding else torch.zeros_like(fixed)

        # Pair
        pair = torch.load(os.path.join(folder, "pair.pt")).float()
        pair = pair if self.use_pair else torch.zeros_like(pair)

        # Coordinates / atom mask
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        atom_mask = coord_data["pos14_mask"].all(dim=-1)
        atom_mask = atom_mask if self.use_atom_mask else torch.ones_like(atom_mask, dtype=torch.bool)

        # pH
        ph_val = 7.0
        if self.use_pH and self.ph_col is not None:
            try:
                ph_val = float(row[self.ph_col])
            except Exception:
                ph_val = 7.0
            ph_val = max(0.0, min(11.0, ph_val))
        ph_feat = torch.full((L, 1), ph_val, dtype=torch.float32)
        ph_feat = ph_feat if self.use_pH else torch.zeros_like(ph_feat)

        # pLDDT from esmf pickle
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        plddt = torch.tensor(pkl.get("plddt", []), dtype=torch.float32)
        if plddt.dim() != 1:
            plddt = plddt.view(-1)
        # align lengths
        if plddt.numel() == 0:
            plddt = torch.zeros((L,), dtype=torch.float32)
        else:
            if plddt.shape[0] > L:
                plddt = plddt[:L]
            elif plddt.shape[0] < L:
                pad_val = plddt[-1] if plddt.numel() > 0 else torch.tensor(0.0, dtype=plddt.dtype)
                pad = pad_val.repeat(L - plddt.shape[0])
                plddt = torch.cat([plddt, pad], dim=0)
        plddt = plddt / 100.0
        plddt_feat = plddt.unsqueeze(-1)
        plddt_feat = plddt_feat if self.use_plddt else torch.zeros_like(plddt_feat)

        # Merge fixed: keep columns stable in order [7 physchem] + [pH] + [pLDDT]
        fixed_full = torch.cat([fixed, ph_feat, plddt_feat], dim=-1)

        # mut_pos mask (optional)
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
        wt_data = self._load_feature_dict(row, "wt_data")
        mut_data = self._load_feature_dict(row, "mut_data")
        label_str = row.get("class", "")
        target = PATHO_CLASS_MAP.get(label_str, 0)
        return wt_data, mut_data, torch.tensor(target, dtype=torch.float32)

###############################################################################
# Model
###############################################################################

class GeoPathoAblationModel(nn.Module):
    def __init__(self, node_dim, n_head, pair_dim, num_layer, fixed_dim):
        super().__init__()
        # Reuse PretrainEncoder components for consistency with other scripts
        self.encoder = PretrainEncoder(node_dim, n_head, pair_dim, num_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, 1),
        )
        self.fixed_dim = fixed_dim
        self.node_dim = node_dim

        # Project fixed vector -> node_dim. Use bias=False so zero inputs map to zero outputs.
        if fixed_dim > 0:
            self.fixed_proj = nn.Sequential(
                nn.Linear(fixed_dim, node_dim, bias=False),
                nn.LeakyReLU(),
                nn.Linear(node_dim, node_dim, bias=False),
            )
            # input_proj maps concatenated [dyn_node, fixed_proj] -> node_dim
            self.input_proj = nn.Linear(node_dim + node_dim, node_dim)
        else:
            self.input_proj = nn.Identity()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        mask = mask_1d.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data):
        # dynamic embedding: allow unbatched input
        dyn_emb = data["dynamic_embedding"]
        if dyn_emb.dim() == 2:
            dyn_emb = dyn_emb.unsqueeze(0)  # [1, L, 1280]

        # pair
        pair = data["pair"]
        if pair.dim() == 3:
            pair = pair.unsqueeze(0)

        # atom_mask
        atom_mask = data["atom_mask"]
        if atom_mask.dim() == 2:
            atom_mask = atom_mask.unsqueeze(0)

        # 1) esm2 -> node_dim
        dyn_node = self.encoder.esm2_transform(dyn_emb)  # [N, L, node_dim]

        # 2) fixed features: dataset provides fixed_embedding = [L, fixed_dim] even when ablated
        fixed = data.get("fixed_embedding", None)
        if fixed is None:
            fixed_proj = torch.zeros_like(dyn_node)
        else:
            if fixed.dim() == 2:
                fixed = fixed.unsqueeze(0)  # [1, L, fixed_dim]
            sample_fixed_dim = fixed.shape[-1]
            if sample_fixed_dim != self.fixed_dim:
                raise RuntimeError(
                    f"fixed_embedding last-dim ({sample_fixed_dim}) != model.fixed_dim ({self.fixed_dim}). "
                    "Model should be constructed with fixed_dim = 7 + 1 + 1 (physchem + pH + pLDDT) "
                    "and the dataset keeps these columns (zeroed for ablation)."
                )
            if self.fixed_dim > 0:
                fixed_proj = self.fixed_proj(fixed)  # [N, L, node_dim]
            else:
                fixed_proj = torch.zeros_like(dyn_node)

        # 3) combine and project to initial embedding
        if isinstance(self.input_proj, nn.Identity):
            embedding = dyn_node
        else:
            embedding = self.input_proj(torch.cat([dyn_node, fixed_proj], dim=-1))  # [N, L, node_dim]

        # 4) pair encoding and apply blocks
        pair_enc = self.encoder.pair_encoder(pair)
        for block in self.encoder.blocks:
            embedding, pair_enc = block(embedding, pair_enc, atom_mask[:, :, ATOM_CA])

        # pooling
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
# Training helpers (unchanged)
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
    optimizer=None,
    device="cuda",
    is_train=False,
):
    model.train(is_train)
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        wt_data, mut_data, target = move_batch_to_device(batch, device)
        logits = model(wt_data, mut_data)
        loss = criterion(logits, target)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * target.size(0)
        n_samples += target.size(0)
        all_preds.append(logits.detach().cpu())
        all_targets.append(target.detach().cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    probs = torch.sigmoid(all_preds)
    preds = (probs > 0.5).float()
    acc = (preds == all_targets).float().mean().item()
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets.numpy(), probs.numpy())
    except Exception:
        auc = float('nan')
    return total_loss / max(n_samples, 1), acc, auc

def ablation_suffix(args):
    ablated = []
    if not args.use_fixed_embedding: ablated.append("no_fixed")
    if not args.use_dynamic_embedding: ablated.append("no_demb")
    if not args.use_pair: ablated.append("no_pair")
    if not args.use_atom_mask: ablated.append("no_atommask")
    if not args.use_pH: ablated.append("no_pH")
    if not args.use_plddt: ablated.append("no_plddt")
    return "_".join(ablated) if ablated else "all_inputs"

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default='/projects/ashehu/amoldwin/datasets/mutation/patho_train.csv')
    parser.add_argument("--val_csv", type=str, default='/projects/ashehu/amoldwin/datasets/mutation/patho_val.csv')
    parser.add_argument("--test_csv", type=str, default='/projects/ashehu/amoldwin/datasets/mutation/patho_test.csv')
    parser.add_argument("--features_dir", type=str,default='/projects/ashehu/amoldwin/GeoStab/data/patho/patho_FASTA/')
    parser.add_argument("--node_dim", type=int, default=64)
    parser.add_argument("--pair_dim", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs_frozen", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=30)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="geopatho_models_ablation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--job_id", type=str, default="ablation_patho_job")
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
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = ablation_suffix(args)
    best_path = os.path.join(args.out_dir, f"{args.job_id}_geopatho_best_{suffix}_{args.seed}.pt")
    test_csv_path = os.path.join(args.out_dir, f"{args.job_id}_geopatho_test_predictions_{suffix}_{args.seed}.csv")

    # fixed_dim: keep deterministic layout -> 7 physchem + pH + pLDDT = 9
    fixed_dim = 7 + 1 + 1
    print(f"[Info] Using fixed_dim = {fixed_dim} (7 physchem + pH + pLDDT). The dataset keeps columns zeroed for ablation.", flush=True)

    # Datasets
    train_df  = pd.read_csv(args.train_csv)
    val_df    = pd.read_csv(args.val_csv)
    full_df   = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    protein_col = "prot_acc_version"  # adapt if needed
    proteins = full_df[protein_col].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(proteins)
    val_frac = 0.1
    n_val_prot = max(1, int(np.ceil(len(proteins) * val_frac)))
    val_proteins = set(proteins[:n_val_prot])
    train_proteins = set(proteins[n_val_prot:])
    train_df = full_df[full_df[protein_col].isin(train_proteins)].reset_index(drop=True)
    val_df   = full_df[full_df[protein_col].isin(val_proteins)].reset_index(drop=True)

    print(f"Protein-disjoint split:")
    print(f"  Train proteins: {len(train_proteins)}, samples: {len(train_df)}")
    print(f"  Val proteins:   {len(val_proteins)}, samples: {len(val_df)}", flush=True)

    train_ds = GeoPathoAblationDataset(train_df, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    val_ds = GeoPathoAblationDataset(val_df, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    test_ds = GeoPathoAblationDataset(args.test_csv, args.features_dir,
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

    # Build model AFTER datasets to ensure consistent behavior and optimizer parameters
    model = GeoPathoAblationModel(
        node_dim=args.node_dim,
        n_head=args.n_head,
        pair_dim=args.pair_dim,
        num_layer=args.num_layer,
        fixed_dim=fixed_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    print("Stage 1: Freezing encoder for rapid head optimization (GeoPatho ablation).", flush=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    best_val_loss, early_counter = float("inf"), 0

    for epoch in range(1, args.epochs_frozen + 1):
        train_loss, train_acc, train_auc = run_epoch(model, train_loader, optimizer, device, is_train=True)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, None, device, is_train=False)
        scheduler.step(val_loss)
        print(
            f"[Frozen] Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | Train acc {train_acc:.2f} | Train AUC {train_auc:.3f} | "
            f"Val loss {val_loss:.4f} | Val acc {val_acc:.2f} | Val AUC {val_auc:.3f}"
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
        train_loss, train_acc, train_auc = run_epoch(model, train_loader, optimizer, device, is_train=True)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, None, device, is_train=False)
        scheduler.step(val_loss)
        print(
            f"[Finetune] Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | Train acc {train_acc:.2f} | Train AUC {train_auc:.3f} | "
            f"Val loss {val_loss:.4f} | Val acc {val_acc:.2f} | Val AUC {val_auc:.3f}"
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

    print(f"Loading best model from {best_path} for test evaluation.", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_auc = run_epoch(model, test_loader, None, device, is_train=False)
    print(
        f"Test | Loss {test_loss:.4f} | Acc {test_acc:.2f} | AUC {test_auc:.3f}",
        flush=True,
    )

    # Save test predictions to CSV
    model.eval()
    test_names, test_preds, test_targets = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            wt_data, mut_data, target = move_batch_to_device(batch, device)
            logits = model(wt_data, mut_data)
            prob = torch.sigmoid(logits).cpu().item()
            sample_name = test_ds.df.iloc[i]["prot_variant"]
            test_names.append(sample_name)
            test_preds.append(prob)
            test_targets.append(float(target.cpu().item()))
    pd.DataFrame({
        "prot_variant": test_names,
        "model_score": test_preds,
        "true_label": test_targets,
    }).to_csv(test_csv_path, index=False)
    print(f"Saved test predictions to: {test_csv_path}", flush=True)

if __name__ == "__main__":
    main()