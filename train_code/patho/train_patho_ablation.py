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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_dTm_3D")))
from model import PretrainEncoder, ATOM_CA

PATHO_CLASS_MAP = {'Rare': 0, 'Common': 0, 'Pathogenic': 1, 'Likely-pathogenic': 1}

class GeoPathoAblationDataset(Dataset):
    def __init__(self, csv_path, features_dir,
                 use_fixed_embedding=True, use_dynamic_embedding=True,
                 use_pair=True, use_atom_mask=True, use_pH=True, use_plddt=True):
        super().__init__()
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
        sample_id = str(row["name"])
        folder = os.path.join(self.features_dir, sample_id, variant)
        L = torch.load(os.path.join(folder, "esm2.pt")).shape[0]
        d_emb = torch.load(os.path.join(folder, "esm2.pt")).float()
        d_emb = d_emb if self.use_dynamic_embedding else torch.zeros_like(d_emb)
        fixed = torch.load(os.path.join(folder, "fixed_embedding.pt")).float()
        fixed = fixed if self.use_fixed_embedding else torch.zeros_like(fixed)
        if fixed.dim() == 1: fixed = fixed.unsqueeze(-1)
        pair = torch.load(os.path.join(folder, "pair.pt")).float()
        pair = pair if self.use_pair else torch.zeros_like(pair)
        coord_data = torch.load(os.path.join(folder, "coordinate.pt"))
        atom_mask = coord_data["pos14_mask"].all(dim=-1)
        atom_mask = atom_mask if self.use_atom_mask else torch.ones_like(atom_mask, dtype=torch.bool)
        ph_val = 7.0
        if self.use_pH and self.ph_col is not None:
            ph_val = float(row[self.ph_col])
            ph_val = max(0.0, min(11.0, ph_val))
        ph_feat = torch.full((L, 1), ph_val, dtype=torch.float32)
        pkl_filename = "wt_esmf.pkl" if variant == "wt_data" else "mut_esmf.pkl"
        pkl_path = os.path.join(folder, pkl_filename)
        with open(pkl_path, "rb") as f:
            pkl = pickle.load(f)
        plddt = torch.tensor(pkl["plddt"], dtype=torch.float32)
        if plddt.dim() != 1: plddt = plddt.view(-1)
        L_plddt = plddt.shape[0]
        if L_plddt > L:   plddt = plddt[:L]
        elif L_plddt < L: plddt = torch.cat([plddt, torch.full((L-L_plddt,), plddt[-1])], dim=0)
        plddt = plddt / 100.0
        plddt_feat = plddt.unsqueeze(-1)
        fixed_full = torch.cat([fixed, ph_feat, plddt_feat], dim=-1)
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
        label_str = row["class"]
        target = PATHO_CLASS_MAP.get(label_str, 0)
        return wt_data, mut_data, torch.tensor(target, dtype=torch.float32)

class GeoPathoAblationModel(nn.Module):
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

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        mask = mask_1d.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def encode(self, data):
        dyn_emb = data["dynamic_embedding"].unsqueeze(0) if data["dynamic_embedding"].dim() == 2 else data["dynamic_embedding"]
        pair = data["pair"].unsqueeze(0) if data["pair"].dim() == 3 else data["pair"]
        atom_mask = data["atom_mask"].unsqueeze(0) if data["atom_mask"].dim() == 2 else data["atom_mask"]
        node_feat = self.encoder(dyn_emb, pair, atom_mask)
        res_mask = atom_mask[:, :, ATOM_CA]
        pooled = self._masked_mean(node_feat, res_mask)
        return pooled

    def forward(self, wt_data, mut_data):
        z_wt = self.encode(wt_data)
        z_mut = self.encode(mut_data)
        delta = z_mut - z_wt
        out = self.head(delta).squeeze(-1)
        return out

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
    # Compute classification metrics
    probs = torch.sigmoid(all_preds)
    preds = (probs > 0.5).float()
    acc = (preds == all_targets).float().mean().item()
    # AUC requires at least two classes
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
    best_path = os.path.join(args.out_dir, f"{args.job_id}_geopatho_best_{suffix}.pt")
    test_csv_path = os.path.join(args.out_dir, f"{args.job_id}_geopatho_test_predictions_{suffix}.csv")
    # fixed_full: 7+1+1 if all present, fewer if ablated
    fixed_dim = 0
    if args.use_fixed_embedding: fixed_dim += 7
    if args.use_pH: fixed_dim += 1
    if args.use_plddt: fixed_dim += 1

    # Datasets
    train_ds = GeoPathoAblationDataset(args.train_csv, args.features_dir,
        use_fixed_embedding=args.use_fixed_embedding,
        use_dynamic_embedding=args.use_dynamic_embedding,
        use_pair=args.use_pair,
        use_atom_mask=args.use_atom_mask,
        use_pH=args.use_pH,
        use_plddt=args.use_plddt
    )
    val_ds = GeoPathoAblationDataset(args.val_csv, args.features_dir,
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
            sample_name = test_ds.df.iloc[i]["name"]
            test_names.append(sample_name)
            test_preds.append(prob)
            test_targets.append(float(target.cpu().item()))
    pd.DataFrame({
        "name": test_names,
        "model_score": test_preds,
        "true_label": test_targets,
    }).to_csv(test_csv_path, index=False)
    print(f"Saved test predictions to: {test_csv_path}", flush=True)

if __name__ == "__main__":
    main()