import os
from glob import glob
from generate_features.esm2_embedding import run_esm2_embedding
from generate_features.esm1v_logits import run_esm1v_logits

FEATURE_SCRIPT_DIR = "generate_features"
DATA_PARENT_DIR = "./data/dTm/S4346/"  # Change if needed

def collect_variant_fastas(samples, variant):
    fasta_files = []
    parent_dirs = []
    for sample_dir in samples:
        vdir = os.path.join(sample_dir, variant)
        fasta = "wt.fasta" if variant == "wt_data" else "mut.fasta"
        fastapath = os.path.join(vdir, fasta)
        if os.path.exists(fastapath):
            fasta_files.append(fastapath)
            parent_dirs.append(vdir)
    return fasta_files, parent_dirs

def batch_run_esm2(fasta_files, parent_dirs, batch_size=8):
    missing = []
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        outpath = os.path.join(vdir, "esm2.pt")
        if not os.path.exists(outpath):
            missing.append((fastapath, vdir))

    # Split into manageable batches
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        input_files = [f for f, _ in batch]
        out_dirs = [d for _, d in batch]
        print(f"Running ESM2 batch: {input_files}", flush=True)
        run_esm2_embedding(input_files, out_dirs)

def batch_run_esm1v(fasta_files, parent_dirs, batch_size=8):
    missing = []
    for idx in range(1, 6):
        for fastapath, vdir in zip(fasta_files, parent_dirs):
            outpath = os.path.join(vdir, f"esm1v-{idx}.pt")
            if not os.path.exists(outpath):
                missing.append((fastapath, vdir, str(idx)))  # model_indices are strings

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        input_files = [f for f, _, _ in batch]
        out_dirs = [d for _, d, _ in batch]
        model_idxs = [idx for _, _, idx in batch]
        print(f"Running ESM1v batch: {input_files} model_idxs: {model_idxs}", flush=True)
        run_esm1v_logits(model_idxs, input_files, out_dirs)

def run_fixed_embedding(fasta_files, parent_dirs):
    
    import generate_features.fixed_embedding as fe
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        print(f"Running fixed embedding for: {fastapath} ", flush=True)

        outpath = os.path.join(vdir, "fixed_embedding.pt")
        if not os.path.exists(outpath):
            fe.main.callback(fasta_file=fastapath, saved_folder=vdir)

def run_coordinate_and_pair(sample_dirs):
    import generate_features.coordinate as coord
    import generate_features.pair as pair
    for sample_dir in sample_dirs:
        for variant in ["wt_data", "mut_data"]:
            print(f"running coordinate and pair for: {variant} ", flush=True)
            vdir = os.path.join(sample_dir, variant)
            pdb = "wt_esmfold.pdb" if variant == "wt_data" else "mut.pdb"
            pdbpath = os.path.join(vdir, pdb)
            coord_path = os.path.join(vdir, "coordinate.pt")
            pair_path = os.path.join(vdir, "pair.pt")
            if os.path.exists(pdbpath):
                if True:# not os.path.exists(coord_path):
                    coord.main.callback(pdb_file=pdbpath, saved_folder=vdir)
                if True:# os.path.exists(coord_path) and not os.path.exists(pair_path):
                    pair.main.callback(coordinate_file=coord_path, saved_folder=vdir)

def run_ensemble(sample_dirs):
    import generate_features.ensemble_ddGdTm as ensemble
    for sample_dir in sample_dirs:
        ensemble_path = os.path.join(sample_dir, "ensemble.pt")
        if not os.path.exists(ensemble_path):
            ensemble.main.callback(saved_folder=sample_dir)

def main():
    # Step 1: Identify sample folders
    sample_dirs = [
        os.path.join(DATA_PARENT_DIR, name)
        for name in os.listdir(DATA_PARENT_DIR)
        if os.path.isdir(os.path.join(DATA_PARENT_DIR, name))
           and os.path.exists(os.path.join(DATA_PARENT_DIR, name, "mut_info.csv"))
    ]

    print(f"Found {len(sample_dirs)} sample dirs", flush=True)

    # Step 2: Collect wildtype and mutant FASTA files for batching
    wt_fastas, wt_dirs = collect_variant_fastas(sample_dirs, "wt_data")
    mut_fastas, mut_dirs = collect_variant_fastas(sample_dirs, "mut_data")

    # Step 3: Batch DL models
    # print("Batching ESM2...", flush=True)
    # batch_run_esm2(wt_fastas + mut_fastas, wt_dirs + mut_dirs, batch_size=8)
    
    
    #removing because esm1v not 
    # print("Batching ESM1v...", flush=True)
    # batch_run_esm1v(wt_fastas + mut_fastas, wt_dirs + mut_dirs, batch_size=8)

    # Step 4: Lightweight features (fixed_embedding, coordinate, pair)
    #run_fixed_embedding(wt_fastas + mut_fastas, wt_dirs + mut_dirs)
    run_coordinate_and_pair(sample_dirs)

    # Step 5: Ensemble generation
    #run_ensemble(sample_dirs)

if __name__ == "__main__":
    main()