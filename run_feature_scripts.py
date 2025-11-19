import os
import subprocess
from glob import glob

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

def batch_run_esm2(fasta_files, parent_dirs):
    # (Assumes generate_features/esm2_embedding.py supports batching: as --fasta_files foo1.fa foo2.fa ...)
    missing = []
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        outpath = os.path.join(vdir, "esm2.pt")
        if not os.path.exists(outpath):
            missing.append((fastapath, vdir))

    if missing:
        # Build file lists
        input_files = [f for f, _ in missing]
        out_dirs = [d for _, d in missing]
        subprocess.run([
            "python", os.path.join(FEATURE_SCRIPT_DIR, "esm2_embedding.py"),
            "--fasta_files", *input_files,
            "--saved_folders", *out_dirs
        ])

def batch_run_esm1v(fasta_files, parent_dirs):
    missing = []
    for idx in range(1, 6):
        for fastapath, vdir in zip(fasta_files, parent_dirs):
            outpath = os.path.join(vdir, f"esm1v-{idx}.pt")
            if not os.path.exists(outpath):
                missing.append((fastapath, vdir, idx))
    if missing:
        # Assumes batching interface
        input_files = [f for f, _, _ in missing]
        out_dirs = [d for _, d, _ in missing]
        model_idxs = [str(i) for _, _, i in missing]
        subprocess.run([
            "python", os.path.join(FEATURE_SCRIPT_DIR, "esm1v_logits.py"),
            "--fasta_files", *input_files,
            "--saved_folders", *out_dirs,
            "--model_indices", *model_idxs
        ])

def run_fixed_embedding(fasta_files, parent_dirs):
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        outpath = os.path.join(vdir, "fixed_embedding.pt")
        if not os.path.exists(outpath):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "fixed_embedding.py"),
                "--fasta_file", fastapath,
                "--saved_folder", vdir
            ])

def run_coordinate_and_pair(sample_dirs):
    for sample_dir in sample_dirs:
        for variant in ["wt_data", "mut_data"]:
            vdir = os.path.join(sample_dir, variant)
            pdb = "wt.pdb" if variant == "wt_data" else "mut.pdb"
            pdbpath = os.path.join(vdir, pdb)
            if os.path.exists(pdbpath):
                coord_path = os.path.join(vdir, "coordinate.pt")
                if not os.path.exists(coord_path):
                    subprocess.run([
                        "python", os.path.join(FEATURE_SCRIPT_DIR, "coordinate.py"),
                        "--pdb_file", pdbpath,
                        "--saved_folder", vdir
                    ])
                if os.path.exists(coord_path) and not os.path.exists(os.path.join(vdir, "pair.pt")):
                    subprocess.run([
                        "python", os.path.join(FEATURE_SCRIPT_DIR, "pair.py"),
                        "--coordinate_file", coord_path,
                        "--saved_folder", vdir
                    ])

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

    # Step 3: Batch DL models (must update called scripts for batching!)
    print("Batching ESM2...", flush=True)
    batch_run_esm2(wt_fastas + mut_fastas, wt_dirs + mut_dirs)
    print("Batching ESM1v...", flush=True)
    batch_run_esm1v(wt_fastas + mut_fastas, wt_dirs + mut_dirs)

    # Step 4: Lightweight features (fixed_embedding, coordinate, pair)
    run_fixed_embedding(wt_fastas + mut_fastas, wt_dirs + mut_dirs)
    run_coordinate_and_pair(sample_dirs)

    # Step 5: Ensemble generation
    for sample_dir in sample_dirs:
        ensemble_path = os.path.join(sample_dir, "ensemble.pt")
        if not os.path.exists(ensemble_path):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "ensemble_ddGdTm.py"),
                "--saved_folder", sample_dir,
            ])

if __name__ == "__main__":
    main()