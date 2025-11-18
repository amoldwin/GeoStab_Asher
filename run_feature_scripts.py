import os
import subprocess

# Location of the scripts and parent data directory
FEATURE_SCRIPT_DIR = "generate_features"
DATA_PARENT_DIR = "./data/dTm/S4346/"  # Change to your actual parent directory, e.g. "data" or "samples"

def run_feature_scripts(sample_dir):
    for variant in ["wt_data", "mut_data"]:
        vdir = os.path.join(sample_dir, variant)
        fasta = "wt.fasta" if variant == "wt_data" else "mut.fasta"
        fastapath = os.path.join(vdir, fasta)
        pdb = "wt.pdb" if variant == "wt_data" else "mut.pdb"
        pdbpath = os.path.join(vdir, pdb)

        # 1) ESM2 embedding
        if not os.path.exists(os.path.join(vdir, "esm2.pt")):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "esm2_embedding.py"),
                "--fasta_file", fastapath,
                "--saved_folder", vdir
            ])

        # 2) Fixed embedding
        if not os.path.exists(os.path.join(vdir, "fixed_embedding.pt")):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "fixed_embedding.py"),
                "--fasta_file", fastapath,
                "--saved_folder", vdir
            ])

        # 3) ESM1v logits (for 1..5)
        for idx in range(1, 6):
            ptfile = os.path.join(vdir, f"esm1v-{idx}.pt")
            if not os.path.exists(ptfile):
                subprocess.run([
                    "python", os.path.join(FEATURE_SCRIPT_DIR, "esm1v_logits.py"),
                    "--model_index", str(idx),
                    "--fasta_file", fastapath,
                    "--saved_folder", vdir
                ])

        # 4) Coordinates
        pdb_file_path = os.path.join(vdir, pdb)
        if os.path.exists(pdb_file_path) and not os.path.exists(os.path.join(vdir, "coordinate.pt")):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "coordinate.py"),
                "--pdb_file", pdb_file_path,
                "--saved_folder", vdir
            ])

        # 5) Pair features (requires coordinate.pt)
        if os.path.exists(os.path.join(vdir, "coordinate.pt")) and not os.path.exists(os.path.join(vdir, "pair.pt")):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "pair.py"),
                "--coordinate_file", os.path.join(vdir, "coordinate.pt"),
                "--saved_folder", vdir
            ])

def main():
    # Find all sample folders (they contain 'mut_info.csv')
    for sample in os.listdir(DATA_PARENT_DIR):
        sample_dir = os.path.join(DATA_PARENT_DIR, sample)
        if not os.path.isdir(sample_dir):
            print(f"could not find {sample_dir}", flush=True)
            continue
        if not os.path.exists(os.path.join(sample_dir, "mut_info.csv")):

            print(f"could not find mut_info in {sample_dir}", flush=True)

            continue

        print(f"Processing {sample_dir}...", flush=True)
        run_feature_scripts(sample_dir)

        # Final ensemble generation
        ensemble_path = os.path.join(sample_dir, "ensemble.pt")
        if not os.path.exists(ensemble_path):
            subprocess.run([
                "python", os.path.join(FEATURE_SCRIPT_DIR, "ensemble_ddGdTm.py"),
                "--saved_folder", sample_dir,
                # If needed: "--af2_pickle_file", ... [add here]
            ])

if __name__ == "__main__":
    main()