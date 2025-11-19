import os
import torch
import esm
from Bio import SeqIO
import argparse
import pickle

def main(data_parent, num_samples=None):
    # Load ESMFold model (ESM package, not HuggingFace!)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    model = esm.pretrained.esmfold_v1()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    print("Loaded ESMFold model to", device, flush=True)

    # List all samples
    jobs = []
    for sample_name in os.listdir(data_parent):
        mut_fasta = os.path.join(data_parent, sample_name, "mut_data", "mut.fasta")
        mut_pdb = os.path.join(data_parent, sample_name, "mut_data", "mut.pdb")
        mut_pkl = os.path.join(data_parent, sample_name, "mut_data", "mut_esmf.pkl")
        if os.path.exists(mut_fasta):
            jobs.append((mut_fasta, mut_pdb, mut_pkl))
    if num_samples is not None:
        jobs = jobs[:num_samples]
    print(f"Will predict {len(jobs)} mutant structures.",flush=True)

    # Predict and write PDBs + pKLTDT pickles
    for mut_fasta, mut_pdb, mut_pkl in reversed(jobs):
        record = list(SeqIO.parse(mut_fasta, "fasta"))[0]
        sequence = str(record.seq)
        # Whether or not to skip prediction based on file existence
        need_pdb = not os.path.exists(mut_pdb)
        need_pkl = not os.path.exists(mut_pkl)
        if not (need_pdb or need_pkl):
            continue
        with torch.no_grad():
            try:
                # Use model.infer() to get output dict with plddt and coordinates
                result = model.infer(sequence)
                # Save mut.pdb
                if need_pdb:
                    output_pdb = model.convert_output_to_pdb(result, sequence)
                    with open(mut_pdb, "w") as f:
                        f.write(output_pdb)
                    print(f"Wrote {mut_pdb}",flush=True)
                # Save mut_esmf.pkl with plddt
                if need_pkl:
                    data = {
                        "plddt": result["plddt"],  # np.ndarray shape [L]
                        # Optionally add more keys if needed (e.g., "ptm", "mean_plddt", "positions")
                    }
                    with open(mut_pkl, "wb") as f:
                        pickle.dump(data, f)
                    print(f"Wrote {mut_pkl}",flush=True)
            except Exception as e:
                print(f"Failed for {mut_fasta}: {e}",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch ESMFold prediction for all mutants (ESM package version; outputs PDB & pKLDT pickle)")
    parser.add_argument("--data_parent", required=True, help="Parent folder containing all sample folders")
    parser.add_argument("--num_samples", type=int, default=None, help="Process only this many samples")
    args = parser.parse_args()
    main(args.data_parent, args.num_samples)