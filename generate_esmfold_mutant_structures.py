import os
import torch
import esm
from Bio import SeqIO
import argparse

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
        if os.path.exists(mut_fasta):
            if not os.path.exists(mut_pdb):
                jobs.append((mut_fasta, mut_pdb))
    if num_samples is not None:
        jobs = jobs[:num_samples]
    print(f"Will predict {len(jobs)} mutant structures.",flush=True)

    # Predict and write PDBs
    for mut_fasta, mut_pdb in reversed(jobs):
        record = list(SeqIO.parse(mut_fasta, "fasta"))[0]
        sequence = str(record.seq)
        if os.path.exists(mut_pdb):
            continue
        with torch.no_grad():
            try:
                output_pdb = model.infer_pdb(sequence)
                with open(mut_pdb, "w") as f:
                    f.write(output_pdb)
                print(f"Wrote {mut_pdb}",flush=True)
            except Exception as e:
                print(f"Failed for {mut_fasta}: {e}",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch ESMFold prediction for all mutants (ESM package version)")
    parser.add_argument("--data_parent", required=True, help="Parent folder containing all sample folders")
    parser.add_argument("--num_samples", type=int, default=None, help="Process only this many samples")
    args = parser.parse_args()
    main(args.data_parent, args.num_samples)