import os
import torch
import esm
from Bio import SeqIO
import argparse
import pickle

def main(data_parent, num_samples=None, batch_size=8):
    print("CUDA available:", torch.cuda.is_available())
    model = esm.pretrained.esmfold_v1()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    print("Loaded ESMFold model to", device, flush=True)

    # Collect jobs
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

    # Batch inference
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i+batch_size]
        batch_records = []
        batch_pdbs = []
        batch_pkls = []
        batch_names = []

        for mut_fasta, mut_pdb, mut_pkl in batch:
            record = list(SeqIO.parse(mut_fasta, "fasta"))[0]
            batch_records.append(str(record.seq))
            batch_pdbs.append(mut_pdb)
            batch_pkls.append(mut_pkl)
            batch_names.append(record.id)

        if not batch_records:
            continue
        with torch.no_grad():
            # Batched inference
            try:
                results = model.infer(batch_records)  # Returns a list of dicts or a dict with each result
                for idx, result in enumerate(results):
                    sequence = batch_records[idx]
                    out_pdb = batch_pdbs[idx]
                    out_pkl = batch_pkls[idx]
                    # Save PDB
                    if True:#not os.path.exists(out_pdb):
                        pdb_string = model.convert_output_to_pdb(result, sequence)
                        with open(out_pdb, "w") as f:
                            f.write(pdb_string)
                        print(f"Wrote {out_pdb}", flush=True)
                    # Save pKLDDT pickle
                    if not os.path.exists(out_pkl):
                        data = {"plddt": result["plddt"]}
                        with open(out_pkl, "wb") as f:
                            pickle.dump(data, f)
                        print(f"Wrote {out_pkl}", flush=True)
            except Exception as e:
                print(f"Batch failed at {batch_names}: {e}",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parent", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args.data_parent, args.num_samples, args.batch_size)