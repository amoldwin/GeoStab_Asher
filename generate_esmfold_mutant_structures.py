import os
from transformers import EsmForProteinFolding, AutoTokenizer
from Bio import SeqIO
import torch
import argparse

def predict_structure(sequence, model, tokenizer):
    # Make folding prediction and return PDB string
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    with torch.no_grad():
        folding_output = model(**inputs)
    pdb_str = folding_output.get("pdb_str", None)
    if pdb_str is not None:
        return pdb_str[0]
    else:
        return None

def main(data_parent, num_samples=None, device="cuda:0"):
    # Load ESMFold HuggingFace model & tokenizer only once
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print('loaded model and tokenizer',flush=True)
    jobs = []
    for sample_name in os.listdir(data_parent):
        mut_fasta = os.path.join(data_parent, sample_name, "mut_data", "mut.fasta")
        mut_pdb = os.path.join(data_parent, sample_name, "mut_data", "mut.pdb")
        if os.path.exists(mut_fasta):
            if not os.path.exists(mut_pdb):
                jobs.append((mut_fasta, mut_pdb))

    if num_samples is not None:
        jobs = jobs[:num_samples]

    print(f"Predicting mutant structures for {len(jobs)} samples on device {device}.",flush=True)

    for mut_fasta, mut_pdb in jobs:
        record = list(SeqIO.parse(mut_fasta, "fasta"))[0]
        sequence = str(record.seq)
        pdb_str = predict_structure(sequence, model, tokenizer)
        if pdb_str is not None:
            with open(mut_pdb, "w") as out_f:
                out_f.write(pdb_str)
            print(f"Wrote {mut_pdb}",flush=True)
        else:
            print(f"Failed for {mut_fasta}",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch ESMFold HuggingFace inference for all mutants")
    parser.add_argument("--data_parent", required=True, help="Parent folder containing all sample folders")
    parser.add_argument("--num_samples", type=int, default=None, help="Process only this many samples")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for HuggingFace model ('cuda:0', 'cpu', etc.)")
    args = parser.parse_args()
    main(args.data_parent, args.num_samples, args.device)