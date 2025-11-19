import click
import torch
from Bio import SeqIO
import os

def run_esm2_embedding(fasta_files, saved_folders):
    if len(fasta_files) != len(saved_folders):
        raise ValueError("Number of fasta_files must match number of saved_folders")
    # Prepare batched inputs
    batch_labels = []
    batch_strs = []
    for fpath in fasta_files:
        record = list(SeqIO.parse(fpath, "fasta"))[0]
        batch_labels.append(record.id)
        batch_strs.append(str(record.seq))

    with torch.no_grad():
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter(list(zip(batch_labels, batch_strs)))
        batch_tokens = batch_tokens.to(device)
        result = model(batch_tokens, repr_layers=[33], return_contacts=False)
        representations = result["representations"][33][:, 1:-1, :]  # shape: (batch, seq_len, 1280)

    # Write output files
    for i, out_folder in enumerate(saved_folders):
        os.makedirs(out_folder, exist_ok=True)
        torch.save(representations[i].detach().cpu().clone(), f"{out_folder}/esm2.pt")

@click.command()
@click.option("--fasta_files", multiple=True, required=True, type=str, help="Input FASTA files (can be more than one for batching)")
@click.option("--saved_folders", multiple=True, required=True, type=str, help="Output folder paths to save embeddings (must match order of fasta_files)")
def main(fasta_files, saved_folders):
    run_esm2_embedding(fasta_files, saved_folders)

if __name__ == "__main__":
    main()