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

        # Do NOT slice here; keep all tokens (CLS/EOS/PAD) and trim per sequence below
        result = model(batch_tokens, repr_layers=[33], return_contacts=False)
        representations = result["representations"][33]  # (batch, token_len, 1280)

    # Write output files, slicing per sequence using its true length
    for i, out_folder in enumerate(saved_folders):
        os.makedirs(out_folder, exist_ok=True)
        fasta_len = len(batch_strs[i])

        # Layout is: [CLS] + residues (fasta_len) + [EOS] + PAD...
        # So residues are at positions 1 .. fasta_len inclusive (Python slice 1:1+fasta_len)
        seq_repr = representations[i, 1:1 + fasta_len, :].detach().cpu().clone()
        emb_len = seq_repr.shape[0]

        if fasta_len != emb_len:
            raise RuntimeError(
                f"ESM2 embedding length mismatch for {out_folder}:\n"
                f"FASTA length: {fasta_len}\n"
                f"ESM2 embedding (after trimming) length: {emb_len}\n"
                f"Input sequence: {batch_strs[i]}\n"
            )

        torch.save(seq_repr, f"{out_folder}/esm2.pt")

@click.command()
@click.option("--fasta_files", multiple=True, required=True, type=str,
              help="Input FASTA files (can be more than one for batching)")
@click.option("--saved_folders", multiple=True, required=True, type=str,
              help="Output folder paths to save embeddings (must match order of fasta_files)")
def main(fasta_files, saved_folders):
    run_esm2_embedding(fasta_files, saved_folders)

if __name__ == "__main__":
    main()
