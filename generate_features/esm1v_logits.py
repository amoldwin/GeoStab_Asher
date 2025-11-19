import click
import torch
from Bio import SeqIO
import os

def run_esm1v_logits(model_indices, fasta_files, saved_folders):
    if len(fasta_files) != len(saved_folders) or len(fasta_files) != len(model_indices):
        raise ValueError("Number of fasta_files, saved_folders, and model_indices must all match")

    # Process in batch by model index
    jobs = []
    for fasta_file, saved_folder, idx in zip(fasta_files, saved_folders, model_indices):
        record = list(SeqIO.parse(fasta_file, "fasta"))[0]
        jobs.append((record.id, str(record.seq), fasta_file, saved_folder, idx))
    data = [(jid, seq) for jid, seq, _, _, _ in jobs]

    model_index_set = sorted(set(model_indices))
    for model_index in model_index_set:
        # Find all jobs for this index
        jobs_for_idx = [job for job in jobs if job[4] == model_index]
        if not jobs_for_idx:
            continue
        batch_labels = [jid for jid, _, _, _, _ in jobs_for_idx]
        batch_strs = [seq for _, seq, _, _, _ in jobs_for_idx]
        output_dirs = [folder for _, _, _, folder, _ in jobs_for_idx]
        with torch.no_grad():
            model, alphabet = torch.hub.load(
                "facebookresearch/esm:main",
                f"esm1v_t33_650M_UR90S_{model_index}",
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.eval().to(device)
            batch_converter = alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter(list(zip(batch_labels, batch_strs)))
            batch_tokens = batch_tokens.to(device)
            token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
            logits_33 = token_probs[:, 1:-1, :]  # shape: (batch, seq_len, vocab)

            amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
            esm1v_amino_acid_dict = {aa: alphabet.get_idx(aa) for aa in amino_acid_list}

            # logits_20: (batch, seq_len, 20)
            logits_20 = torch.zeros((logits_33.shape[0], logits_33.shape[1], 20), dtype=torch.float32)
            for sample_idx in range(logits_33.shape[0]):
                seq = batch_strs[sample_idx]
                for wt_pos, wt_amino_acid in enumerate(seq):
                    for mut_pos, mut_amino_acid in enumerate(amino_acid_list):
                        logits_20[sample_idx, wt_pos, mut_pos] = (
                            logits_33[sample_idx, wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] -
                            logits_33[sample_idx, wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]
                        )

            for i, out_folder in enumerate(output_dirs):
                os.makedirs(out_folder, exist_ok=True)
                torch.save(logits_20[i].detach().cpu().clone(), f"{out_folder}/esm1v-{model_index}.pt")

@click.command()
@click.option("--model_indices", multiple=True, required=True, type=str, help="ESM1v model indices (e.g. 1 2 3 4 5). Each sample should have an index")
@click.option("--fasta_files", multiple=True, required=True, type=str, help="FASTA files to process (same order as saved_folders/model_indices)")
@click.option("--saved_folders", multiple=True, required=True, type=str, help="Output folders (must match order of fasta_files)")
def main(model_indices, fasta_files, saved_folders):
    run_esm1v_logits(model_indices, fasta_files, saved_folders)

if __name__ == "__main__":
    main()