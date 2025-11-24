import click
import torch
from Bio import SeqIO
import numpy as np


aa_phy_chem_7_dict = {"A": [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476], "C": [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476], "D": [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635], "E": [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582], "F": [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318], "G": [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900], "H": [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106], "I": [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688], "K": [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265], "L": [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053], "M": [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001], "N": [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529], "P": [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106], "Q": [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371], "R": [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371], "S": [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212], "T": [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212], "V": [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900], "W": [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529], "Y": [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}


# Extend dictionary to handle rare amino acids
aa = aa_phy_chem_7_dict  # shorthand

# Convert lists to numpy arrays for arithmetic
for k in aa:
    aa[k] = np.array(aa[k], dtype=float)

# U — selenocysteine, treat as cysteine
aa["U"] = aa["C"].copy()

# O — pyrrolysine, treat as lysine
aa["O"] = aa["K"].copy()

# B — Aspartic acid (D) / Asparagine (N) ambiguity
aa["B"] = (aa["D"] + aa["N"]) / 2

# Z — Glutamic acid (E) / Glutamine (Q) ambiguity
aa["Z"] = (aa["E"] + aa["Q"]) / 2

# X — unknown; use mean over all 20 standard amino acids
standard = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
aa["X"] = np.mean([aa[k] for k in standard], axis=0)

aa_phy_chem_7_dict=aa


@click.command()
@click.option("--fasta_file", required=True, type=str)
@click.option("--saved_folder", required=True, type=str)
def main(fasta_file, saved_folder):
    seq = str(list(SeqIO.parse(fasta_file, "fasta"))[0].seq)
    data = torch.zeros((len(seq), 7), dtype=torch.float32)
    for index, value in enumerate(list(seq)):
        for i in range(7):
            data[index, i] = float(aa_phy_chem_7_dict[value][i])
    torch.save(data.detach().cpu().clone(), f"{saved_folder}/fixed_embedding.pt")


if __name__ == "__main__":
    main()
