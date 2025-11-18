import pandas as pd
import os
data_dir = '/projects/ashehu/amoldwin/datasets/protein_melting_temps/'
out_dir = '/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/'
for csv_file in ["S4346.csv", "S571.csv"]:
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    for idx, row in df.iterrows():
        name = row['name']

        # Make parent and subfolders
        parent_folder = os.path.join(out_dir,name)
        wt_folder = os.path.join(parent_folder, "wt_data")
        mut_folder = os.path.join(parent_folder, "mut_data")
        os.makedirs(wt_folder, exist_ok=True)
        os.makedirs(mut_folder, exist_ok=True)

        # Write WT FASTA
        with open(os.path.join(wt_folder, "wt.fasta"), "w") as fwt:
            fwt.write(  f">{name}_wt\n{row['wt_seq']}\n")

        # Write mutant FASTA
        with open(os.path.join(mut_folder, "mut.fasta"), "w") as fmut:
            fmut.write(f">{name}_mut\n{row['mut_seq']}\n")