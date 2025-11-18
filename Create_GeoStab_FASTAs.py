import pandas as pd
import os

data_dir = '/projects/ashehu/amoldwin/datasets/protein_melting_temps/'
out_dir = '/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/'

for csv_file in ["S4346.csv", "S571.csv"]:
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    for idx, row in df.iterrows():
        name = row['name']

        # Make parent and subfolders
        parent_folder = os.path.join(out_dir, name)
        wt_folder = os.path.join(parent_folder, "wt_data")
        mut_folder = os.path.join(parent_folder, "mut_data")
        os.makedirs(wt_folder, exist_ok=True)
        os.makedirs(mut_folder, exist_ok=True)

        # Write WT FASTA
        with open(os.path.join(wt_folder, "wt.fasta"), "w") as fwt:
            fwt.write(f">{name}_wt\n{row['wt_seq']}\n")

        # Write mutant FASTA
        with open(os.path.join(mut_folder, "mut.fasta"), "w") as fmut:
            fmut.write(f">{name}_mut\n{row['mut_seq']}\n")

        # Create mut_info.csv in the parent_folder
        mut_info_dict = {
            "version": "Seq",         # assuming sequence mode for dTm
            "seq": row['wt_seq'],
            "mut_pos": "",            # Needs to be set appropriately
            "mut_res": "",            # Needs to be set appropriately
            "pH": row['pH'] if 'pH' in row else "",
            "temperature": row['dTm'] if 'dTm' in row else "",  # can treat dTm as temperature
        }
        # Try to infer mutation position and residue if possible
        # This compares first difference between wt_seq and mut_seq
        wt_seq = row['wt_seq']
        mut_seq = row['mut_seq']
        mut_pos = ''
        mut_res = ''
        if isinstance(wt_seq, str) and isinstance(mut_seq, str) and len(wt_seq) == len(mut_seq):
            for i, (a, b) in enumerate(zip(wt_seq, mut_seq)):
                if a != b:
                    mut_pos = i      # 0-based position
                    mut_res = b
                    break
        mut_info_dict["mut_pos"] = mut_pos
        mut_info_dict["mut_res"] = mut_res

        # Save mut_info.csv: required format is index "test" for the single example
        mut_info_df = pd.DataFrame(mut_info_dict, index=["test"])
        mut_info_df.to_csv(os.path.join(parent_folder, "mut_info.csv"))