import pandas as pd
import os

# Parent and output folders (adjust as needed)
data_dir = '/projects/ashehu/amoldwin/datasets/mutation/'   # Update to your actual data folder
out_dir = './data/patho/patho_FASTA/'   # Update to your desired output folder

for split in ["train", "val", "test"]:
    csv_file = f"patho_{split}.csv"
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    for idx, row in df.iterrows():
        # Use prot_variant as the parent folder name for uniqueness
        name = row['prot_variant']

        parent_folder = os.path.join(out_dir, name)
        wt_folder = os.path.join(parent_folder, "wt_data")
        mut_folder = os.path.join(parent_folder, "mut_data")
        os.makedirs(wt_folder, exist_ok=True)
        os.makedirs(mut_folder, exist_ok=True)

        # Write WT FASTA
        with open(os.path.join(wt_folder, "wt.fasta"), "w") as fwt:
            fwt.write(f">{name}_wt\n{row['wt_sequence']}\n")

        # Write mutant FASTA
        with open(os.path.join(mut_folder, "mut.fasta"), "w") as fmut:
            fmut.write(f">{name}_mut\n{row['mutated_sequence']}\n")

        # Create mut_info.csv in the parent_folder
        mut_info_dict = {
            "version": "Seq",
            "seq": row['wt_sequence'],
            "mut_pos": "",    # will be set below
            "mut_res": "",    # will be set below
        }

        # Find mutation position and residue
        mut_pos = row.get('1indexed_prot_mt_pos', '')
        mut_res = row.get('mt_aa_1letter', '')

        # Convert mutation position to 0-based index if possible
        try:
            mut_pos = int(mut_pos) - 1 if mut_pos != '' else ''
        except Exception:
            mut_pos = ''

        mut_info_dict["mut_pos"] = mut_pos
        mut_info_dict["mut_res"] = mut_res

        # Save mut_info.csv with "test" index
        mut_info_df = pd.DataFrame(mut_info_dict, index=["test"])
        mut_info_df.to_csv(os.path.join(parent_folder, "mut_info.csv"))