#!/bin/bash
#SBATCH --job-name=intramutatex
#SBATCH --output=/projects/ashehu/amoldwin/logs/intragenic/-%j.out
#SBATCH --error=/projects/ashehu/amoldwin/logs/intragenic/-%j.err
#SBATCH --mail-user=<amoldwin@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=gpuq        # contrib-gpuq, gpuq
#SBATCH --qos=gpu # gpu, cs_dept
#SBATCH --nodes=1 
#SBATCH --gres=gpu:A100.80gb:1
##SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=128G 
#SBATCH --time=02-00:30:00


# source /projects/ashehu/amoldwin/envs/mutation/bin/activate
export TORCH_HOME=/scratch/amoldwin/torch_cache
export TRANSFORMERS_CACHE=/scratch/amoldwin/HF_cache

source ../miniconda/bin/activate 
conda activate openfold_venv
ml cuda/11
## pip uninstall -y torch
## conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge


##python generate_esmfold_mutant_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/
##python generate_esmfold_wt_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/

python generate_esmfold_wt_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/patho/patho_FASTA/ ##--start_from_longest
python generate_esmfold_mutant_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/patho/patho_FASTA/ ##--start_from_longest


##python -m run_feature_scripts --use_esmfold_wt

##python generate_esmfold_mutant_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/patho/patho_FASTA/
##python generate_esmfold_wt_structures.py --data_parent /projects/ashehu/amoldwin/GeoStab/data/patho/patho_FASTA/
##python -m train_code.geodtm.train_geodtm_no_plddt