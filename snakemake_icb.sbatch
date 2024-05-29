#!/bin/bash

#SBATCH -o logs/snakemake_output_%j.txt
#SBATCH -e logs/snakemake_output_%j.txt
#SBATCH -J sp_snakemake
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 10-00:00:00
#SBATCH --nice=0
#SBATCH -p cpu_p
#SBATCH --qos cpu_long

source $HOME/.bashrc
conda activate spapros-smk

export SINGULARITY_TMPDIR="/lustre/groups/ml01/workspace/lena.strasser/tmp"
export APPTAINER_TMPDIR="/lustre/groups/ml01/workspace/lena.strasser/tmp"

# submit main snakemake job
snakemake --profile config/cluster/config_icb --rerun-triggers mtime -n
# and in cmd line: bash snakemake_icb.sbatch

# or submit jobs without main snakemake job in cmd line using:
# snakemake --profile config/cluster/config_icb
