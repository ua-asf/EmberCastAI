#!/bin/bash

#SBATCH --job-name="EmberCast-CPU-Training"
#SBATCH --partition=t1small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=dmmaltos@alaska.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/training_cpu.out
#SBATCH --error=logs/training_cpu.err

module purge
module load slurm

# Load conda 
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment 
conda activate embercast

# Move to your project directory
cd $SLURM_SUBMIT_DIR/EmberCastAI

# Set environment variables for multithreaded CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run your script
python training.py
