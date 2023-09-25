#!/bin/bash -l
#SBATCH --job-name=fourier        # name
#SBATCH -p skl
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32           # number of cores per tasks
#SBATCH --comment pytorch
#SBATCH --time 1-23:59:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=fourier-%j.out           # output file name
#SBATCH -e fourier-%j.err


module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 conda/pytorch_1.12.0 singularity/3.11.0


# srun --jobid $SLURM_JOBID bash -c "python prover/evaluate.py --data-path data/leandojo_benchmark/random/random_split/random1  --ckpt_path lightning_logs/version_232973/checkpoints/last-v1.ckpt --split test --num-cpus 16"
python fourier.py