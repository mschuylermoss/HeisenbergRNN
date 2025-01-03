#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=1                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=3-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venvcuda/bin/activate

exp_name=$1

echo $exp_name
echo "Launching job Open with ${SLURM_CPUS_PER_TASK} CPUs"
srun --output="./outputs/heisenberg_open.out" --ntasks=1 python -u enlargeTriangular_scaling.py \
                                                                --experiment_name $exp_name \
                                                                --bc "open" \
                                                                --which_MS "Square" \
                                                                --schedule "rate"
wait