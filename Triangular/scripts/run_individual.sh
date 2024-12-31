#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=1                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=0-06:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
bc=$2
which_MS=$3
weight_sharing=$4
echo $exp_name
echo $bc
echo $which_MS Marshall Sign
echo $weight_sharing weights shared

echo "Launching job rate=${rate} with ${SLURM_CPUS_PER_TASK} CPUs"
srun --output="./outputs/heisenberg_6x6_${bc}_${which_MS}MS_${weight_sharing}WS.out" python -u enlargeTriangular_scaling.py \
                                                                            --experiment_name $exp_name \
                                                                            --bc $bc \
                                                                            --which_MS $which_MS \
                                                                            --weight_sharing $weight_sharing \
                                                                            --schedule "6x6" \
                                                                            # --scale $scale \
                                                                            # --rate $rate