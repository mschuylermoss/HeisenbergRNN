#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --nodes=1                                                   # node count
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16                                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=2                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=3-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venvcuda/bin/activate

#exp_name="Jan3Even"
#
#echo $exp_name
#echo "Launching job Open with ${SLURM_CPUS_PER_TASK} CPUs"
#srun --output="./outputs/heisenberg_SquareEven.out" --ntasks=1 python -u enlargeTriangular_scaling.py \
#                                                                --experiment_name $exp_name \
#                                                                --bc "open" \
#                                                                --which_MS "Square" \
#                                                                --schedule "even" &
#sleep 0.1
#echo "And another one"
exp_name="Jan3"
srun --output="./outputs/heisenberg_SquareComm.out" --ntasks=1 python -u enlargeTriangular_scaling.py \
                                                                --experiment_name $exp_name \
                                                                --bc "open" \
                                                                --rate 0.158\
                                                                --which_MS "Square" \
                                                                --schedule "rate24" &
sleep 0.1
srun --output="./outputs/heisenberg_TriangularComm.out" --ntasks=1 python -u enlargeTriangular_scaling.py \
                                                                --experiment_name $exp_name \
                                                                --bc "open" \
                                                                --rate 0.158\
                                                                --which_MS "Triangular" \
                                                                --schedule "rate24" &
sleep 0.1
wait