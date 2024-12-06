#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --nodes=1                                                   # node count
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16                                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=1                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=3-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
bc=$2
rate=$3
echo $exp_name
echo $bc
echo $rate

for scale in 0.25 0.5 1 2; do
    while [ "$(jobs -p | wc -l)" -ge "$SLURM_NTASKS" ]; do
        sleep 30
    done
    echo "Launching job ${scale} with ${SLURM_CPUS_PER_TASK} CPUs"
    srun --output="./outputs/heisenberg_${scale}_${bc}_${rate}.out" --ntasks=1 python -u enlargeSquare_scaling.py --seed 100 --experiment_name $exp_name	--bc $bc --units 256 --scale $scale --rate $rate&
done
wait