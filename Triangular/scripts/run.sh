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

source ../../venvcuda/bin/activate

exp_name=$1
scale=$2
rate=$3
bc=$4
schedule=$5

echo $exp_name
echo $scale
echo $rate
echo $bc
echo $schedule

for T0 in 0.25 1.; do
    for which_MS in "Square" "Triangular"; do
        while [ "$(jobs -p | wc -l)" -ge "$SLURM_NTASKS" ]; do
            sleep 30
        done
        echo "Launching job T0=${T0}, ms=${which_MS} with ${SLURM_CPUS_PER_TASK} CPUs"
        srun --output="./outputs/heisenberg_${scale}_${rate}_${bc}_T${T0}_ms${which_MS}.out" --ntasks=1 python -u enlargeTriangular_scaling.py \
                                                                                                          --experiment_name $exp_name \
                                                                                                          --bc $bc \
                                                                                                          --T0 $T0 \
                                                                                                          --which_MS $which_MS \
                                                                                                          --scale $scale\
                                                                                                          --schedule $schedule\
                                                                                                          --rate $rate&
    done
done
wait