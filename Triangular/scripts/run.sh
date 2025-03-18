#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --nodes=1                                                   # node count
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32                                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=2                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G                                            # memory per cpu-core (4G is default)
#SBATCH --time=3-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
scale=$2
T0=$3
bc=$4
which_MS=$5

echo exp_name=$exp_name
echo scale=$scale
echo T0=$T0
echo boundary=$bc
echo MS=$which_MS

for rate in 0.25 0.475; do
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
                                                                                                        --rate $rate\
                                                                                                        --weight_sharing "sublattice"&
    sleep 0.05
done
wait