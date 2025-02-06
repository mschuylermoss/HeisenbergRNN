#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --nodes=1                                                   
#SBATCH --ntasks-per-node=4                                         
#SBATCH --gpus-per-task=1                                           
#SBATCH --cpus-per-task=16                                          
#SBATCH --mem-per-cpu=4G                                            
#SBATCH --time=3-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
scale=$2
echo $exp_name
echo scale=$scale

for bc in 'open' 'periodic'; do
    for rate in 0.25 0.475; do
        while [ "$(jobs -p | wc -l)" -ge "$SLURM_NTASKS" ]; do
            sleep 30
        done
        echo "Launching job bc=${bc}, rate=${rate} with ${SLURM_CPUS_PER_TASK} CPUs"
        srun --output="./outputs/heisenberg_${scale}_${bc}_${rate}.out" --ntasks=1 python -u enlargeSquare_scaling.py --seed 100 --experiment_name $exp_name --bc $bc --units 256 --scale $scale --rate $rate&
    done
done
wait