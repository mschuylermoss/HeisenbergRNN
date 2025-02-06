#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100-80gb                                                # for multi-gpu jobs ask for larger memory GPU
#SBATCH --job-name=heisenberg                                       # create a short name for your job
#SBATCH --nodes=1                                                   
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=4                                           
#SBATCH --cpus-per-task=16                                          
#SBATCH --mem-per-cpu=4G                                            
#SBATCH --time=5-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
bc=$2
scale=$3
rate=$4
echo $exp_name
echo $bc
echo scale=$scale
echo rate=$rate

echo "Launching job with ${SLURM_CPUS_PER_TASK} CPUs"
srun --output="./outputs/heisenberg_${scale}_${bc}_${rate}.out" python -u enlargeSquare_scaling.py --seed 100 --experiment_name $exp_name --bc $bc --units 256 --scale $scale --rate $rate 