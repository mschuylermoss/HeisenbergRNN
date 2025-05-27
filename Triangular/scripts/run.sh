#!/bin/bash
#SBATCH -p h200q
#SBATCH --job-name=heisenberg
#SBATCH --nodes=1                                                
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-cpu=32G                                           
#SBATCH --time=1-00:00:00
#SBATCH --output=./outputs/heisenberg.%j.%N.out

source ../../venv/bin/activate

exp_name=$1
scale=$2
rate=$3
T0=$4
bc=$5
which_MS=$6
schedule=$7

echo $exp_name
echo scale = $scale
echo rate = $rate
echo T0 = $T0
echo boundary condition = $bc
echo MS = $which_MS
echo which schedule = $schedule

python -u enlargeTriangular_scaling.py \
            --experiment_name $exp_name \
            --bc $bc \
            --T0 $T0 \
            --which_MS $which_MS \
            --scale $scale\
            --schedule $schedule\
            --rate $rate \
