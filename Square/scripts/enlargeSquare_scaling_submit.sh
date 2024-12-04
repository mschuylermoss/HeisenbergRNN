#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=t4:2
#SBATCH --account=def-rgmelko
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=msmoss@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/square-%A.out

module load python/3

source ../scripts/ENV/bin/activate

echo $exp_name
echo $bc
echo $scale
echo $rate

python -u enlargeSquare_scaling.py \
	--seed 100 \
	--experiment_name $exp_name \
	--bc $bc \
	--units 256 \
	--scale $scale \
	--rate $rate \
	