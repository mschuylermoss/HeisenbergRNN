# #!/bin/bash

######## Training with different scales and rates for enlarging scheme
for rate in 0.475 0.25;
    do
    exp_name='Nov29'

    bc='periodic'
    X="p|$rate"
    sbatch -J "$X" run.sh $exp_name $bc $rate

    bc='open'
    X="o|$rate"
    sbatch -J "$X" run.sh $exp_name $bc $rate

done