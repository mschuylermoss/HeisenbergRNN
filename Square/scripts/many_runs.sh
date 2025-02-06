# #!/bin/bash

for scale in 0.25 0.5 1 2 4; do 
    for rate in 0.475 0.25; do

        exp_name='YourExperimentName'
        bc='open'
        X="O|s=$scale|r=$rate"
        sbatch -J "$X" run_oneNode_fourGPU.sh $exp_name $bc $scale $rate

        exp_name='YourExperimentName'
        bc='periodic'
        X="P|s=$scale|r=$rate"
        sbatch -J "$X" run_oneNode_fourGPU.sh $exp_name $bc $scale $rate

    done
done
