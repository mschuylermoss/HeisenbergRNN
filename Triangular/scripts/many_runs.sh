# #!/bin/bash

######## Training with different scales and rates for enlarging scheme
# for scale in 0.25 0.5 1 2 4; do

#     exp_name='Dec6'
#     X="scale=$scale"
#     sbatch -J "$X" run.sh $exp_name $scale

# done

exp_name='initTri_Dec29_3'

for which_MS in "Square" "Triangular"; do

    # bc="open"
    
    # weight_sharing="all"
    # X="O|A|$which_MS"
    # sbatch -J "$X" run_individual.sh $exp_name $bc $which_MS $weight_sharing

    # weight_sharing="sublattice"
    # X="O|SL|$which_MS"
    # sbatch -J "$X" run_individual.sh $exp_name $bc $which_MS $weight_sharing

    bc="periodic"
    
    weight_sharing="all"
    X="P|A|$which_MS"
    sbatch -J "$X" run_individual.sh $exp_name $bc $which_MS $weight_sharing

    weight_sharing="sublattice"
    X="P|SL|$which_MS"
    sbatch -J "$X" run_individual.sh $exp_name $bc $which_MS $weight_sharing

done