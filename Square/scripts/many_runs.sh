# #!/bin/bash

for scale in 0.25 0.5 1 2 4; do
    for rate in 0.25 0.475; do

        exp_name='Dec6'
        X="s=$scale,r=$rate"
        sbatch -J "$X" run.sh $exp_name $scale $rate

    done 
done

# for bc in 'periodic' 'open'; do
#     exp_name='Dec6'
#     scale=2
#     rate=0.25
#     X="$bc"
#     sbatch -J "$X" run_individual.sh $exp_name $scale $rate $bc
# done
