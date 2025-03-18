# #!/bin/bash

####### Training with different scales and rates for enlarging scheme
exp_name='Jan15'
bc="periodic"
schedule="rate"
T0=0.25
which_MS="Square"
for scale in 1; do
     X="s=$scale"
     sbatch -J "$X" run.sh $exp_name $scale $T0 $bc $which_MS
     sleep 0.05
done

# X="Open"
# sbatch -J "$X" run_open.sh $exp_name
