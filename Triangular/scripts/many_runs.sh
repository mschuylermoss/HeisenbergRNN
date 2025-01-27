# #!/bin/bash

####### Training with different scales and rates for enlarging scheme
exp_name='Jan3'
bc="periodic"
schedule="6x6"
for scale in 1 2; do
 for rate in 0.25 0.475; do
     X="s=$scale|r=$rate"
     sbatch -J "$X" run.sh $exp_name $scale $rate $bc $schedule
     sleep 0.05
  done
done

X="Open"
sbatch -J "$X" run_open.sh $exp_name
