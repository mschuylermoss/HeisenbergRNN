# #!/bin/bash


exp_name='Jan15'
bc="periodic"
schedule="rate"
T0=1.0
for which_MS in 'Triangular'; do
  for scale in 4; do
    for rate in 0.158; do
        X="s=$scale|T0=$T0"
        sbatch -J "$X" run.sh $exp_name $scale $rate $T0 $bc $which_MS $schedule 
        sleep 0.05
    done
  done
done

# exp_name='Jan15_signrules'
# bc="periodic"
# schedule="6x6"
# T0=1.0
# for which_MS in 'No'; do # 'No' 'Square' 'Triangular'
#   for scale in 8; do
#     for rate in 0.475; do
#       for T0 in 0.25; do
#           X="s=$scale|T0=$T0"
#           sbatch -J "$X" run.sh $exp_name $scale $rate $T0 $bc $which_MS $schedule 
#           sleep 0.05
#       done
#     done
#   done
# done

# X="times"
# exp_name="TrainTimes"
# schedule="times"
# which_MS="Triangular"
# bc="periodic"
# scale=1.0
# rate=0.475
# T0=1.0
# sbatch -J "$X" run.sh $exp_name $scale $rate $T0 $bc $which_MS $schedule 

# X="Even"
# exp_name='Jan3Even'
# schedule="even"
# bc='open'
# which_MS='Square'
# rate=0.475
# for scale in 1; do
#  for T0 in 0.25; do
#      sbatch -J "$X" run.sh $exp_name $scale $rate $T0 $bc $which_MS $schedule 
#      sleep 0.05
#   done
# done
