# #!/bin/bash

joblist=$(sq -h --format="%j")

######## Training with different scales and rates for enlarging scheme

for scale in 0.25 0.5 1.0 2.0
do 
    for rate in 0.475 0.25
    do 

    exp_name='Nov29'

    bc='periodic'
    X="p|$scale|$rate"
    sbatch -J "$X" --export="exp_name=$exp_name,bc=$bc,scale=$scale,rate=$rate" enlargeSquare_scaling_submit.sh

    bc='open'
    X="o|$scale|$rate"
    sbatch -J "$X" --export="exp_name=$exp_name,bc=$bc,scale=$scale,rate=$rate" enlargeSquare_scaling_submit.sh

    done
done

######## Training with enlarging or from scratch

# for scale in  1 #0.25 0.5 2
# do
#     for nh in 256 #128 #512 
#     do

#     # nh=256
#     na=10000
#     T0=0.
    
#     exp_name='Square_16x16_fromScratch'

#     bc='periodic'
#     X="p|$nh|$scale"
#     sbatch -J "$X" --export="seed=100,num_units=$nh,exp_name=$exp_name,bc=$bc,na=$na,scale=$scale,T0=$T0,lsym=1" Square10x10_submit.sh
    
#     bc='open'
#     X="o|$nh|$scale"
#     sbatch -J "$X" --export="seed=100,num_units=$nh,exp_name=$exp_name,bc=$bc,na=$na,scale=$scale,T0=$T0,lsym=1" Square10x10_submit.sh

#     done
# done
# sleep 0.5s

