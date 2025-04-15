#!/bin/bash

source $HOME/.bash_profile

# ********************************** USER INPUT ********************************** #
turbsimbin='/full/path/to/your/binary/.../bin/turbsim'
basepath='/full/path/to/your/case/dir'

condList=('Cond00_v08.6_PL0.2_TI10' 'Cond01_v10.6_PL0.2_TI10' 'Cond02_v12.6_PL0.2_TI10')

nSeeds=6
# ******************************************************************************** #

for cond in ${condList[@]}; do
    for((seed=0; seed<$nSeeds; seed++)); do
       dir=$(printf "%s/%s/Seed_%01d" $basepath $cond $seed)
       echo "Running $dir/Low.inp"
       $turbsimbin $dir/Low.inp > $dir/log.low.seed$seed.txt 2>&1 &
   done
done

wait

