#!/bin/bash
source $HOME/.bash_profile

# ********************************** USER INPUT ********************************** #
turbsimbin='/full/path/to/your/binary/.../bin/turbsim'
basepath='/full/path/to/your/case/dir'

condList=('Cond00_v08.6_PL0.2_TI10' 'Cond01_v10.6_PL0.2_TI10' 'Cond02_v12.6_PL0.2_TI10')

caseList=('Case00_wdirp00_WSfalse_YMfalse' 'Case01_wdirp00_WStrue_YMfalse')

nSeeds=6
nTurbines=12
# ******************************************************************************** #

for cond in ${condList[@]}; do
    for case in ${caseList[@]}; do
        for ((seed=0; seed<$nSeeds; seed++)); do
            for ((t=1; t<=$nTurbines; t++)); do
                dir=$(printf "%s/%s/%s/Seed_%01d/TurbSim" $basepath $cond $case $seed)
                echo "Submitting $dir/HighT$t.inp"
                $turbsimbin $dir/HighT$t.inp > $dir/log.hight$t.seed$seed.txt 2>&1 &
                sleep 0.1
            done
        done
    done
done

wait

