#!/bin/bash

source $HOME/.bash_profile

# ********************************** USER INPUT ********************************** #
fastfarmbin='/full/path/to/your/binary/.../bin/FAST.Farm'
basepath='/full/path/to/your/case/dir'

cond='Cond00_v08.6_TI10'
case='Case00_wdirp00'
seed=0
# ******************************************************************************** #

dir=$(printf "%s/%s/%s/Seed_%01d" $basepath $cond $case $seed)
cd $dir
export OMP_STACKSIZE="32 M"
echo "Running $dir/FFarm_mod.fstf with OMP_STACKSIZE=32M"
$fastfarmbin $dir/FFarm_mod.fstf > $dir/log.fastfarm.seed$seed.txt 2>&1

