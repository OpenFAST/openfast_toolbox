#!/bin/bash
#SBATCH --job-name=runFF
#SBATCH --output log.fastfarm_c0_c0_seed0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --time=10-00
#SBATCH --account=total

source $HOME/.bash_profile

echo "Working directory is" $SLURM_SUBMIT_DIR
echo "Job name is" $SLURM_JOB_NAME
echo "Job ID is " $SLURM_JOBID
echo "Job took the following nodes (SLURM_NODELIST)" $SLURM_NODELIST
echo "Submit time is" $(squeue -u $USER -o '%30j %20V' | grep -e $SLURM_JOB_NAME | awk '{print $2}')
echo "Starting job at: " $(date)

module purge
module load PrgEnv-intel/8.5.0
module load intel-oneapi-mkl/2024.0.0-intel
module load intel-oneapi
module load binutils
module load hdf5/1.14.3-intel-oneapi-mpi-intel

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
echo "Submitting $dir/FFarm_mod.fstf with OMP_STACKSIZE=32M"
$fastfarmbin $dir/FFarm_mod.fstf > $dir/log.fastfarm.seed$seed.txt 2>&1

echo "Ending job at: " $(date)
