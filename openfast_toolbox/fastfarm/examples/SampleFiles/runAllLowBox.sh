#!/bin/bash
#SBATCH --job-name=lowBox
#SBATCH --output log.lowBox
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --time=2-00
#SBATCH --account=total

source $HOME/.bash_profile

echo "Working directory is" $SLURM_SUBMIT_DIR
echo "Job name is" $SLURM_JOB_NAME
echo "Job ID is " $SLURM_JOBID
echo "Job took the following nodes (SLURM_NODELIST)" $SLURM_NODELIST
echo "Submit time is" $(squeue -u $USER -o '%30j %20V' | grep -e $SLURM_JOB_NAME | awk '{print $2}')
echo "Starting job at: " $(date)

nodelist=`scontrol show hostname $SLURM_NODELIST`
nodelist=($nodelist)
echo "Formatted list of nodes is: $nodelist"

module purge
module load PrgEnv-intel/8.5.0
module load intel-oneapi-mkl/2024.0.0-intel
module load intel-oneapi
module load binutils
module load hdf5/1.14.3-intel-oneapi-mpi-intel

# ********************************** USER INPUT ********************************** #
turbsimbin='/full/path/to/your/binary/.../bin/turbsim'
basepath='/full/path/to/your/case/dir'

condList=('Cond00_v08.6_PL0.2_TI10' 'Cond01_v10.6_PL0.2_TI10' 'Cond02_v12.6_PL0.2_TI10')

nSeeds=6
# ******************************************************************************** #

nodeToUse=0
for cond in ${condList[@]}; do
    currNode=${nodelist[$nodeToUse]}
    for((seed=0; seed<$nSeeds; seed++)); do
       dir=$(printf "%s/%s/Seed_%01d" $basepath $cond $seed)
       echo "Submitting $dir/Low.inp in node $currNode"
       srun -n1 -N1 --exclusive --nodelist=$currNode --mem-per-cpu=25000M $turbsimbin $dir/Low.inp > $dir/log.low.seed$seed.txt 2>&1 &
   done
   (( nodeToUse++ ))
done

wait

echo "Ending job at: " $(date)
