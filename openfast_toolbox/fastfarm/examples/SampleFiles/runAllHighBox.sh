#!/bin/bash
#SBATCH --job-name=highBox
#SBATCH --output log.highBox
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=104
#SBATCH --time=4:00:00
#SBATCH --mem=250G
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
turbsimbin='/full/path/to/your/binary/.../bin/turbsim'
basepath='/full/path/to/your/case/dir'

condList=('Cond00_v08.6_PL0.2_TI10' 'Cond01_v10.6_PL0.2_TI10' 'Cond02_v12.6_PL0.2_TI10')

caseList=('Case00_wdirp00_WSfalse_YMfalse' 'Case01_wdirp00_WStrue_YMfalse')

nSeeds=6
nTurbines=12
# ******************************************************************************** #

rampercpu=$((249000/36))

for cond in ${condList[@]}; do
    for case in ${caseList[@]}; do
        for ((seed=0; seed<$nSeeds; seed++)); do
            for ((t=1; t<=$nTurbines; t++)); do
                dir=$(printf "%s/%s/%s/Seed_%01d/TurbSim" $basepath $cond $case $seed)
                echo "Submitting $dir/HighT$t.inp"
                srun -n1 -N1 --exclusive --mem-per-cpu=$rampercpu $turbsimbin $dir/HighT$t.inp > $dir/log.hight$t.seed$seed.txt 2>&1 &
                sleep 0.1
            done
        done
    done
done

wait

echo "Ending job at: " $(date)
