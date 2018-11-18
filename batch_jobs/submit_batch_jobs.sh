#!/bin/sh

## Set number of nodes for the job
#SBATCH -N 1

## Set number of tasks for the job
#SBATCH -n 1

## Specify to only use GPU nodes; no. per node
##SBATCH --gres=gpu:1

## Specify requested time
## day-hr
#SBATCH -t 5-00:00:00

## Specify stdout, stderr log; default is slurm-jobid.out
#SBATCH -o output.log
#SBATCH -e error.log

## Specify what to notify: BEGIN, END, FAIL...., ALL
##SBATCH --mail-type=FAIL
##SBATCH --mail-user arbint@bnl.gov

## Print time and date and hostname
echo "JOB $SLURM_JOB_ID is running on $SLURM_JOB_NODELIST "
echo "Date:"
date
echo "Hostname:"
hostname

## Load cuda module for GPU job
module load anaconda2
module load cuda
source activate envDeepLearningWithProtoDUNE

echo "Running python train_model.py"
cd ../
srun python train_model.py -o Training -e Default
