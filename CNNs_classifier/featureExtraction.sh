#!/bin/bash
module load essentia/2.1_python-2.7.3
# Name the process
# ----------------
#$ -N FeaExtr_$JOB_ID
#
# Call from the current working directory; no need to cd; using default.q configuration, HPC will distribute the best choice of the nodes
# ------------------------------------------------------
#$ -cwd
# -q default.q
#
#
# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/DCASE/out/fea_extr.$JOB_ID.out
#$ -e /homedtic/rgong/DCASE/error/fea_extr.$JOB_ID.err
#
# Create an array job = !!!!!!number of audio in the target folder!!!!!!, create 49 tasks, every task corresponds to a parameter setting for SVM
# ----------------
#$ -t 1-4:1
#
#
# Start script
# --------------------------------


printf "Starting execution of job $JOB_ID from user $SGE_O_LOGNAME at `date`\n"

# force UTF 8
export LANG="en_US.utf8"

python /homedtic/rgong/DCASE2017/CNNs_classifier/featureExtraction.py ${SGE_TASK_ID}

# Print job Done
printf "Job $JOB_ID done at `date`\n"