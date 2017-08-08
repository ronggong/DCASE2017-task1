#!/bin/bash

# change python version
module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

#$ -N vgg_j_eval
#$ -q default.q
#$ -l h=node01

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/DCASE/out/vgg_sequence_shuffle.$JOB_ID.out
#$ -e /homedtic/rgong/DCASE/error/vgg_sequence_shuffle.$JOB_ID.err

python /homedtic/rgong/DCASE2017/CNNs_classifier/runProcess_sequence_shuffle.py

printf "Job done. Ending at `date`\n"
