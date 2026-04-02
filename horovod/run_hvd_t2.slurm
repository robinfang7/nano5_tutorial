#!/bin/bash
#SBATCH -A <Project ID>          # iService Project id
#SBATCH -J hvd                # job name
#SBATCH -p gtest              # partition dev, normal normal2
#SBATCH --nodes=1             # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4     # Number of cores per MPI task
#SBATCH --ntasks-per-node=2   # Number of MPI tasks (i.e. processes)
#SBATCH --gres=gpu:2
#SBATCH -o %x_%j.out          # Path to the standard output file

export TF_ENABLE_ONEDNN_OPTS=0 # disable oneDNN message
export TF_CPP_MIN_LOG_LEVEL=2 # disable tensorflow info & warm message
export PMIX_MCA_gds=hash

SECONDS=0

SIF="/work/$(whoami)/sif/tensorflow_24.05-tf2-py3.sif"
SINGULARITY="singularity exec -B /work:/work --nv $SIF"

CMD="python tf2_keras_mnist.py" # test_horovod.py  pytorch_mnist.py
#tf2_keras_mnist.py
#tf2_keras_synthetic_benchmark.py

RUN="srun --mpi=pmi2 --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS $SINGULARITY $CMD"
echo $RUN; $RUN

RUNTIME=$SECONDS
HOUR=$(($RUNTIME / 3600)) ;MIN=$(((RUNTIME /60)%60)) ;SEC=$((RUNTIME % 60))
echo -e "\nRuntime:$HOUR:$MIN:$SEC\n"
