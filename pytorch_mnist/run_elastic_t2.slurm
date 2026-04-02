#!/bin/bash
#SBATCH -A <Project ID>          # iService Project id
#SBATCH -J ddp                # job name
#SBATCH -p gtest             # partition dev normal normal2
#SBATCH --nodes=1             # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4     # Number of cores per MPI task
#SBATCH --ntasks-per-node=1   # Number of MPI tasks (i.e. processes)
#SBATCH --gres=gpu:2          # GPUS per node
#SBATCH -o %x_%j.out          # Path to the standard output file

SECONDS=0
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(($RANDOM % 10000 + 20000)) # random port form 20000 to 29999
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# Get GPU count on T2
IFS=',' read -ra gpu_array <<< "$SLURM_JOB_GPUS"
GPUS_PER_NODE=${#gpu_array[@]}

[ -f "snapshot.pt" ] && rm "snapshot.pt" # remove snapshot.pt

SIF=/work/$(whoami)/sif/pytorch_24.05-py3.sif
SINGULARITY="singularity exec -B /work:/work --nv $SIF"

CMD="torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node=$GPUS_PER_NODE \
--rdzv_id $SLURM_JOBID \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
multigpu_torchrun.py 10 10"
#multigpu_torchrun.py
#multinode.py

if [[ "$SLURM_NNODES" = 1 ]]; then
    RUN="$SINGULARITY $CMD" # for one node
else
    RUN="srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES $SINGULARITY $CMD" # for multi nodes
fi

echo "$RUN"; $RUN

RUNTIME=$SECONDS
HOUR=$(($RUNTIME / 3600)) ;MIN=$(((RUNTIME /60)%60)) ;SEC=$((RUNTIME % 60))
echo -e "\n  Done Runtime:$HOUR:$MIN:$SEC\n"

