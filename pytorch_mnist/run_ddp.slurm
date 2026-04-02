#!/bin/bash
#SBATCH -A <Project_ID>        # iService Project id
#SBATCH -J ddp                 # job name
#SBATCH -p gtest               # partition gtest gp1d, gpNCHC_LLM
#SBATCH --nodes=2              # Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=4      # Number of cores per MPI task
#SBATCH --ntasks-per-node=1    # Number of MPI tasks (i.e. processes)
#SBATCH --gres=gpu:1
#SBATCH -o %x_%j.out           # Path to the standard output file

export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(($RANDOM % 10000 + 20000)) # random port form 20000 to 29999 

# Get GPU count on T2
IFS=',' read -ra gpu_array <<< "$SLURM_JOB_GPUS"
GPUS_PER_NODE=${#gpu_array[@]}

[ -f "snapshot.pt" ] && rm "snapshot.pt" # remove snapshot.pt

SIF=/work/$(whoami)/sif/pytorch_24.05-py3.sif
export SINGULARITY="singularity exec --nv -B /work:/work $SIF"

export LAUNCHER="torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node $GPUS_PER_NODE \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
"
export TRAIN="multinode.py 10 10" # multigpu_torchrun.py multinode.py

SECONDS=0

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES \
bash -c "
    CMD=\"\$SINGULARITY \$LAUNCHER --node_rank \$SLURM_NODEID \$TRAIN\"
    echo \" \$CMD \"
    exec \$CMD
    "

RUNTIME=$SECONDS
HOUR=$(($RUNTIME / 3600)) ;MIN=$(((RUNTIME /60)%60)) ;SEC=$((RUNTIME % 60))
echo -e "\nRuntime:$HOUR:$MIN:$SEC\n"

