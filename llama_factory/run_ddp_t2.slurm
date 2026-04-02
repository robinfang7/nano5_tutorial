#!/bin/bash
#SBATCH -A <Project_ID>           # iService Project id
#SBATCH -J llama_8B_sft_rola8   # job name
#SBATCH -p gtest                 # partition gpu-amd gp1d gpNCHC_LLM
#SBATCH --nodes=2              # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=1    # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=4      # Number of cores per MPI task
#SBATCH --gres=gpu:2
#SBATCH -o %x_%j.out          # Path to the standard output file
##SBATCH -w hgpn41            # requests specific nodes

export OMP_NUM_THREADS=1 ; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(($RANDOM % 10000 + 20000)) # random port form 20000 to 29999

# Get GPU count
IFS=',' read -ra gpu_array <<< "$SLURM_JOB_GPUS"
GPUS_PER_NODE=${#gpu_array[@]}

echo -e "\n  ========== print yaml =========="
cat examples/train_lora/llama3_lora_sft.yaml
echo -e "  ===================================\n"
#cat examples/deepspeed/ds_z3_config.json
#echo -e "  ===================================\n"

SIF=/work/$(whoami)/sif/llamafactory_0.9.4.sif
export SINGULARITY="singularity exec -B /work:/work --nv $SIF"

export LAUNCHER="torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node $GPUS_PER_NODE \
--master-addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
"
export TRAIN="src/train.py examples/train_lora/llama3_lora_sft.yaml"
#src/train.py examples/train_lora/llama3_lora_sft.yaml #
#src/train.py examples/train_lora/llama3_lora_sft_ds3.yaml # deepspeed zero3

# clean output
rm -rf save/*

SECONDS=0
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES \
bash -c "
    CMD=\"\$SINGULARITY \$LAUNCHER --node_rank \$SLURM_NODEID \$TRAIN\"
    echo \" \$CMD \"
    exec \$CMD
    "

RUNTIME=$SECONDS
HOUR=$(($RUNTIME / 3600)) ;MIN=$(((RUNTIME /60)%60)) ;SEC=$((RUNTIME % 60))
echo -e "\n  Runtime:$HOUR:$MIN:$SEC\n"

