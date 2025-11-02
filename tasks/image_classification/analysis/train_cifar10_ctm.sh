#!/bin/bash
#SBATCH --job-name=ctm_cifar_train
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/jacopo04/continuous-thought-machines/logs/ctm_cifar_train-%j.out
#SBATCH --error=/scratch/jacopo04/continuous-thought-machines/logs/ctm_cifar_train-%j.err
#SBATCH -D /scratch/jacopo04/continuous-thought-machines

set -euo pipefail

# --- Environment Setup ---
module --force purge
module load StdEnv/2023
module load python
module load gcc opencv arrow

source .venv/bin/activate

mkdir -p checkpoints/cifar10
mkdir -p logs/cifar10_ctm_seed1
mkdir -p /scratch/jacopo04/continuous-thought-machines/logs

LOG_DIR="logs/cifar10_ctm_seed1"
CHECKPOINT_OUT="checkpoints/cifar10/ctm_cifar10_seed1.pt"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  -m tasks.image_classification.train_distributed \
    --log_dir "${LOG_DIR}" \
    --model ctm \
    --dataset cifar10 \
    --d_model 256 \
    --d_input 64 \
    --synapse_depth 5 \
    --heads 16 \
    --n_synch_out 256 \
    --n_synch_action 512 \
    --n_random_pairing_self 0 \
    --neuron_select_type random-pairing \
    --iterations 50 \
    --memory_length 15 \
    --deep_memory \
    --memory_hidden_dims 64 \
    --dropout 0.0 \
    --dropout_nlm 0 \
    --no-do_normalisation \
    --positional_embedding_type none \
    --backbone_type resnet18-1 \
    --training_iterations 600001 \
    --warmup_steps 1000 \
    --use_scheduler \
    --scheduler_type cosine \
    --weight_decay 0.0001 \
    --save_every 5000 \
    --track_every 5000 \
    --n_test_batches 50 \
    --num_workers_train 4 \
    --batch_size 256 \
    --batch_size_test 256 \
    --lr 1e-4 \
    --seed 1 \
    --use_amp

if [ -f "${LOG_DIR}/checkpoint.pt" ]; then
  cp "${LOG_DIR}/checkpoint.pt" "${CHECKPOINT_OUT}"
fi
