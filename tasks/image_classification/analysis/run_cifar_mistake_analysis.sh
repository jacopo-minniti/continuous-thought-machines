#!/bin/bash
#SBATCH --job-name=ctm_cifar_mistakes
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/scratch/jacopo04/continuous-thought-machines/logs/ctm_cifar_mistakes-%j.out
#SBATCH --error=/scratch/jacopo04/continuous-thought-machines/logs/ctm_cifar_mistakes-%j.err
#SBATCH -D /scratch/jacopo04/continuous-thought-machines

set -euo pipefail

# --- Environment Setup ---
module --force purge
module load StdEnv/2023
module load python
module load gcc opencv arrow

source .venv/bin/activate

python -m tasks.image_classification.analysis.run_cifar_mistake_analysis \
  --checkpoint checkpoints/cifar10/ctm.pt \
  --dataset cifar10 \
  --data-root data/ \
  --output-dir tasks/image_classification/analysis/outputs/cifar_mistakes \
  --batch-size 256 \
  --device 0 \
  --inference-iterations 50 \
  --store-all
