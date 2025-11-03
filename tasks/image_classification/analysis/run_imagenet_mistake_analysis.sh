#!/bin/bash
#SBATCH --job-name=ctm_imagenet_mistakes
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/jacopo04/continuous-thought-machines/logs/ctm_imagenet_mistakes-%j.out
#SBATCH --error=/scratch/jacopo04/continuous-thought-machines/logs/ctm_imagenet_mistakes-%j.err
#SBATCH -D /scratch/jacopo04/continuous-thought-machines

set -euo pipefail

# --- Environment Setup ---
module --force purge
module load StdEnv/2023
module load python
module load gcc opencv arrow

source .venv/bin/activate

python -m tasks.image_classification.analysis.run_imagenet_mistake_analysis \
  --checkpoint checkpoints/imagenet/ctm_clean.pt \
  --split validation \
  --output-dir tasks/image_classification/analysis/outputs/imagenet_mistakes \
  --batch-size 128 \
  --device 0 \
  --inference-iterations 50
