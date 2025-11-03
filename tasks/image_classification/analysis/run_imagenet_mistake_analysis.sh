#!/bin/bash
#SBATCH --job-name=ctm_imagenet_mistakes
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
export HF_HOME="/scratch/jacopo04/continuous-thought-machines/.cache"

python -m tasks.image_classification.analysis.run_imagenet_mistake_analysis \
  --checkpoint checkpoints/imagenet/ctm.pt \
  --split validation \
  --output-dir "tasks/image_classification/analysis/outputs/imagenet/t=50_temp=dynamic_min=1.5" \
  --batch-size 128 \
  --device 0 \
  --inference-iterations 50 \
  --dwell-threshold 0.05 \
  --dwell-steps 3


# run --time=00:05:00 --gpus-per-node=h100:1     .venv/bin/python -m tasks.image_classification.analysis.run_imagenet_analysis     --debug     --actions videos     --data_indices 0 1 2 --device 0
