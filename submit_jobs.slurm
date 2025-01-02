#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/zc1592/small_data/log/%A_%a.out
#SBATCH --error=/scratch/zc1592/small_data/log/%A_%a.out
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --array=0-1



singularity exec --bind /scratch --nv --overlay  /scratch/zc1592/overlay-50G-10M.ext3:ro /scratch/zc1592/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate sd
export SSL_CERT_DIR=/etc/ssl/certs/
cd /scratch/zc1592/small_data
bash train_jobs.sh $SLURM_ARRAY_TASK_ID
"