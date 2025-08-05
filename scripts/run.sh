#!/bin/bash
#SBATCH --output /home/s/swein/gwdet/output/train_mha.out
#SBATCH --nodes=1
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=12 
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition gpuv100,gputitanrtx,gpua100,gpuhgx,gpu3090
#SBATCH --time=5:00:00
#SBATCH --job-name=train_mha
#SBATCH --mail-type=END,FAIL

source /scratch/tmp/swein/gwdet/gwdet_env/bin/activate
srun python /home/s/swein/gwdet/main.py fit -c configs/mha_convfirst.yaml