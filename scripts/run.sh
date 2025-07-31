#!/bin/bash
#SBATCH --output /home/s/swein/gwdet/output/train.out
#SBATCH --nodes=1
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=12 
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition gputitanrtx,gpu3090,gpu2080,gpu4090
#SBATCH --time=5:00:00
#SBATCH --job-name=gwdet_train
#SBATCH --mail-type=END,FAIL

source /scratch/tmp/swein/gwdet/gwdet_env/bin/activate
srun python /home/s/swein/gwdet/main.py fit -c config.yaml