#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=PA4_Train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal

echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
source activate assignment
cd Coursework/UCF-Courses/Fall 2023/CAP 5415/Assignments/PA4/
python train.py --mode 1 --learning_rate 0.001 --batch_size 32 --num_epochs 30
python train.py --mode 2 --learning_rate 0.001 --batch_size 32 --num_epochs 30