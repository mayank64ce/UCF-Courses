#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=PA3_Train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal

echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
source activate assignment
python train.py --mode "FC" --batch_size 10 --epochs 10 --learning_rate 0.001 --model_save_path model1.pth > logs/output1.txt
python train.py --mode "CNN" --batch_size 10 --epochs 10 --learning_rate 0.001 --model_save_path model2.pth > logs/output2.txt