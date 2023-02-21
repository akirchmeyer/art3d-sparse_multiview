#! /bin/bash
#SBATCH --output=/home/akirchme/logs/%x.out
#SBATCH --error=/home/akirchme/logs/%x.err
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=6:00:00

cd /home/akirchme/art3d/art3d-multiviewdepthdiffusion/sparse_multiview
accelerate launch --config_file configs/accelerator.yaml train.py $@
