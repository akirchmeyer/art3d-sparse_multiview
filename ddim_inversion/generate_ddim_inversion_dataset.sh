#! /bin/bash
#SBATCH --output=/home/akirchme/logs/%x.out
#SBATCH --error=/home/akirchme/logs/%x.err
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=4:00:00

cd /home/akirchme/art3d/art3d-multiviewdepthdiffusion/ddim_inversion
python script_generate_ddim_inversion_dataset.py --cls ${CLASS:=dog} --step 8 --start ${START:=0} 