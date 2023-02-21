ID="1"
#sbatch -J nopose_noddim_$ID ./scripts/sbatch_train.sh --config configs/nopose_noddim.yaml --mode train --id $ID
#sbatch -J nopose_ddim_class_$ID ./scripts/sbatch_train.sh --config configs/nopose_ddim_class.yaml --mode train --id $ID
sbatch -J nopose_ddim_data_$ID ./scripts/sbatch_train.sh --config configs/nopose_ddim_data.yaml --mode train --id $ID
#sbatch -J nopose_ddim_arg_$ID ./scripts/sbatch_train.sh --config configs/nopose_ddim_arg.yaml --mode train --id $ID
#sbatch -J nopose_ddim_arg_005_$ID ./scripts/sbatch_train.sh --config configs/nopose_ddim_arg_005.yaml --mode train --id $ID 
#sbatch -J nopose_ddim_data_10_$ID ./scripts/sbatch_train.sh --config configs/nopose_ddim_data_10.yaml --mode train --id $ID 