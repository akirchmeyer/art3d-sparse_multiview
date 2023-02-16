sbatch -J nopose_ddim_class ./scripts/sbatch_train.sh --config configs/nopose_ddim_class.yaml --mode train    
sbatch -J nopose_ddim_data ./scripts/sbatch_train.sh --config configs/nopose_ddim_data.yaml --mode train    
sbatch -J nopose_ddim_arg ./scripts/sbatch_train.sh --config configs/nopose_ddim_arg.yaml --mode train    
sbatch -J nopose_ddim_arg_005 ./scripts/sbatch_train.sh --config configs/nopose_ddim_arg_005.yaml --mode train    