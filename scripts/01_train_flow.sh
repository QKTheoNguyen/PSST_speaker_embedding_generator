#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

### Use the following command for monitoring,
# srun --gpus=1 --mem=32G --time=24:00:00 selec_anon/compute_anon_spk_vector/01_train_flow.sh

source ./gen/env.sh

cd ./gen

python selec_anon/compute_anon_spk_vector/flow_matching/train_flow.py \
    --embeddings_dir \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_dev_enrolls \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_test_enrolls \
    --device cuda:0 \
    --config_path ./configs/training/flow_matching/training_config.json

echo "✓ Training flow-based pseudo-vector generator completed"
