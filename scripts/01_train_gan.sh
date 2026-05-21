#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ./gen/env.sh

cd ./gen

python selec_anon/compute_anon_spk_vector/WGAN/train_gan.py \
    --embeddings_dir \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_dev_enrolls \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_test_enrolls \
    --device cuda:0 \
    --config_path ./configs/training/WGAN/training_config.json

echo "✓ Training gan-based pseudo-vector generator completed"
