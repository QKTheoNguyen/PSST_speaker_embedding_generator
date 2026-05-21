#!/bin/bash
# Generate pseudo-speaker xvectors using GAN-based anonymization
source ./gen/env.sh

cd ./gen

training=gan_3_epochs
similarity_threshold=0.7

# Generate utterance-level pseudo vectors for ALL datasets
for dset in IEMOCAP_dev IEMOCAP_test savee; do
    echo "Generating GAN-based pseudo vectors for $dset..."
    python selec_anon/compute_anon_spk_vector/gen_pseudo_gan_utterance_level.py \
        --data_dir ../../data/$dset \
        --vector_dir ../compute_ori_spk_vector/output_ori_spk_vector/$dset \
        --gan_model_dir ./trained/gan_models/${training} \
        --output_dir ../results_embeddings/${training}/${dset}/s_${similarity_threshold} \
        --similarity_threshold ${similarity_threshold} \
        --device cuda:0
done

echo "✓ GAN-based pseudo-vector generated"
