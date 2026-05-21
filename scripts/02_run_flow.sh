#!/bin/bash
# Generate pseudo-speaker xvectors using flow-based anonymization
source ./gen/env.sh

cd ./gen

### Parameters ###
training=ecapa_flow_5000_epochs
speaker_weight=0
flow_steps=50

# Generate utterance-level pseudo vectors using flow matching
for dset in IEMOCAP_dev IEMOCAP_test savee; do
    echo "Generating flow-based pseudo vectors for $dset..."
    python selec_anon/compute_anon_spk_vector/gen_pseudo_flow_utterance_level.py \
        --data_dir ../data/$dset \
        --vector_dir ../compute_ori_spk_vector/output_ori_spk_vector/$dset \
        --flow_model_dir ./trained/flow_models/${training} \
        --output_dir ../results_embeddings/${training}/${dset}/w_${speaker_weight}_step_${flow_steps} \
        --speaker_weight ${speaker_weight} \
        --flow_steps ${flow_steps} \
        --device cuda:0
done

echo "✓ Flow-based pseudo-vector generated"
