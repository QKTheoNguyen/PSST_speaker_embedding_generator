#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ./gen/env.sh

cd ./gen

# ============================================
# Parameters
# ============================================

# anon modes in: flow, gan
anon_mode=gan
epochs=3

# HiFiGAN model
model_type=libri_tts_clean_100_fbank_xv_ssl_freeze

# Flow-matching parameters
speaker_weight=0
flow_steps=5

# GAN parameters
similarity_threshold=0.7


# ============================================
# Download pretrained models
# ============================================
if [ ! -e "pretrained_models_anon_xv/" ]; then
    if [ -f pretrained_models_anon_xv.tar.gz ];
    then
        rm pretrained_models_anon_xv.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    wget https://zenodo.org/record/6529898/files/pretrained_models_anon_xv.tar.gz
    tar -xzvf pretrained_models_anon_xv.tar.gz
    cd pretrained_models_anon_xv/
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi

# ============================================
# Training and generating speaker embeddings (Flow or GAN)
# ============================================

if [ "$anon_mode" == "flow" ]; then

    # ============================================
    # Flow - Step 1: Train flow-based pseudo-vector generator
    # ============================================

    python selec_anon/compute_anon_spk_vector/flow_matching/train_flow.py \
    --embeddings_dir \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_dev_enrolls \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_test_enrolls \
    --device cuda:0 \
    --config_path ./configs/training/flow_matching/training_config.json \
    --epochs ${epochs}

    training=flow_${epochs}_epochs

    # ============================================
    # Flow - Step 2: Generate flow-based pseudo-vector
    # ============================================

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

    dir_xvec_gen=w_${speaker_weight}_step_${flow_steps}


elif [ "$anon_mode" == "gan" ]; then

    # ============================================
    # GAN - Step 1: Train gan-based pseudo-vector generator
    # ============================================
    python selec_anon/compute_anon_spk_vector/WGAN/train_gan.py \
    --embeddings_dir \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_dev_enrolls \
        ../compute_ori_spk_vector/output_ori_spk_vector/libri_test_enrolls \
    --device cuda:0 \
    --config_path ./configs/training/WGAN/training_config.json

    training=gan_${epochs}_epochs

    # ============================================
    # GAN - Step 2: Generate gan-based pseudo-vector
    # ============================================
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

    dir_xvec_gen=s_${similarity_threshold}

else
    echo "Error: Invalid anon_mode. Choose 'flow' or 'gan'."
    exit 1
fi

# ============================================
# Flow/GAN - Step 3: Generate anonymized audio
# ============================================
for dset in IEMOCAP_dev IEMOCAP_test savee; do

    xv_dir=../results_embeddings
    xv_dir="$xv_dir/$training/$dset/$dir_xvec_gen"
    
    if [ ! -f "$xv_dir/pseudo_xvector.scp" ]; then
        echo "Error: $training vectors not found at $xv_dir"
        exit 1
    fi
    
    python adapted_from_facebookresearch/inference.py \
        --input_test_file ../data/$dset/wav.scp \
        --xv_dir "$xv_dir" \
        --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/$model_type \
        --output_dir ../results_audio/${training}/${dset}/${dir_xvec_gen} \
        --output_original_dir ../results_audio/${dset} \
        --write_original
done

echo "✓ Anonymization complete"
