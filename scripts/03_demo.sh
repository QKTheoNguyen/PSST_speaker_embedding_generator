#!/bin/bash
source ./gen/env.sh

cd ./gen

#download pretrain models
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

# specify the anon modes: _ohnn_pre_esd_msp_sad1, _flow, _gan

training=gan_3_epochs
dir_xvec_gen=s_0.7

model_type=libri_tts_clean_100_fbank_xv_ssl_freeze

echo "Generating anonymized speech with $anon_mode-based pseudo-speakers..."

for dset in IEMOCAP_dev IEMOCAP_test savee; do

    # Use $anon_mode-generated pseudo vectors

    # xv_dir="$xv_base_dir/${dset}_$anon_mode"
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