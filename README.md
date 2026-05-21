## PSST speaker embedding generator

This repository is based on an implementation of the following paper as a baseline- [Adapting General Disentanglement-Based Speaker Anonymization for Enhanced Emotion Preservation](https://arxiv.org/abs/2408.05928)

In this work we explore speech anonymization system's speaker embedding anonymization. We propose 2 (for now) speaker generator models, based on flow-matching and GANs.

## How to run

`git clone https://github.com/QKTheoNguyen/PSST_speaker_embedding_generator.git`

`cd emotion-compensation/gen`

`bash scripts/install.sh`

fairseq is deprecated so you have to install it elsewhere and modify manually the `gen/env.sh` script with your fairseq path

### Flow-based speaker embedding generator

- Train flow-matching speaker embedding generator

`bash script/01_train_flow.sh`

- Generate flow-based pseudo-speaker embedding

`bash script/02_run_flow.sh`

- Generate anonymized speech using edited pseudo-speaker embeddings, 

`bash script/03_demo.sh`

### GAN-based speaker embedding generator

- Train flow-matching speaker embedding generator

`bash script/01_train_gan.sh`

- Generate flow-based pseudo-speaker embedding

`bash script/02_run_gan.sh`

- Generate anonymized speech using edited pseudo-speaker embeddings, 

`bash script/03_demo.sh`

### Run all

You may also run all with

`bash script/train_run_gen.sh`




## Acknowledgments
This work was funded by the European Union’s Horizon Europe research and innovation programme grant No 101168193.

## License
The whole project follows the Attribution-NonCommercial 4.0 International License




