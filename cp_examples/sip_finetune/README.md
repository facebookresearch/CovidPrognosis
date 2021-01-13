# Single-image prediction (SIP) Finetuning

This directory includes scripts for fine-tuning a single-image prediction
model. The fine-tuning procedure is based on the simplified fine-tuning case in
the following paper:

[Momentum Contrast for Unsupervised Visual Representation Learning (He et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)

Specifically, the fine-tuning process freezes the entire model, including batch
norm statistics, and only trains the final fully-connected layer. This was
reported as the "MoCo PT CL" ablation in the COVID deterioration prediction
paper.

No COVID data is released with the paper. To allow researchers to reproduce
results on their own COVID data sets, the scripts in this directory show
fine-tuning examples for well-studied public X-ray data sets. Using MoCo
pretraining will typically yield average AUC values of 0.83-0.9 on
[CheXpert competition tasks](https://stanfordmlgroup.github.io/competitions/chexpert/).

## Usage

Prior to fine-tuning, you need to pretrain a model using the scripts in
`moco_pretrain` or download one of the publicly-available models.

The workhorse script is `train_sip.py`. To get a list of options, you can type

```bash
python train_sip.py -h
```

By default, the script will train for 10 epochs on 1 GPU with a batch size
of 32. Altering the batch size and learning rates can impact results.

If you want to train using one of the open-sourced pretrained models, simply
pass it into the script:

```bash
python train_sip.py --pretrained_file https://dl.fbaipublicfiles.com/CovidPrognosis/pretrained_models/mimic_lr_0.1_bs_128_fd_128_qs_65536.pt
```

A list of pretrained models is in the `configs/models.yaml` configuration file.

The script includes validation loops for plotting accuracies and AUC values to
Tensorboard.
