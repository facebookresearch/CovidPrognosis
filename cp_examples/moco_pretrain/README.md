# Momentum Contrast Pretraining

This directory contains example code for pretraining COVID prognosis prediction
models by applying the momentum contrast technique to public X-ray data.

The Momentum Contrast pretraining technique was described in the following
paper:

[Momentum Contrast for Unsupervised Visual Representation Learning (He et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)

## Usage

The workhorse script is `train_moco.py`. To get a list of options, you can type

```bash
python train_moco.py -h
```

By default, the script will train for 200 epochs on 2 GPUs with a batch size
of 128. Note that this corresponds to 256 examples per gradient update in `ddp`
mode. The paper used 8 GPUs, which corresponds to 1024 samples per gradient
update. If you don't have 8 GPUs to use for pretraining, your pretrained models
may behave differently.

The script doesn't include any validation loops, but it does track the
contrastive loss via Tensorboard and you'll see the accuracy of the contrastive
classifier.

After pretraining, you can pass your model to the scripts in `sip_finetune` or
`mip_finetune` depending on your downstream task.
