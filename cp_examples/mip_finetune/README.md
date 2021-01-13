# Multiple-image prediction (MIP) Finetuning

This directory includes scripts for fine-tuning a multiple-image prediction (MIP)
model. The MIP model can make predictions from sequences of images.

We do not release any COVID data with the paper. Therefore, this directory only
contains an implementation of the MIP model and a trainer. Users are expected to
implement their own data loader for their image sequence data.

## Data Loader

Create your own data loader similar to `covidprognosis.plmodules.XrayDataModule`.
The data loader should return batches containing the following fields:

* `images`: sequence of X-ray images
* `times`: relative scan times
* `lens`: sequence lengths
* `labels`: labels

The DropImage regularizer can be implemented as part of the data loader by
randomly dropping some images.

Once this is implemented, replace the line:

```python
data_module = None
```

in `train_mip.py` with your data loader.

## Usage

Prior to fine-tuning, you need to pretrain a model using the scripts in
`moco_pretrain` or download one of the publicly-available models.

The workhorse script is `train_mip.py`. To get a list of options, you can type

```bash
python train_mip.py -h
```

By default, the script will train for 50 epochs on 1 GPU with a batch size
of 32.
