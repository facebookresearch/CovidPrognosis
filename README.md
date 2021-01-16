# CovidPrognosis

This repository contains code for reproducing the following paper:

[A. Sriram*, M. Muckley*, K. Sinha, F. Shamout, J. Pineau, K. J. Geras, L. Azour, Y. Aphinyanaphongs, N. Yakubova, W. Moore. COVID-19 Deterioration Prediction via Self-Supervised Representation Learning and Multi-Image Prediction. *arXiv preprint arXiv:2101.04909* (2020).](https://arxiv.org/abs/2101.04909)

We also include models from the MoCo pretraining process for groups interested
in fine-tuning them on their own data. Prior to using this code or pretrained
models please consult the [Disclaimer](#Disclaimer).

## Installation

First, follow the
[official instructions](https://pytorch.org/get-started/locally/) for
installing PyTorch. Then, navigate to the root `CovidPrognosis` directory and
run

```bash
pip install -e .
```

After that you should be able to run the examples in `cp_examples`.

## Usage

For pretraining, you'll need to download the
[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/2.0.0/) or
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) datasets.
Once you've downloaded the data, add the path to `configs/data.yaml` and it
should be used as a default.

The `cp_examples` directory contains three subdirectories corresponding to the
training stages in the paper:

- `moco_pretrain`: Momentum-contrast (MoCo) pretraining (e.g., with MIMIC,
CheXpert, or both)
- `sip_finetune`: Fine-tuning of MoCo models for single-image prediction tasks
(i.e., single-image adverse event prediction or oxygen requirements prediction)
- `mip_finetune`: Fine-tuning of MoCo models for multi-image prediction tasks

Our code is built on top of the
[PyTorch Lightning](https://www.pytorchlightning.ai/) framework.

The examples scripts for MoCo pretraining and SIP fine-tuning are set up for
public X-ray data sets - due to consideration of patient privacy we do not
release COVID data and use the public data as examples.

## Pretrained Models

We provide pretrained models that use the
[MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/2.0.0/) and
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
datasets. These datasets are subject to additional terms and conditions as
indicated on their respective websites. For a list of models, see
[here](configs/models.yaml). For an example of how to download and train with
the models, please look at the
[SIP Fine-tuning example](cp_examples/sip_finetune).

The following publication describes MIMIC-CXR:

[MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports (A.E.W. Johnson et al., 2019)](https://www.nature.com/articles/s41597-019-0322-0)

And this publication describes CheXpert:

[CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison (J. Irvin et al., 2019)](https://ojs.aaai.org//index.php/AAAI/article/view/3834)

## Disclaimer

This code and accompanying pretrained models are provided with no guarantees
regarding their reliability, accuracy or suitability for any particular
application and should be used for research purposes only. The models and code
are not to be used for public health decisions or responses, or for any
clinical application or as a substitute for medical advice or guidance.

## Citation

If you use this code or models in your scientific work, please cite the
following paper:

```bibtex
@misc{sriram2021covid19,
      title={COVID-19 Deterioration Prediction via Self-Supervised Representation Learning and Multi-Image Prediction}, 
      author={Anuroop Sriram and Matthew Muckley and Koustuv Sinha and Farah Shamout and Joelle Pineau and Krzysztof J. Geras and Lea Azour and Yindalon Aphinyanaphongs and Nafissa Yakubova and William Moore},
      year={2021},
      eprint={2101.04909},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

The code is MIT licensed, as found in the [LICENSE file](LICENSE).
