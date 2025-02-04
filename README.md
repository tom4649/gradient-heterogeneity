# Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers

This repository contains the code for our paper:

> Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers. Akiyoshi Tomihari and Issei Sato.
[arXiv](https://arxiv.org/abs/2502.00213)
<!-- [OpenReview]() -->

## Dependencies
The main dependencies are:
```plaintext
Python 3.10 or higher
torch = 2.4.0
```

Please refer to the `pyproject.toml` file for more details.

## Setup
To set up and run the project, follow these steps:
```bash
# Configure the project to create virtual environments within the project directory
poetry config virtualenvs.in-project true

# Set the local python version using pyenv
pyenv local 3.12.6

# Install dependencies and activate the virtual environment
poetry install
poetry shell
```
## Training the Models

To train the models, run the following script:

```bash
bash shell_scripts/train_<task>.sh <dataset_name> <optimizer_name> <model_name> [<lr_scheduler_type>]
```
- `<task>`: Specify `nlp` or `vision`.
- `<dataset_name>`: Name of the dataset (e.g., `rte`, `flowers102`).
- `<optimizer_name>`: Name of the optimizer (e.g., `adam`, `sgd_momentum`).
- `<model_name>`: Name of the model to be trained (e.g., `roberta-base`, `resnet18`).
- `<lr_scheduler_type>` (optional): Learning rate scheduler type, applicable only for NLP tasks. Defaults to `default` (meaning linear) if not provided.

### Example
```bash
bash shell_scripts/train_nlp.sh rte adam roberta-base
```

## Calculation of Hessian per Parameter

To calculate the maximum Hessian values for each parameter, run the following script:

```bash
bash shell_scripts/hessian_per_param.sh <dataset_name> <optimizer_name> <model_name> <domain> [<training_mode>]
```

- `<domain>`: Specify `nlp` or `vision`.
- `<training_mode>` (optional): Specify `pretrained` to use a pre-trained model. If omitted, a trained model will be used.

When using a trained model, you need to specify the directory in `"results/hessian_per_param/model_dir_dict.json"`.

### Example
```bash
bash shell_scripts/hessian_per_param.sh rte adam roberta-base pretrained
```

## Acknowledgments
We use the following resources and libraries:
- Base code structure: [lp-ft_ntk](https://github.com/tom4649/lp-ft_ntk)

- Libraries for NLP tasks: [Hugging Face Transformers](https://github.com/huggingface/transformers)

- Calculation of Hessian: [PyHessian](https://github.com/amirgholami/PyHessian)

- Signum optimizer: [Signum](https://github.com/jiaweizzhao/Signum_pytorch/blob/master/signum.py)

## Citation
```bibtex
@misc{tomihari2025understandingadamoutperformssgd,
      title={Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers},
      author={Akiyoshi Tomihari and Issei Sato},
      year={2025},
      eprint={2502.00213},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.00213},
}
```
