# dtu_mlops_project

A short description of the project.

<b>Authors:</b>
- Jan Cuhel
- Adam Jirkovsky
- Mikhail Poludin

## Overall goal of the project
The goal of the project is to use power of the Natural Language Processing to solve a classification task of predicting sentiment of finance-related tweets.

## What framework are you going to use and you do you intend to include the framework into your project?
We plan to use Hugging Face to obtain the dataset and the baseline model. We will leverage the [Transformers](https://github.com/huggingface/transformers) library to manipulate with the model. [LoRA](https://arxiv.org/pdf/2106.09685.pdf) will be used to finetune the selected model for our specific task. We plan to use DVC for data versioning, Weights and Biases for experiment tracking, and Hydra to ensure reproducibility. The project will also use Docker.

## What data are you going to run on (initially, may change)?
We are using the [Twitter Financial News dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) available through [HuggingFace Datasets](https://huggingface.co/docs/datasets/index). The dataset is an english sentiment analysis dataset containing an annotated corpus of finance-related tweets. The dataset is divided into 2 splits: `train` and `validation`. `train` split contains `9 938` samples and `validation` contains `2 486` samples. Each sample contains a text and its corresponding label. The dataset was chosen because it is quite simple, interesting and straightforward which makes it a great dataset for the purposes of this project.

## What deep learning models do you expect to use? :brain:
We are going to use a pre-trained BERT-like model and fine-tune it with the LoRA technique on the above-mentioned financial dataset. For example, the model we have in mind is DeBERTaV3, which is available on Hugging Face [here](https://huggingface.co/microsoft/deberta-v3-xsmall).

> The DeBERTa V3 xsmall model comes with 12 layers and a hidden size of 384. It has only **22M** backbone parameters, with a vocabulary containing 128K tokens which introduces 48M parameters in the Embedding layer.

This DeBERTa model has significantly fewer parameters compared to the classical RoBERTa-base (86M) and XLNet-base (92M), yet it achieves equal or better results on a majority of NLU tasks, such as on SQuAD 2.0 (F1/EM) or MNLI-m/mm (ACC).

Since the DeBERTa model is available on Hugging Face, the inference and training processes should be straightforward, allowing us to spend more time on the MLOps aspects of the project.

## Run training and inference inside a Docker container
### Training:
```shell
you@your-pc:~.../dtu_mlops_project$ docker build -t trainer_image -f dockerfiles/train_model.dockerfile .
```
```shell
you@your-pc:~.../dtu_mlops_project$ docker run -it --gpus all --name trainer_container -v $(pwd)/models:/models/ trainer_image
```
### Inference:
```shell
you@your-pc:~.../dtu_mlops_project$ docker build -t predictor_image -f dockerfiles/predict_model.dockerfile .
```
```shell
you@your-pc:~.../dtu_mlops_project$ docker run -it --gpus all --name predictor_container predictor_image
```

During the training run you will be prompted your W&B API key which you can find in your profile settings on weights and biases website.

You can remove the `--gpu all` switch for gpu-less machines.

The `-v $(pwd)/models:/models/` makes the `models/` folder shared between the host and the container so that the learned weights were saved to the host.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── dtu_mlops_project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
