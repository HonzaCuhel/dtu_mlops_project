# dtu_mlops_project

A short description of the project.

<b>Authors:</b>
- Jan Cuhel
- Adam Jirkovsky
- Mikhail Poludin
- Antonis Pavlou

<b>Date:</b> 2024

<b>Deployed FE app:</b> [link](https://fe-financial-tweet-sentiment-o64hln5vbq-ew.a.run.app)

<b>Deployed API server:</b> [link](https://deployed-financial-tweet-sentiment-o64hln5vbq-ew.a.run.app)


## Overall goal of the project
The goal of the project is to use power of the Natural Language Processing to solve a classification task of predicting sentiment of finance-related tweets.

## What framework are you going to use and you do you intend to include the framework into your project?
We plan to use Hugging Face to obtain the dataset and the baseline model. We will leverage the [Transformers](https://github.com/huggingface/transformers) library to manipulate with the model. We will finetune the selected model for our specific task. We plan to use DVC for data versioning, Weights and Biases for experiment tracking, and Hydra to ensure reproducibility. The project will also use Docker.

## What data are you going to run on (initially, may change)?
We are using the [Twitter Financial News dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) available through [HuggingFace Datasets](https://huggingface.co/docs/datasets/index). The dataset is an english sentiment analysis dataset containing an annotated corpus of finance-related tweets. The dataset is divided into 2 splits: `train` and `validation`. `train` split contains `9 938` samples and `validation` contains `2 486` samples. Each sample contains a text and its corresponding label. The dataset was chosen because it is quite simple, interesting and straightforward which makes it a great dataset for the purposes of this project.

## What deep learning models do you expect to use? :brain:
We are going to use a pre-trained BERT-like model and fine-tune it on the above-mentioned financial dataset. For example, the model we have in mind is DeBERTaV3, which is available on Hugging Face [here](https://huggingface.co/microsoft/deberta-v3-xsmall).

> The DeBERTa V3 xsmall model comes with 12 layers and a hidden size of 384. It has only **22M** backbone parameters, with a vocabulary containing 128K tokens which introduces 48M parameters in the Embedding layer.

This DeBERTa model has significantly fewer parameters compared to the classical RoBERTa-base (86M) and XLNet-base (92M), yet it achieves equal or better results on a majority of NLU tasks, such as on SQuAD 2.0 (F1/EM) or MNLI-m/mm (ACC).

Since the DeBERTa model is available on Hugging Face, the inference and training processes should be straightforward, allowing us to spend more time on the MLOps aspects of the project.

## Results

To see the wandb logs, please refer [here](https://wandb.ai/dtu-mlops-financial-tweets/train/?workspace=user-).

## Commands

#### Process raw data into processed data
```shell
make data
```
#### Train model
During the training run you will be prompted your W&B API key which you can find in your profile settings on weights and biases website.
##### On CPU
```shell
make train_cpu_model
```
##### On GPU
```shell
make train_gpu_model
```
<!-- You can remove the `--gpu all` switch for gpu-less machines.

The `-v $(pwd)/models:/models/` makes the `models/` folder shared between the host and the container so that the learned weights were saved to the host. -->
#### Inference on CPU:
```shell
make infer_cpu_model
```
#### Inference on GPU:
```shell
make infer_gpu_model
```
#### Local Build of Inference API:
```shell
make build_api_local
```

Now, go to [here](http://0.0.0.0:8080/) to open the app.

The app offers 4 endpoints:
- [`/`](http://0.0.0.0:8080/) - Health check
- [`/docs`](http://0.0.0.0:8080/docs) - Documentation
- [`/metrics`](http://0.0.0.0:8080/docs) - Endpoint that serves Prometheus metrics.
- [`/predict_batch/`](http://0.0.0.0:8080/predict_batch/) - Endpoint that infers a trained model.

Example of how to call the `/predict_batch/` endpoint:
- Using `curl`
```shell
curl -X 'POST' \
  'http://0.0.0.0:8080/predict_batch/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '["I think $TSLA is going to the moon!"]'
```
- Using Python
```python
import requests

url = "http://0.0.0.0:8080/predict_batch/"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
data = [
    "I think $TSLA is going to the moon!"
]

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

#### Local Build of Front-End App for Inference:
```shell
make build_fe_local
```

Now, go to [here](http://0.0.0.0:8501) to open the app.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── api                  <- Source code for building an inference AP
│
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── dockerfiles          <- Folder with docker files
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── dtu_mlops_project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── configs          <- Config files for Hydra
│   │   ├── train_config.yaml   <- A `.yaml` config for training
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── evaluate_model.py       <- script for evaluating trained model on test dataset
│   ├── hydra_usage_example.py  <- script for showing how to work with hydra
│   ├── train_model.py          <- script for training the model
│   └── predict_model.py        <- script for predicting from a model
│
├── front_end            <- Source code for building a FE app
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
