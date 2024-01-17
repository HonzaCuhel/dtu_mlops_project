import logging
import sys

import hydra

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train_config.yaml", version_base="1.1")
def train(cfg):
    cuda = cfg.hyperparameters.cuda
    seed = cfg.hyperparameters.seed
    lora_rank = cfg.hyperparameters.lora_rank
    batch_size = cfg.hyperparameters.batch_size
    train_epochs = cfg.hyperparameters.train_epochs
    learning_rate = cfg.hyperparameters.learning_rate
    lr_warmup_steps = cfg.hyperparameters.lr_warmup_steps
    store_weights_to = cfg.hyperparameters.store_weights_to
    train_set_path = cfg.dataset.train_path
    test_set_path = cfg.dataset.test_path

    # Training or inference:
    # base_model = AutoModelForCausalLM.from_pretrained(...)

    logger.info("Training started.") # saved to ./outputs/date/time/hydra_usage_example.log
    logger.info(f"Model weights are stored to: {store_weights_to}...")


if __name__ == "__main__":
    train()
