import logging
import sys

import hydra

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train_config.yaml", version_base="1.1")
def train(cfg):
    """Template example of using Hydra for training."""
    seed = cfg.hyperparameters.seed
    batch_size = cfg.hyperparameters.batch_size
    learning_rate = cfg.hyperparameters.lr
    store_weights_to = cfg.hyperparameters.output_dir
    train_set_path = cfg.dataset.train_set_path
    val_set_path = cfg.dataset.val_set_path

    # Training or inference:
    # base_model = AutoModelForCausalLM.from_pretrained(...)

    logger.info("Training started.")  # saved to ./outputs/date/time/hydra_usage_example.log
    logger.info(f"Model weights are stored to: {store_weights_to}...")


if __name__ == "__main__":
    train()
