import logging
import hydra

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="test_config.yaml", version_base="1.1")
def train(cfg):    
    batch_size = cfg.hyperparameters.batch_size
    learning_rate = cfg.hyperparameters.learning_rate
    lora_rank = cfg.hyperparameters.lora_rank
    train_epochs = cfg.hyperparameters.train_epochs
    cuda = cfg.hyperparameters.cuda
    seed = cfg.hyperparameters.seed
    store_weights_to = cfg.hyperparameters.store_weights_to

    logger.info("Training started.") # saved to ./outputs/date/time/hydra_usage_example.log

    logger.critical(f"LoRA weights are stored to: {store_weights_to}...")


if __name__ == "__main__":
    train()

