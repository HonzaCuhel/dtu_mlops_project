import os

from datasets import load_dataset, load_from_disk


def main():
    """Download and preprocess data."""
    if os.path.exists("data/raw/train") and os.path.exists("data/raw/val"):
        train_set_raw = load_from_disk(os.path.join("data", "raw", "train"))
        val_set = load_from_disk(os.path.join("data", "raw", "val"))
    else:
        train_set_raw = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        train_set_raw.save_to_disk(os.path.join("data", "raw", "train"))
        val_set = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
        val_set.save_to_disk(os.path.join("data", "raw", "val"))

    split_dict = train_set_raw.train_test_split(test_size=2500, seed=42)
    split_dict["train"].save_to_disk(os.path.join("data", "processed", "train"))
    split_dict["test"].save_to_disk(os.path.join("data", "processed", "test"))
    val_set.save_to_disk(os.path.join("data", "processed", "val"))


if __name__ == "__main__":
    main()
