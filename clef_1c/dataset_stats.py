from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from datasets import load_dataset, Features, Value
import pandas as pd
import json
from utils import get_dataset_length_stats
# from utils import pd

# ---- Tokenizer loading
tokenizer_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# ---- Utils
def get_dataset_ratios(dataset, label="label"):
    result = {}
    for split in dataset:
        df = dataset[split].to_pandas()
        result[split] = df[label].value_counts(normalize=True).to_json()
    return result


# ---- dataset loading/prep
features = Features(
    {
        "tweet_text": Value("string"),
        "class_label": Value("int64"),
    }
)

data_files = {
    "train": "./data/CT22_english_1C_harmful_train.tsv",
    "test": "./data/CT22_english_1C_harmful_test_gold.tsv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", features=features)
dataset = dataset.rename_column("class_label", "label")
dataset = dataset.rename_column("tweet_text", "text")
print(dataset)

# ---- Stats
print("-" * 20, "Stats Overall", "-" * 20)
# ratios = get_dataset_ratios(dataset, "label")
# print(json.dumps(ratios, indent=4))

# length_stats = get_dataset_length_stats(tokenizer, dataset)
# print(json.dumps(length_stats, indent=4))


print("-" * 20, "Stats Per Split", "-" * 20)
ratios = get_dataset_ratios(dataset, "label")
print(json.dumps(ratios, indent=4))

length_stats = get_dataset_length_stats(tokenizer, dataset)
print(json.dumps(length_stats, indent=4))
