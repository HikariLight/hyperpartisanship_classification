from transformers import AutoTokenizer, set_seed
from datasets import DatasetDict, load_dataset, concatenate_datasets
import json
from utils import get_dataset_length_stats

set_seed(42)

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
data_files = {
    "train": "./data/HP_news_title_train.csv",
    "test": "./data/HP_news_title_test.csv",
}

dataset = load_dataset("csv", data_files=data_files)
dataset = dataset.rename_column("title", "text")

labels = ["neutral", "hyperpartisan"]
def map_label(element):
    element["label"] = labels[element["label"]]
    return element


dataset = dataset.map(map_label)

print(dataset)

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# ---- Stats Overall
print("-" * 20, "Stats Overall", "-" * 20)
combined_dataset = DatasetDict(
    {"train": concatenate_datasets([dataset["train"], dataset["test"]])}
)
print(combined_dataset)

print("-" * 10, "Ratios", "-" * 10)
ratios = get_dataset_ratios(combined_dataset, "label")
print(json.dumps(ratios, indent=4))
print("-" * 10, "Lengths", "-" * 10)
length_stats = get_dataset_length_stats(tokenizer, combined_dataset)
print(json.dumps(length_stats, indent=4))

# ---- Stats per split
print("-" * 20, "Stats Per Split", "-" * 20)
print("-" * 10, "Ratios", "-" * 10)
ratios = get_dataset_ratios(dataset, "label")
print(json.dumps(ratios, indent=4))

print("-" * 10, "Lengths", "-" * 10)
length_stats = get_dataset_length_stats(tokenizer, dataset)
print(json.dumps(length_stats, indent=4))
