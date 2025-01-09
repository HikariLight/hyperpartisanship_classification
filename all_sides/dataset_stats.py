from transformers import AutoTokenizer, set_seed
from datasets import load_dataset
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
labels = ["left", "center", "right"]
dataset_path = "./data/allsides_balanced_news_headlines-texts.csv"
dataset = load_dataset("csv", data_files=dataset_path)


def concatenate_header_text(example):
    heading = example["heading"]
    text = example["text"]
    example["text"] = f"{heading}\n{text}"
    example["label"] = example["bias_rating"]
    return example


dataset = dataset.map(concatenate_header_text)
print(dataset)

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# ---- Stats Overall
print("-" * 20, "Stats Overall", "-" * 20)
print("-" * 10, "Ratios", "-" * 10)
ratios = get_dataset_ratios(dataset, "label")
print(json.dumps(ratios, indent=4))
print("-" * 10, "Lengths", "-" * 10)
length_stats = get_dataset_length_stats(tokenizer, dataset)
print(json.dumps(length_stats, indent=4))

# ---- Stats per split
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

print("-" * 20, "Stats Per Split", "-" * 20)
print("-" * 10, "Ratios", "-" * 10)
ratios = get_dataset_ratios(dataset, "label")
print(json.dumps(ratios, indent=4))

print("-" * 10, "Lengths", "-" * 10)
length_stats = get_dataset_length_stats(tokenizer, dataset)
print(json.dumps(length_stats, indent=4))
