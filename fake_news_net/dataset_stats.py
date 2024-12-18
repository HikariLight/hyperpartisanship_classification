from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
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
max_len = 512
fake_dataset = load_dataset("csv", data_files="./data/politifact_fake.csv")
real_dataset = load_dataset("csv", data_files="./data/politifact_real.csv")

fakes = fake_dataset["train"].to_list()
real = real_dataset["train"].to_list()

for i in range(len(fakes)):
    fakes[i]["label"] = 0

for i in range(len(real)):
    real[i]["label"] = 1

dataset = Dataset.from_list(fakes + real)
dataset = dataset.remove_columns(["id", "news_url", "tweet_ids"])
dataset = dataset.rename_column("title", "text")
print(dataset)

# ---- Stats
print("-"*20, "Stats Overall", "-"*20)
# ratios = get_dataset_ratios(dataset, "label")
# print(json.dumps(ratios, indent=4))

# length_stats = get_dataset_length_stats(tokenizer, dataset)
# print(json.dumps(length_stats, indent=4))


print("-"*20, "Stats Per Split", "-"*20)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
print(dataset)
ratios = get_dataset_ratios(dataset, "label")
print(json.dumps(ratios, indent=4))

length_stats = get_dataset_length_stats(tokenizer, dataset)
print(json.dumps(length_stats, indent=4))
