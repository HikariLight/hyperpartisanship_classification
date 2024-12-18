from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import Dataset, load_dataset
import argparse
import torch
import time
import random
import json
import re
import wandb
from utils import compute_metrics, compute_fews_hot_nested_avg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Randomized Few-Shot Eval script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)


main_run = wandb.init(
    project="FakeNewsNet",
    entity="michelej-m",
    name=f"[Random] {args.model_name.split('/')[1]}_few_shot",
)
main_run.log({"num_runs": 5})

# ---- Model/Tokenizer loading
if args.use_quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
print(model.generation_config)

# ---- Dataset loading/Processing
fake_split = [
    "./data/politifact_fake.csv",
    "./data/gossipcop_fake.csv",
]

real_split = [
    "./data/politifact_real.csv",
    "./data/gossipcop_real.csv",
]

fakes = load_dataset("csv", data_files=fake_split, split="train").to_list()
real = load_dataset("csv", data_files=real_split, split="train").to_list()

for i in range(len(real)):
    real[i]["label"] = 0

for i in range(len(fakes)):
    fakes[i]["label"] = 1

dataset = Dataset.from_list(fakes + real)
dataset = dataset.rename_column("title", "text")
dataset = dataset.remove_columns(["id", "news_url", "tweet_ids"])
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
print(dataset)

dataset_labels = list(set(dataset["train"]["label"]))
print(dataset_labels)

# ---- Inference utils
prompt = """
### Instruction:

Given an article, determine if it is real or fake news.
The output consists in only one integer with this format: 'integer'. If the news headline is fake the output will be '1', if real '0'.

### Examples
{}

###Input:
{}

### Response:
"""


def parse_label(model_output):
    match = re.search(r"\b[0-1]\b", model_output)
    return int(match.group()) if match else None


def generate(model, tokenizer, prompt, few_shot_examples, element, temperature=0.1):
    messages = [
        {
            "role": "system",
            "content": "You have received an instruction that describes a task and it has been combined with an input that provides more context. Respond as directed in the instruction.",
        },
        {"role": "user", "content": prompt.format(few_shot_examples, element)},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=20, temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


def construct_few_shot_string(few_shot_examples):
    few_shot_string = ""

    for item in few_shot_examples:
        few_shot_string += f"{item['text']},{item['label']}\n"

    return few_shot_string


# ---- Inference
seeds = [42, 12345, 9876, 2024, 8675309]

results = {}
model_outputs = {}
run_settings = {}

start_time = time.time()
for seed in seeds:
    print("=" * 15, f" Running evals using the seed: {seed} ", "=" * 15)
    random.seed(seed)

    results[f"seed_{seed}"] = {}
    model_outputs[f"seed_{seed}"] = {}
    run_settings[f"seed_{seed}"] = {}

    few_shot_examples = []

    for n in range(1, 10 + 1):
        print("-" * 10, f" Evaluating {n}-shot ", "-" * 10)
        results[f"seed_{seed}"][f"{n}_shot"] = {}

        # ---- Selecting a unique example from the dataset (alternating labels)
        shot_label = n % len(dataset_labels)
        filtered_dataset = dataset["train"].filter(
            lambda example: example["label"] == shot_label
        )
        while True:
            random_index = random.randint(0, len(filtered_dataset) - 1)
            random_element = dataset["train"][random_index]
            if random_element not in few_shot_examples:
                break
        few_shot_examples.append(random_element)

        # ---- Constructing few-shot example
        few_shots_string = construct_few_shot_string(few_shot_examples)
        run_settings[f"seed_{seed}"][f"{n}_shot"] = few_shot_examples[:]

        irregular_outputs = 0
        preds = []
        refs = []

        for element in dataset["test"]:
            pred = generate(model, tokenizer, prompt, few_shots_string, element["text"])

            args.verbose and print(" > Pred: ", pred)
            args.verbose and print(" > Ref: ", element["label"])

            if parse_label(pred) is None:
                print(" > Irregular output:  ", pred)

                print("*" * 5, "Trying to resolve irregularity", "*" * 5)
                while True:
                    pred = generate(
                        model,
                        tokenizer,
                        prompt,
                        few_shots_string,
                        element["text"],
                        temperature=0.7,
                    )
                    print(" >> Attempted Pred: ", pred)

                    if parse_label(pred) is not None:
                        print(" >> Regularized output: ", pred)
                        break
                irregular_outputs += 1
                continue

            preds.append(parse_label(pred))
            refs.append(element["label"])

        evals = compute_metrics(preds, refs)
        evals["irregular_outputs"] = irregular_outputs
        results[f"seed_{seed}"][f"{n}_shot"] = evals
        model_outputs[f"seed_{seed}"] = {
            "ground_truth": refs,
            "model_predictions": preds,
        }
        print(json.dumps(evals, indent=4))

    args.verbose and print(json.dumps(results, indent=4))

final_evals = compute_fews_hot_nested_avg(results)
print(json.dumps(final_evals, indent=4))

print(f" > Inference execution time: {(time.time() - start_time):.2f}s")

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
main_run.save("results.json")

with open("avg_results.json", "w") as json_file:
    json.dump(final_evals, json_file, indent=4)
main_run.save("avg_results.json")

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)
main_run.save("model_outputs.json")

with open("run_settings.json", "w") as json_file:
    json.dump(run_settings, json_file, indent=4)
main_run.save("run_settings.json")

main_run.finish()

# ---- Logging average metrics as separate runs
for few_shot_config in final_evals:
    run = wandb.init(
        project="FakeNewsNet",
        entity="michelej-m",
        name=f"[Random] {few_shot_config}",
        reinit=True,
    )
    run.log({"num_runs": 5})

    for metric in final_evals[few_shot_config]:
        run.log({f"avg_{metric}": final_evals[few_shot_config][metric]["score"]})

    run.finish()
