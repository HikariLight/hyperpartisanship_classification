from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset, Features, Value
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
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)


main_run = wandb.init(
    project="CLEF2022task1C",
    entity="michelej-m",
    name=f"{args.model_name.split('/')[1]}_few_shot_random",
)
main_run.log({"num_runs": 5})
main_run.log({"language": args.language})

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
features = Features(
    {
        "tweet_text": Value("string"),
        "class_label": Value("int64"),
    }
)

data_files = {
    "train": f"./data/CT22_{args.language.lower()}_1C_harmful_train.tsv",
    "test": f"./data/CT22_{args.language.lower()}_1C_harmful_test_gold.tsv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", features=features)
dataset = dataset.rename_column("class_label", "label")
dataset = dataset.rename_column("tweet_text", "text")
print(dataset)

# ---- Inference utils
prompt = """
### Instruction:

Given a news headline determine if it is real or fake news.
The output consists in only one integer with this format: 'integer'. If the news headline is fake the output will be '1', if real '0'.

###Input:
{}

### Response:
"""


def parse_label(model_output):
    match = re.search(r"\d+", model_output)
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
        model_inputs.input_ids, max_new_tokens=16, temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

# ---- Inference
dataset_labels = list(set(dataset["train"]["label"]))

results = {}
model_outputs = {}

start_time = time.time()
irregular_outputs = 0
preds = []
refs = []

for element in dataset["test"]:
    pred = generate(model, tokenizer, prompt, element["text"])

    args.verbose and print(" > Pred: ", pred)
    args.verbose and print(" > Ref: ", element["label"])

    if parse_label(pred) is None:
        print(" > Irregular output:  ", pred)

        print("*" * 5, "Trying to resolve irregularity", "*" * 5)
        for _ in range(
            5
        ):  # Try 5 times to resolve  the irregularity with higher temperature
            pred = generate(
                model,
                tokenizer,
                prompt,
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
model_outputs = {
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

main_run.finish()

# ---- Logging average metrics as separate runs
for few_shot_config in final_evals:
    run = wandb.init(
        project="CLEF2022task1C",
        entity="michelej-m",
        name=f"random_{few_shot_config}",
        reinit=True,
    )
    run.log({"num_runs": 5})
    run.log({"language": args.language})

    for metric in final_evals[few_shot_config]:
        run.log({f"avg_{metric}": final_evals[few_shot_config][metric]["score"]})

    run.finish()
