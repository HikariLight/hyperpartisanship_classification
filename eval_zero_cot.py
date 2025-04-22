from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset
import argparse
import torch
import time
import json
import re
import wandb
from utils import compute_metrics
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)


# --- Params parsing
parser = argparse.ArgumentParser(prog="Zero-Shot Eval script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument(
    "--configuration",
    type=str,
    default="",
    choices=["cot", "zero_shot_generic", "zero_shot_specific", "codebook"],
)
parser.add_argument(
    "--language", type=str, default="", help="Language for prompts ('bg', 'en', 'pt')"
)
parser.add_argument(
    "--task_labels", type=str, default="", choices=["hp", "pl", "ht", "fn"]
)
args = parser.parse_args()
print(args)


main_run = wandb.init(
    project=args.dataset_name,
    entity="michelej-m",
    name=f"{args.model_name.split('/')[1]}_{args.configuration}",
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
    # attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
print(model.generation_config)


# ---- Dataset loading/Processing
current_dir = os.getcwd()  # Gets the current working directory

if args.dataset_name == "clef_1c":
    dataset_path_train = os.path.join(current_dir, "processed_data", "train.json")
    dataset_path_test = os.path.join(current_dir, "processed_data", "test.json")
else:
    dataset_path_train = os.path.join(current_dir, "processed_data", "train.json")
    dataset_path_test = os.path.join(current_dir, "processed_data", "test.json")

# Load the dataset with explicit splits
dataset = load_dataset(
    "json", data_files={"train": dataset_path_train, "test": dataset_path_test}
)
print(dataset["train"][0])
num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# Load the prompts with specific config and language

prompt_path = "./prompts_ICWSM.json"
with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

if args.dataset_name == "clef_1c":
    prompts = prompts[f"{args.dataset_name}_{args.language}"][args.configuration]
else:
    prompts = prompts[args.dataset_name][args.configuration]

print(f"Using prompts: {prompts}")
prompt = f'"""\n{prompts}\n"""'


def parse_label(model_output):
    if args.task_labels == "hp":
        match = re.search(r"\b(hyperpartisan|neutral)\b", model_output.lower())

        if match:
            found_text = match.group()
            if found_text == "zero":
                return 0
            elif found_text == "one":
                return 1

    if args.task_labels == "fn":
        match = re.search(r"\b(fake|true)\b", model_output.lower())

        if match:
            found_text = match.group()
            if found_text == "true":
                return 0
            elif found_text == "fake":
                return 1

    if args.task_labels == "ht":
        match = re.search(r"(harmful|not)", model_output.lower())

        if match:
            found_text = match.group()
            if found_text == "not":
                return 0
            elif found_text == "harmful":
                return 1

    if args.task_labels == "pl":
        match = re.search(r"\b(left|center|right)\b", model_output.lower())

        if match:
            found_text = match.group()
            if found_text == "center":
                return 1
            elif found_text == "left":
                return 0
            elif found_text == "right":
                return 2

    return None


def generate(model, tokenizer, prompt, element, temperature=0.1):
    messages = [
        {
            "role": "system",
            "content": "You have received an instruction that describes a task and it has been combined with an input that provides more context. Respond as directed in the instruction.",
        },
        {"role": "user", "content": prompt.format(element)},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    if args.configuration == "cot":
        generated_ids = model.generate(
            model_inputs.input_ids, max_new_tokens=512, temperature=temperature
        )
    else:
        generated_ids = model.generate(
            model_inputs.input_ids, max_new_tokens=20, temperature=temperature
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


# ---- Inference
results = {}
model_outputs = {}

irregular_outputs = 0
preds = []
refs = []

start_time = time.time()
for idx, element in enumerate(dataset["test"]):
    print(idx)
    if idx >= 10:
        break
    pred = generate(model, tokenizer, prompt, element["text"])
    print(pred)

    args.verbose and print(" > Pred: ", pred)
    args.verbose and print(" > Ref: ", element["label"])

    if parse_label(pred) is None:
        print(" > Irregular output:  ", pred)
        print(f" > Index: {idx}")
        print("*" * 5, "Trying to resolve irregularity", "*" * 5)
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            pred = generate(
                model,
                tokenizer,
                prompt,
                element["text"],
                temperature=0.7,
            )
            print(f" >> Attempted Pred: {pred}")

            if parse_label(pred) is not None:
                print(" >> Regularized output: ", pred)
                break

            retry_count += 1

        if retry_count == max_retries:
            print(" >> Failed to get valid prediction after max retries")
            irregular_outputs += 1
            continue

    preds.append(parse_label(pred))
    refs.append(element["label"])

results = compute_metrics(preds, refs)
results["irregular_outputs"] = irregular_outputs
model_outputs = {
    "ground_truth": refs,
    "model_predictions": preds,
}

print(json.dumps(results, indent=4))

for metric in results:
    main_run.log({f"avg_{metric}": results[metric]})

print(f" > Inference execution time: {(time.time() - start_time):.2f}s")

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
main_run.save("results.json")

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)
main_run.save("model_outputs.json")

main_run.finish()
