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
import random
import json
import re
import wandb
from utils import compute_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Zero-Shot Evaluation Script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument(
    "--configuration",
    type=str,
    default="zero_shot_generic",
    choices=["zero_shot_generic", "zero_shot_specific", "zero_shot_cot"],
    help="Evaluation configuration: zero-shot variants",
)
parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Language for prompts ('en', 'bg', 'ar', 'es', 'pt')",
)
parser.add_argument(
    "--task_labels", type=str, default="", choices=["hp", "pl", "ht", "fn"]
)
parser.add_argument(
    "--label_type",
    type=str,
    default="string",
    choices=["string", "int"],
    help="Format of the labels in the model's output: 'string' for text labels or 'int' for integer labels",
)
args = parser.parse_args()

print("Parsed Arguments:", args)

main_run = wandb.init(
    project="AllSides",
    entity="michelej-m",
    name=f"[{args.configuration}] {args.model_name.split('/')[1]}_zero_shot",
    reinit=True,
)

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
dataset = load_dataset(
    "json",
    data_files={
        "train": "./processed_data/train.json",
        "test": "./processed_data/test.json",
    },
)
print(dataset)

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# Load prompts for zero-shot evaluation
if args.label_type == "string":
    prompt_path = "./prompts_ICWSM_str.json"
else:
    prompt_path = "./prompts_ICWSM_int.json"

with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

# Select the appropriate prompt based on task, dataset, and language
if args.task_labels == "fn" or args.task_labels == "ht":
    # Multi-language tasks
    if args.language in prompts[args.task_labels]:
        prompt_template = prompts[args.task_labels][args.language][args.configuration]
    else:
        # Exit with error if specified language not available
        print(
            f"Error: Language {args.language} not available for task {args.task_labels}."
        )
        exit(1)
else:
    # Single language tasks
    prompt_template = prompts[args.task_labels][args.configuration]

print(f"Using prompt template: {prompt_template}")
prompt = f'"""\n{prompt_template}\n"""'


def parse_label(model_output):
    # First extract content after "==>" or following "Final prediction/Answer"
    match = re.search(
        r"Final (?:prediction|Answer)(?:\s*==>|\s*:)\s*(?:\*\*)?([^\s*]+)(?:\*\*)?",
        model_output,
    )
    if not match:
        return None

    content = match.group(1).strip().lower()

    # Handle integer-based labels if specified
    if args.label_type == "int":
        # Look for "0" or "1" (and "2" for pl task) in the output
        if "0" in content:
            return 0
        elif "1" in content:
            return 1
        elif args.task_labels == "pl" and "2" in content:
            return 2
        else:
            return None

    # Handle string-based labels (default behavior)
    if args.task_labels == "hp":
        if "neutral" in content:
            return 0
        elif "hyperpartisan" in content:
            return 1

    elif args.task_labels == "fn":
        if "true" in content:
            return 0
        elif "fake" in content:
            return 1

    elif args.task_labels == "ht":
        if "neutral" in content:
            return 0
        elif "harmful" in content:
            return 1

    elif args.task_labels == "pl":
        if "center" in content:
            return 1
        elif "left" in content:
            return 0
        elif "right" in content:
            return 2

    return None


def generate(model, tokenizer, prompt, element, temperature=0.1):
    formatted_prompt = prompt.format(element)

    messages = [
        {"role": "user", "content": formatted_prompt},
    ]
    # print(messages)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=512, temperature=temperature
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
    pred = generate(model, tokenizer, prompt, element["text"])

    args.verbose and print(" > Pred: ", pred)
    args.verbose and print(f" > Ref #{idx}: {element['label']}")

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
            args.verbose and print(f" >> Attempted Pred: {pred}")

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
