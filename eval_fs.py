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
#import wandb
from utils import compute_metrics, compute_fews_hot_nested_avg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Few-Shot Evaluation Script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument(
    "--method", 
    type=str, 
    default="fs_dpp",
    choices=["fs_dpp", "fs_random"],
    help="Evaluation method: fs_dpp (DPP few-shot) or fs_random (random few-shot)"
)
parser.add_argument("--language", type=str, default="", help="Language for prompts ('bg', 'en', 'pt')")
args = parser.parse_args()

print("Parsed Arguments:", args)

"""
main_run = wandb.init(
    project="AllSides",
    entity="michelej-m",
    name=f"[{args.method}] {args.model_name.split('/')[1]}_few_shot",
    reinit=True,
)
main_run.log({"num_runs": 5})
"""

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
    #attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
print(model.generation_config)

# ---- Dataset loading/Processing
"""if args.dataset_name == "clef_1c":
    dataset_path_train = f"./processed_data/train_{args.language}.json"
    dataset_path_test = f"./processed_data/test_{args.language}.json"
else:"""
dataset_path_train = f"./processed_data/train.json"
dataset_path_test = f"./processed_data/test.json"

# Load the dataset with explicit splits
dataset = load_dataset("json", data_files={"train": dataset_path_train, "test": dataset_path_test})
print(dataset['train'][0])
num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# Load prompts for few-shot evaluation
prompt_path = "./prompts_ICWSM.json"

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

if args.dataset_name == "clef_1c":
    prompts = prompts[f"{args.dataset_name}_{args.language}"]["few_shot"]
else:
    prompts = prompts[args.dataset_name]["few_shot"]

print(f"Using prompts: {prompts}")
prompt = f'"""\n{prompts}\n"""'

# Load DPP few-shot examples if using the DPP method
if args.method == "fs_dpp":
    if args.dataset_name == "clef_1c":
        with open(f"./data/{args.dataset_name}/{args.dataset_name}_{args.language}_5_runs.json") as json_file:
            few_shot_examples = json.load(json_file)
    else:
        with open(f"./data/{args.dataset_name}/{args.dataset_name}_5_runs.json") as json_file:
            few_shot_examples = json.load(json_file)

def parse_label(model_output):
    match = re.search(r"\b[0-2]\b", model_output)
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

def construct_few_shot_string_dpp(data, example_set, few_shot_n, num_labels):
    few_shot_string = ""

    examples_per_label = few_shot_n // num_labels

    for label in data[example_set]:
        for n in range(examples_per_label):
            few_shot_string += data[example_set][label][n] + "\n"

    return few_shot_string

def construct_few_shot_string_random(few_shot_examples):
    few_shot_string = ""

    for item in few_shot_examples:
        few_shot_string += f"{item['text']},{item['label']}\n"

    return few_shot_string

# ---- Run DPP Few-Shot evaluation
def run_dpp_evaluation():
    results = {}
    model_outputs = {}

    start_time = time.time()
    for few_shot_set in few_shot_examples:
        print("=" * 15, f" Running evals using: {few_shot_set} ", "=" * 15)

        results[few_shot_set] = {}
        model_outputs[few_shot_set] = {}

        for n in range(
            num_labels, 11, num_labels
        ):  # 10 shot with a step size of num_labels
            print("-" * 10, f" Evaluating {n}-shot ", "-" * 10)
            results[few_shot_set][f"{n}_shot"] = {}

            few_shots_string = construct_few_shot_string_dpp(
                few_shot_examples, few_shot_set, n, num_labels
            )

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
            results[few_shot_set][f"{n}_shot"] = evals
            model_outputs[few_shot_set] = {"ground_truth": refs, "model_predictions": preds}
            print(json.dumps(evals, indent=4))

        args.verbose and print(json.dumps(results, indent=4))

    final_evals = compute_fews_hot_nested_avg(results)
    print(json.dumps(final_evals, indent=4))

    print(f" > Inference execution time: {(time.time() - start_time):.2f}s")
    
    return results, model_outputs, final_evals

# ---- Run Random Few-Shot evaluation
def run_random_evaluation():
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
        
        # Get all unique labels
        dataset_labels = list(set(example["label"] for example in dataset["train"]))

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
                random_element = filtered_dataset[random_index]
                if random_element not in few_shot_examples:
                    break
            few_shot_examples.append(random_element)

            # ---- Constructing few-shot example
            few_shots_string = construct_few_shot_string_random(few_shot_examples)
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
    
    return results, model_outputs, final_evals, run_settings

# ---- Run the selected evaluation method
if args.method == "fs_dpp":
    results, model_outputs, final_evals = run_dpp_evaluation()
    
    # ---- Saving results/outputs to JSON files
    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    with open("avg_results.json", "w") as json_file:
        json.dump(final_evals, json_file, indent=4)

    with open("model_outputs.json", "w") as json_file:
        json.dump(model_outputs, json_file, indent=4)
    
    """
    main_run.save("results.json")
    main_run.save("avg_results.json")
    main_run.save("model_outputs.json")

    main_run.finish()

    # ---- Logging average metrics as separate runs
    for few_shot_config in final_evals:
        run = wandb.init(
            project="AllSides",
            entity="michelej-m",
            name=f"[DPP] {few_shot_config}",
            reinit=True,
        )
        run.log({"num_runs": 5})

        for metric in final_evals[few_shot_config]:
            run.log({f"avg_{metric}": final_evals[few_shot_config][metric]["score"]})

        run.finish()
    """
    
elif args.method == "fs_random":
    results, model_outputs, final_evals, run_settings = run_random_evaluation()
    
    # ---- Saving results/outputs to JSON files
    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    with open("avg_results.json", "w") as json_file:
        json.dump(final_evals, json_file, indent=4)
    
    with open("model_outputs.json", "w") as json_file:
        json.dump(model_outputs, json_file, indent=4)
    
    with open("run_settings.json", "w") as json_file:
        json.dump(run_settings, json_file, indent=4)
    
    """
    main_run.save("results.json")
    main_run.save("avg_results.json")
    main_run.save("model_outputs.json")
    main_run.save("run_settings.json")
    
    main_run.finish()
    
    # ---- Logging average metrics as separate runs
    for few_shot_config in final_evals:
        run = wandb.init(
            project="AllSides",
            entity="michelej-m",
            name=f"[Random] {few_shot_config}",
            reinit=True,
        )
        run.log({"num_runs": 5})

        for metric in final_evals[few_shot_config]:
            run.log({f"avg_{metric}": final_evals[few_shot_config][metric]["score"]})

        run.finish()
    """