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
from utils import compute_metrics, compute_fews_hot_nested_avg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)


# --- Params parsing
parser = argparse.ArgumentParser(prog="DPP Few-Shot Eval script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)

main_run = wandb.init(
    project="Fake.br_Corpus",
    entity="michelej-m",
    name=f"[DPP] {args.model_name.split('/')[1]}_few_shot",
    reinit=True,
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
dataset = load_dataset(
    "csv", data_files="./data/Fake.br-Corpus.tsv", delimiter="\t"
)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

def format_func(element):
    element["label"] = ["true", "fake"].index(element["label"])
    return element

dataset = dataset.map(format_func)
print(dataset)
print(dataset["train"][0])

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# Loading DPP Few-Shot examples
with open("./data/FakeBr_5_runs.json") as json_file:
    few_shot_examples = json.load(json_file)


# ---- Inference utils
prompt = """
### Instrução:

Dado um artigo, determine se é uma notícia real ou falsa.
A saída consiste em apenas um inteiro com este formato: 'integer'. Se a manchete da notícia for falsa, a saída será '1', se for real, '0'.

### Exemplos
{}

###Entrada:
{}

### Resposta:
"""


def parse_label(model_output):
    match = re.search(r"\d+", model_output)
    return int(match.group()) if match else None


def generate(model, tokenizer, prompt, few_shot_examples, element, temperature=0.1):
    messages = [
        {
            "role": "system",
            "content": "Você recebeu uma instrução que descreve uma tarefa e ela foi combinada com uma entrada que fornece mais contexto. Responda conforme orientado na instrução.",
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


def construct_few_shot_string(data, example_set, few_shot_n, num_labels):
    few_shot_string = ""

    examples_per_label = few_shot_n // num_labels

    for label in data[example_set]:
        for n in range(examples_per_label):
            few_shot_string += data[example_set][label][n] + "\n"

    return few_shot_string


# ---- Inference
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

        few_shots_string = construct_few_shot_string(
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

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

with open("avg_results.json", "w") as json_file:
    json.dump(final_evals, json_file, indent=4)

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)

main_run.save("results.json")
main_run.save("avg_results.json")
main_run.save("model_outputs.json")

main_run.finish()

# ---- Logging average metrics as separate runs
for few_shot_config in final_evals:
    run = wandb.init(
        project="Fake.br_Corpus",
        entity="michelej-m",
        name=f"[DPP] {few_shot_config}",
        reinit=True,
    )
    run.log({"num_runs": 5})

    for metric in final_evals[few_shot_config]:
        run.log({f"avg_{metric}": final_evals[few_shot_config][metric]["score"]})

    run.finish()
