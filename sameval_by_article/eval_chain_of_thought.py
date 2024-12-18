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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="CoT Eval script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)


main_run = wandb.init(
    project="SemEval-b-article",
    entity="michelej-m",
    name=f"{args.model_name.split('/')[1]}_CoT",
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
data_name = "semeval2019"  ###This is the initial part of the name in DPP_points


base_dir = os.path.dirname(os.path.abspath(__file__))


data_files = {
    "train": os.path.join(base_dir, "./data/train.tsv"),
    "test": os.path.join(base_dir, "./data/test.tsv"),
}


dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

dataset = dataset.rename_column("sentence", "text")
print(dataset)


prompt = """
### Instruction: Determine if an article is hyperpartisan by following the reasoning steps below. You’ll return the explanation (max 50 words per step) and the output of the label as only one integer. The possible choices are: ’0’ if the article is neutral, ’1’ if the article is hyperpartisan. The output will be only one label like this example: ’integer’. 

Reason step by step as follows: 
1. Sentiment analysis: Analyze the tone and language to see if there are polarizing words or emotional language.
2. Rhetorical bias: Rhetoric refers to speaking or writing designed to have a persuasive or impressive effect but lacking meaningful content. Analyze the presence of rhetorical biases like ad hominem attacks.
3. Framing bias: Assess how the information is presented to shape or influence perceptions by emphasizing certain aspects while downplaying others.
4. Ideological bias: Determine if specific moral values linked to a particular ideology appear when carefully reading the text.
5. Intention: Analyze the intent of the article. Does it aim to persuade or merely inform?
6. Unilateral coverage: Does the article provide only one point of view? Is it unilateral in its coverage?
7. Left-wing hyperpartisan: Consider yourself a left-wing reader. Would you consider this article hyperpartisan from your political stance? [Follow the instructions from 1 to 6.]
8. Right-wing hyperpartisan: Consider yourself a right-wing reader. Would you consider this article hyperpartisan from your political stance? [Follow the instructions from 1 to 6.]
9. Final prediction: the output format should be: ’label’. All labels are integers: 0 for neutral, 1 for hyperpartisan. [Follow the instructions skipping steps 7 and 8.] 


The output will be in this format. Remember to use all reasoning steps followed by <end>:

Example output:
# Sentiment analysis: EXPLANATION: 
# Rhetorical bias: EXPLANATION <end>
# Framing bias: EXPLANATION <end>
# Ideological bias: EXPLANATION <end>
# Intention detection: EXPLANATION<end>
# Unilateral coverage: EXPLANATION<end>
# Left-wing: EXPLANATION <end>
# Right-wing: EXPLANATION <end>
# Final prediction: INTEGER  LABEL<end>

### Input:
{}

### Response:
"""


def parse_label(model_output):
    match = re.search(r"\b[0-1]\b", model_output)
    return int(match.group()) if match else None


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

results = {}
model_outputs = {}

irregular_outputs = 0
preds = []
refs = []

start_time = time.time()
for element in dataset["test"]:
    pred = generate(model, tokenizer, prompt, element["text"])

    args.verbose and print(" > Pred: ", pred)
    args.verbose and print(" > Ref: ", element["label"])

    if parse_label(pred) is None:
        print(" > Irregular output:  ", pred)

        print("*" * 5, "Trying to resolve irregularity", "*" * 5)
        while True:
            pred = generate(model, tokenizer, prompt, element["text"], temperature=0.7)
            print(" >> Attempted Pred: ", pred)

            if parse_label(pred) is not None:
                print(" >> Regularized output: ", pred)
                break

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
