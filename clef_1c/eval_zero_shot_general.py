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
import json
import re
import wandb
from utils import compute_metrics

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
    name=f"[General] {args.model_name.split('/')[1]}_zero_shot",
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
prompts = {
    "english": {
        "system": "You have received an instruction that describes a task and it has been combined with an input that provides more context. Respond as directed in the instruction.",
        "user": "### Instruction:\nDetect if a tweet is harmful to society or not. The possible choices are: '0' if the article is neutral, '1' if the tweet is harmful.\nThe output of the label is only one integer like this example: 'integer'.\n\n###Input:\n{}\n\n###Response:",
    },
    "bulgarian": {
        "system": "Получихте инструкция, която описва задача и тя е комбинирана с въвеждане, което предоставя повече контекст. Отговорете, както е указано в инструкцията.",
        "user": "### Инструкция:\nОткрийте дали един туит е вреден за обществото или не. Възможните избори са: „0“, ако статията е неутрална, „1“, ако туитът е вреден.\nРезултатът от етикета е само едно цяло число като този пример: „цяло число“.\n\n###Вход:\n{}\n\n###Отговор:",
    },
    "arabic": {
        "system": "لقد تلقيت تعليمات تصف مهمة وتم دمجها مع مدخلات توفر سياقًا أكثر. استجب وفقًا للتوجيهات الواردة في التعليمات.",
        "user": "### التعليمات:\nاكتشف ما إذا كانت التغريدة ضارة بالمجتمع أم لا. الخيارات الممكنة هي: '0' إذا كانت المقالة محايدة، و'1' إذا كانت التغريدة ضارة.\nيكون ناتج العلامة عددًا صحيحًا واحدًا فقط مثل هذا المثال: 'integer'.\n\n###مثال:\n{}\n\n###الاستجابة:",
    },
}

prompt = prompts[args.language.lower()]["user"]
system_prompt = prompts[args.language.lower()]["system"]


def parse_label(model_output):
    match = re.search(r"\b[0-1]\b", model_output)
    return int(match.group()) if match else None


def generate(model, tokenizer, prompt, element, temperature=0.1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.format(element)},
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


# ---- Inference
dataset_labels = list(set(dataset["train"]["label"]))

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
        while True:
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

results = compute_metrics(preds, refs)
results["irregular_outputs"] = irregular_outputs
model_outputs = {
    "ground_truth": refs,
    "model_predictions": preds,
}
print(json.dumps(results, indent=4))

print(f" > Inference execution time: {(time.time() - start_time):.2f}s")

for metric in results:
    main_run.log({f"avg_{metric}": results[metric]})

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
main_run.save("results.json")

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)
main_run.save("model_outputs.json")

main_run.finish()
