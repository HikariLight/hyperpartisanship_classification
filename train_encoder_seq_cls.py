import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
import evaluate
import wandb
import argparse
import json
from utils import compute_average_metrics, get_dataset_length_stats

parser = argparse.ArgumentParser(prog="Sequence Classification Training Script")

parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--save", action="store_true")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--configuration", type=str, default="")
parser.add_argument("--language", type=str, default="", help="Language for prompts ('bg', 'en', 'pt')")
args = parser.parse_args()
print(args)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.init(
    project=args.dataset_name,
    entity="michelej-m",
    name=args.model_name.split("/")[1],
)
wandb.log({"num_runs": args.runs})

results = []

for _ in range(args.runs):
    # ---- Model / Tokenizer loading
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map=DEVICE,
        num_labels=2,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        # target_modules=["query", "value"],
        # target_modules=["query_proj", "value_proj"],
    )
    model = get_peft_model(model, lora_config)

    # # ---- Dataset loading + Processing
    max_len = 512

    # ---- Dataset loading/Processing
    if args.dataset_name == "clef_1c":
        dataset_path_train = f"./processed_data/train_{args.language}.json"
        dataset_path_test = f"./processed_data/test_{args.language}.json"
    else:
        dataset_path_train = f"./processed_data/train.json"
        dataset_path_test = f"./processed_data/test.json"

    # Load the dataset with explicit splits
    dataset = load_dataset("json", data_files={"train": dataset_path_train, "test": dataset_path_test})
    print(dataset['train'][0])
    num_labels = len(dataset["train"].unique("label"))
    print(" > Label num: ", num_labels)

    def tokenization_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    tokenized_dataset = dataset.map(tokenization_func)

    length_stats = get_dataset_length_stats(tokenizer, dataset)
    print(json.dumps(length_stats, indent=4))
    if args.configuration != "ft_encoder":
        raise ValueError("Invalid configuration. Please use 'ft_encoder' as the configuration name. REMEMBER YOU ARE USING ft_encoder HERE!!!")

    # ------Training prep
    # -- Hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs

    def compute_metrics(eval_pred):
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"]

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="hyperpartisanship_classification",
        learning_rate=lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        eval_strategy="epoch",
        #report_to="wandb",
        logging_steps=5,
        fp16=True,
        gradient_checkpointing=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    run_results = trainer.evaluate(tokenized_dataset["test"])
    results.append(run_results)
    print(json.dumps(run_results, indent=4))

avg_results = compute_average_metrics(results)
wandb.log(
    {
        "avg_accuracy": avg_results["eval_accuracy"]["score"],
        "avg_precision": avg_results["eval_precision"]["score"],
        "avg_recall": avg_results["eval_recall"]["score"],
        "avg_f1_score": avg_results["eval_f1-score"]["score"],
    }
)
print(json.dumps(avg_results, indent=4))
