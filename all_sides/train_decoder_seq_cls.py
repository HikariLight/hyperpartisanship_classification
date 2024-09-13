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

parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()
print(args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.init(
    project="AllSides",
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
        device_map=device,
        num_labels=3,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    # ---- Dataset loading + Processing
    max_len = 512

    dataset_path = "./data/allsides_balanced_news_headlines-texts.csv"
    dataset = load_dataset("csv", data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=0.2)

    labels = ["left", "center", "right"]

    def concatenate_header_text(example):
        heading = example["heading"]
        text = example["text"]
        example["text"] = f"{heading}\n{text}"
        example["label"] = labels.index(example["bias_rating"])
        return example

    def tokenization_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)

    processed_dataset = dataset.map(concatenate_header_text)
    tokenized_dataset = processed_dataset.map(tokenization_func)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["tags", "source", "Unnamed: 0", "title", "heading", "text", "bias_rating"]
    )
    print(tokenized_dataset)

    length_stats = get_dataset_length_stats(tokenizer, processed_dataset)
    print(json.dumps(length_stats, indent=4))

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
        report_to="wandb",
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
