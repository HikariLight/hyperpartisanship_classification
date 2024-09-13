import numpy as np


def compute_average_metrics(data):
    metrics = ["eval_accuracy", "eval_recall", "eval_precision", "eval_f1-score"]
    result = {}

    for metric in metrics:
        scores = [element[metric] for element in data]
        avg = np.mean(scores)
        minimum = np.min(scores)
        maximum = np.max(scores)
        stddev = np.std(scores)

        result[metric] = {
            "score": avg,
            "stddev": stddev,
            "min": minimum,
            "max": maximum,
        }

    return result


def get_dataset_length_stats(tokenizer, dataset):
    result = {}

    for split in dataset:
        lengths = []
        for element in dataset[split]:
            tokens = tokenizer(element["text"], return_tensors="pt")
            lengths.append(tokens["input_ids"].shape[1])

        result[split] = {
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
        }

    return result
