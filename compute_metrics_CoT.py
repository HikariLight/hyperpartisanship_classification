import re
import pandas as pd
import os
import itertools
import json
import evaluate

cd = '/home/michele.maggini/Llama3/'
filename = 'SemEval_sys_ans_CoT_2.csv'
path = os.path.join(cd, filename)


tsv_directory = '/home/michele.maggini/dataset/'
train_filename = 'train.tsv'
test_filename = 'test.tsv'
train_path = os.path.join(tsv_directory, train_filename)
test_path = os.path.join(tsv_directory, test_filename)
df_train = pd.read_csv(train_path, delimiter="\t")
df_test = pd.read_csv(test_path, delimiter="\t")
df_test.columns = ['sentence','label']
df = pd.concat([df_train, df_test], ignore_index=True, sort=False)


df_ans = pd.read_csv(path)


def extract_integers(text):
    # Regular expression to match integers enclosed in single quotes
    pattern = r"'(\d+)'"
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Convert matches to integers
    integers = [int(match) for match in matches]
    
    return integers


gold_ans = df['label'][:20].tolist()
print("gold_ans:", gold_ans)

df_ans_int= df_ans['0'].apply(lambda x: extract_integers(x) if extract_integers(x) else [1])
df_ints = df_ans_int.to_list()



"""
list_ans = list(itertools.chain(*df_ints))
print(len(list_ans))


def compute_metrics(predictions, labels):
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")


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


results=compute_metrics(list_ans, gold_ans)
print(f"SemEval, CoT, prompt with temperature=0.1:\n", json.dumps(results, indent=4))

"""