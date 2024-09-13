import re
import pandas as pd
import os
import itertools
import json
import evaluate

cd = '/home/michele.maggini/Llama3/'
filename = 'CLEF3A_sys_ans_CoT_test.csv'
path = os.path.join(cd, filename)


tsv_directory = '/home/michele.maggini/dataset/CLEF_3A'
dev_filename = 'updated_task_3A_news_article_bias_dev.tsv'
train_filename = 'updated_task_3A_news_article_bias_train.tsv'
test_filename = 'updated_task_3A_news_article_bias_test.tsv'

train_path = os.path.join(tsv_directory, train_filename)
tsv_path = os.path.join(tsv_directory, dev_filename)
test_path = os.path.join(tsv_directory, test_filename)

df_train = pd.read_csv(train_path, delimiter='\t')
df_dev = pd.read_csv(tsv_path, delimiter='\t')
df_test = pd.read_csv(test_path, delimiter='\t')

df = pd.concat([df_train, df_dev, df_test], ignore_index=True, sort=False)
df = df[['label', 'text']]
n_example=20
df = df.groupby('label').head(n_example)

df_ans = pd.read_csv(path)

def extract_last_line(text):
    # Split the text into lines
    lines = text.strip().split('\n')
    # Return the last line
    return lines[-1]

listLastline = df_ans['0'].apply(lambda x: extract_last_line(x))
df_lastline = listLastline.to_list()

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
labels_parsed =extract_integers(df_lastline)

def extract_numbers(string_list):
    """
    This function takes a list of strings, each representing a number,
    and returns a new list of integers.

    :param string_list: List of strings representing numbers
    :return: List of integers
    """
    return [int(num) for num in string_list]

labels_predicted = extract_numbers(labels_parsed)

#df_ans_int= df_ans['0'].apply(lambda x: extract_integers(x) if extract_integers(x) else [1])
#df_ints = df_ans_int.to_list()



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
print(f"CLEF3A, CoT, prompt with temperature=0.1:\n", json.dumps(results, indent=4))
