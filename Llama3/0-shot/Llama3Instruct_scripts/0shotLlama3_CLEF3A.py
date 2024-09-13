import os
import pandas as pd
import transformers
from transformers import LlamaTokenizer, pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, AutoModelForSequenceClassification
import os
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import precision_recall_fscore_support, classification_report
import torch
import numpy as np
import re
import time
import random
import json
import evaluate 


configuration = "General_Prompt"
#n_example = 10
temp = 0.1

tsv_directory = ''
dev_filename = 'updated_task_3A_news_article_bias_dev.tsv'
train_filename = 'updated_task_3A_news_article_bias_train.tsv'
test_filename = 'updated_task_3A_news_article_bias_test.tsv'

train_path = os.path.join(tsv_directory, train_filename)
tsv_path = os.path.join(tsv_directory, dev_filename)
test_path = os.path.join(tsv_directory, test_filename)

df_train = pd.read_csv(train_path, delimiter='\t')
df_dev = pd.read_csv(tsv_path, delimiter='\t')
df_test = pd.read_csv(test_path, delimiter='\t')

n_example=5
df = pd.concat([df_train, df_dev, df_test], ignore_index=True, sort=False)
df = df[['label', 'text']]
df = df.groupby('label').head(n_example)


model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'v_proj'],
    lora_dropout = 0.1, # dropout probability of the LoRA layers
    bias = 'none', #wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

device = torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map=device, 
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.float16)


prompt_gen = """
### Instruction:
You are an expert Political Scientist with background in Journalism.
Detect the political leaning of the article. I'll provide an example so you can learn more features.
The output will be only one label like this: 'integer'. You'll return only the output as an integer like 'integer': '0' if the article is left-wing, '1' if the political leaning is from the center, '2' if the article is right-wing'.

### Input:
{}

### Response:
"""


sys_ans = []
for index, row in df.iterrows():
    messages = [
      {"role": "system", "content": "You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt_gen.format(row['text'])}
  ]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

  # Directly use generate() and tokenizer.decode() to get the output.
  # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
      model_inputs.input_ids,
      max_new_tokens=512,
      temperature = temp
  )
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

    sys_ans.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
    print("sys_inserted")

print(sys_ans)

sys_ans = [1 if item == '1' else 2 if item == '2' else 0 for item in sys_ans]

gold_ans = df['label'].tolist()



print("gold_ans:", gold_ans)
print("list_answer:", sys_ans)



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

print(prompt)

results=compute_metrics(sys_ans, gold_ans)
print(f"CLEF3A, {configuration}, model: {model_id}, prompt with temperature={temp}, n_labels=ALL:\n", json.dumps(results, indent=4))
# Compute precision, recall, and f1-score
print(f"CLEF3A, {configuration}, model: {model_id}, prompt with temperazture={temp}, n_labels=ALL:\n", classification_report(gold_ans,sys_ans))
