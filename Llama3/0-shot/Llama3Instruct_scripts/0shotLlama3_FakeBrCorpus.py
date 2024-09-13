import pandas as pd
import transformers
from transformers import LlamaTokenizer,pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, AutoModelForSequenceClassification
import os
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import evaluate 
import torch
import re
import time
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json

random.seed(42)

start_time = time.time()
configuration = "10-shot"
n_example = 5
temp = 0.1
sys_ans = []

tsv_directory = '/home/michele.maggini/datasets/FNNet'
filename = 'FakeNewsNet.csv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path)
#df = df.groupby('bias_rating').head(n_example)
df['label'] = df['label'].map({'real': 0, 'fake': 1})
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'o_proj', 'k_proj', 'v_proj'],
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


prompt = """
### Instruction:
Given a news headline determine if it is real or fake news.
The output consists in only one integer with this format: 'integer'. If the news headline is fake the output will be '1', if real '0'.
I am providing you some examples:
#Example: McCain: McSame as Bush, '0'
#Example: BREAKING: First NFL Team Declares Bankruptcy Over Kneeling Thugs,'1'
#Example: A Solar Grand Plan: Scientific American, '0'
#Example: Trump Votes For Death Penalty For Being Gay,'1'
#Exmaple: Oscar Pistorius Attempts To Commit Suicide, '1'
#Exmaple: New York Man Wanted For Infecting 240 Men And Women With HIV!!!,'1'
#Exmaple: Saudi Arabia to Behead 6 School Girls for Being With Their Male Friends Without Parents or a Guardian,'1'
#Exmaple: Malia Obama Fired From Cushy Internship At Spanish Embassy,'1'
#Exmaple: Cindy Crawford and Son Presley Gerber to Star in New Pepsi Commercial, '0'
#Exmaple: WATCH: First Trailer for Bobbi Kristina Brown TV Biopic, '0'
###Input:
This is the news headlines: {}

### Response:
"""



for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "You are an expert in Political Communication Science. You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt.format(row['title'])}
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
      max_new_tokens=1,
      temperature = temp
  )
  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  sys_ans.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

print(sys_ans)

sys_ans = [1 if item == '1' else 0 for item in sys_ans]

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


results=compute_metrics(sys_ans, gold_ans)
print(prompt)
print(f"Allsides, {configuration}, prompt with temperature={temp},  n_labels=ALL:\n", json.dumps(results, indent=4))
# Compute precision, recall, and f1-score
print(f"Allsides, {configuration}, prompt with temperature={temp}, n_labels=ALL:\n", classification_report(gold_ans,sys_ans))

print("My program took", time.time() - start_time, "to run")
