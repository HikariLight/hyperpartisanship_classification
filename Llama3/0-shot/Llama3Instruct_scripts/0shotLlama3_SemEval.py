####FEW_SHOT###

import os
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
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
import json
import time


start_time = time.time()


local_dataset_path2 = ""
df2= pd.read_csv(local_dataset_path2, delimiter='\t', header=None)
df2.columns=['sentence', 'label']
local_dataset_path = "/home/michele.maggini/datasets/train.tsv"
df1 = pd.read_csv(local_dataset_path, delimiter='\t')
df = pd.concat([df1, df2], ignore_index=True, sort=False)
df = df.groupby('label').head(30)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'output_proj'],
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

device = torch.device('cuda:1')
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map=device, 
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.float16)


prompt = """
### Instruction:
You are an expert in Political Science and your task is to detect hyperpartisan news. I'll provide some examples.
Hyperpartisan news is labelled with 1, and neutral ones with 0. 
Given an article, determine and label it as indicated if a news is hyperpartisan or not. The output will be only one integer. Please provide as output only a python dictionary with one labels: dic=('label':label).

Example 1: Trump Just Woke Up Viciously Attacked Puerto Ricans On Twitter Like A Cruel Old Man: Donald Trump ran on many braggadocios and largely unrealistic campaign promises. One of those promises was to be the best, the hugest, the most competent infrastructure president the United States has ever seen. Trump was going to fix every infrastructure problem in the country and Make America Great Again in the process. That is, unless you’re a brown American. In that case, you’re on your own, even after a massive natural disaster like Hurricane Maria. Puerto Rico’s debt, which the Puerto Rican citizens not in government would have no responsibility for, has nothing to do with using federal emergency disaster funds to save the lives of American citizens there. The infrastructure is certainly a mess at this point after a Category 5 hurricane ripped through the island, and 84 percent of Puerto Rican people are currently without electricity. Emergency efforts after Hurricanes Irma and Harvey reportedly went very well and Trump praised himself as well and even saw his disastrous approval ratings tick up slightly as a result. However, the insufficient response in Puerto Rico has nothing to do with Trump, in his mind, and can only be blamed on the people there who do not live in a red state and have no electoral college votes to offer the new president for 2020. They’re on their own. Twitter responded with sheer incredulity at Trump’s vicious attack on an already suffering people. Featured image screengrab via YouTube // 1
Example 2: Chappaqua's Hillary Clinton Lets 'Guard Down,' Will Reveal 'What Happened': CHAPPAQUA, N.Y. -- Hillary Clinton will be telling her side of the story. Clinton, a Chappaqua resident, is set to write "What Happened," her recounting of her loss to Donald Trump in the 2016 presidential election. The book will be published on Sept. 12 by Simon and Schuster. “In the past, for reasons I try to explain, I’ve often felt I had to be careful in public, like I was up on a wire without a net. Now I’m letting my guard down," Clinton, who previously served as Senator and Secretary of State, said. The book will discuss what it was like to face Trump, Russian interference and the experience Clinton had of being the first woman nominated for president by a major party. Simon and Schuster is calling it "her most personal memoir yet." Clinton had previously written the memoirs "Living History," and "Hard Choices." In November, Clinton is set to speak at the Business Council of Westchester's annual dinne r, where she will receive the Westchester Global Leadership Laureate Award Click here to sign up for Daily Voice's free daily emails and news alerts. // 0


### Input:
{}

### Response:
"""
sys_ans = []
for index, row in df[:2].iterrows():
    if index % 100 == 0:  # Check if index is a multiple of 100
        print(f"Reached iteration {index}")
        continue
    messages = [
      {"role": "system", "content": "You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt.format(row['sentence'])}
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
      max_new_tokens=512
  )
    generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

    sys_ans.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
#print(sys_ans)

import ast
import re

def extract_specific_pairs(text):
    # Utilizzare regex per trovare il contenuto del dizionario nel testo
    dict_match = re.search(r"\{.*\}", text, re.DOTALL)
    
    if not dict_match:
        raise ValueError("Nessun dizionario trovato nel testo fornito.")
    
    dict_content = dict_match.group(0)
    
    # Convertire la stringa del dizionario estratta in un dizionario Python
    extracted_dict = ast.literal_eval(dict_content)
    
    # Estrarre le coppie chiave-valore specifiche
    keys_of_interest = ['label']
    result_dict = {key: extracted_dict[key] for key in keys_of_interest if key in extracted_dict}
    
    return result_dict

# Esempio di utilizzo
answer=[]
for el in sys_ans:

    result_dict = extract_specific_pairs(el)
    answer.append(result_dict)


list_label=[el['label'] for el in answer]
print("list_label:", list_label)

#sys_ans = [1 if item == '1' else 0 for item in sys_ans]
#print("sys_ans:", sys_ans)

from sklearn.metrics import precision_recall_fscore_support, classification_report

gold_ans = df['label'][:2].tolist()
print("gold_ans:", gold_ans)

# Compute precision, recall, and f1-score
print(classification_report(gold_ans,list_label))

print("My program took", time.time() - start_time, "to run")
