from transformers import GenerationConfig
from unsloth import FastLanguageModel
import re
import pandas as pd
import os
import random
import itertools
import json
import evaluate

random.seed(42)
max_seq_length = 4000 # Choose any! Unsloth supports RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
#shot= "4"
configuration = "CoT"
data = "FakeNewsNet"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
n_example = 3
###Dataset###
tsv_directory = '/home/michele.maggini/dataset'
filename = 'FakeNewsNet.csv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path)
df = df.groupby('label').head(n_example)

prompt_0_spec = """### Instruction: 
Determine if the news title is false or true by following the reasoning steps below. You will return the explanation and the result as a single integer. The result will be just a label like this: 'integer'. You will only return the result as an integer like 'integer': '0' if the article is true, '1' if it is false.

Reason step by step:
Identify keywords and phrases that may omit information: list words or phrases that could indicate illogical thinking.
Sentiment analysis: Analyze the tone and language to see if there are polarizing words or emotional language.
Are rhetorical devices used? (e.g., hyperbole, loaded questions)
Look for content: How likely is it that the content was written by a bot?
Identify the target audience: Who appears to be the target audience for this headline?
Adequate information: Is the information presented comprehensively in the news? Is it reliable?
Consider omissions and emphasis: What information is emphasized and what might be omitted?
Here is the news title: {}

### Response:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  print(row['title'])
  messages = [
      {"role": "system", "content": "You are an expert in Political Communication. You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt_0_spec.format(row['title'])}
  ]
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

  # Directly use generate() and tokenizer.decode() to get the output.
  # Use `max_new_tokens` to control the maximum output length.
  generated_ids = model.generate(
      model_inputs.input_ids,
      max_new_tokens=500,
      temperature = temp
  )
  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  sys_ans.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

#df_ans=pd.DataFrame(sys_ans)
#filename = 'FNNET_sub_sys_ans_CoT_.csv'
#df_ans.to_csv(filename, index=False)