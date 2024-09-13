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
dataset = "Vista"
#shot= "4"
configuration = "CoT"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


tsv_directory = '/home/michele.maggini/dataset'
train_filename = 'train.tsv'
test_filename = 'test.tsv'

train_path = os.path.join(tsv_directory, train_filename)
test_path = os.path.join(tsv_directory, test_filename)
df_train = pd.read_csv(train_path, delimiter="\t")
df_test = pd.read_csv(test_path, delimiter="\t")
df_test.columns = ['sentence','label']
df = pd.concat([df_train, df_test], ignore_index=True, sort=False)
#n_example=20
#df = df.groupby('label').head(n_example)
#df = df.groupby('label').tail(n_example)

prompt_0_spec = """### Instruction: 
Determine if the article is hyperpartisan by following the instruction below. You'll return the explanation and the the output of the label as an integer. The output format is 'label': '1' if the headline is hyperpartisan, else '0'.
Reasoning steps:
1. Sentiment analysis: Analyze the tone and the language to see if there are polarizing words, or emotional language.
2. Rhetorical bias: Rhetoric refers to speaking or writing that is designed to have a persuasive or impressive effect but is lacking in meaningful content. Analyze the presence of rhethorical biases like ad hominem attack.
3. Framing bias:  involves presenting information to shape or influence people's perceptions of an issue or event by emphasizing certain aspects while downplaying others  
4. Ideological bias: Analyze the presence of ideological bias, that is if by carefully reading the text specific moral values appear related to a specific ideology.
5. Intention: Analyze the intent of the article. Does it aim to persuade or just to inform?
6. Unilateral: Does the article provide only a point of view on the subject? Is the article unilateral in its coverage of the subject?
7. Left-wing hyperpartisan: Consider you are a left-wing reader. Would you consider this article as hyperpartisan to your political stance?
8. Right-wing hyperpartisan: Consider you are a right-wing reader. Would you consider this article as hyperpartisan to your political stance?

### Input:
{}

### Response:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt_0_spec.format(row['sentence'])}
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

df_ans=pd.DataFrame(sys_ans)
filename = 'CLEF3A_sys_ans_CoT.csv'
df_ans.to_csv(filename, index=False)