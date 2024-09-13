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
dataset = "ALLSIDES"
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
filename = 'sub_allsides.csv'

df_path = os.path.join(tsv_directory, filename)

df = pd.read_csv(df_path)


#df = df[['label', 'article']]
#n_example= 1000
#df = df.groupby('label').head(n_example)
#filename = 'CLEF3A_sampled.csv'
#df.to_csv(filename, index=False)
#df = df.groupby('label').tail(n_example)

prompt_0_spec = """### Instruction: 
Determine if the headline is hyperpartisan by following the reasoning steps below. You'll return the explanation and the the output of the label as only one integer. 
The output will be only one label like this: 'integer'. You'll return only the output as an integer like 'integer': '0' if the article is left-wing, '1' if the political leaning is from the center, '2' if the article is right-wing'.
Reason step by step:
1. Identify key words and phrases: list of words or phrases that might indicate political bias.
2. Sentiment analysis: Analyze the tone and the language to see if there are polarizing words, or emotional language.
3. Are there any rhetorical devices used? (e.g., hyperbole, loaded questions)
4. Look for policy implications: Does the headline suggest support for or opposition to specific policies? Are there any implicit assumptions about political or economic systems?. 
5. Identify target audience: Who seems to be the intended audience for this headline?
6. Left-wing hyperpartisan: Consider you are a left-wing reader. Would you consider this article as hyperpartisan to your political stance?
7. Right-wing hyperpartisan: Consider you are a right-wing reader. Would you consider this article as hyperpartisan to your political stance?
8. Consider omissions and emphasis: What information is emphasized, and what might be omitted? How might this emphasis or omission indicate bias?

### Input:
{}

### Response:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "You are an expert in Political Communication. You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
      {"role": "user", "content": prompt_0_spec.format(row['text'])}
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
filename = 'ALLSIDES_sub_sys_ans_CoT.csv'
df_ans.to_csv(filename, index=False)