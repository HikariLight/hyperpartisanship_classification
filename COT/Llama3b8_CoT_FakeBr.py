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

configuration = "CoT"
data = "FakeNewsNet"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto"
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
n_example = 2
###Dataset###
tsv_directory = '/home/michele.maggini/dataset/Fake.br-Corpus/'
filename = 'Fake.br-Corpus.tsv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path, delimiter="\t")
df = df.groupby('label').head(n_example)

prompt_0_spec = """### Instruction: 
Determine se o título da notícia é falso ou verdadeiro seguindo os passos de raciocínio abaixo. Você retornará a explicação e o resultado como um único número inteiro. O resultado será apenas um rótulo como este: 'inteiro'. Você retornará apenas o resultado como um inteiro, assim: 'inteiro': '0' se o artigo for verdadeiro, '1' se for falso.

Identifique palavras-chave e frases que podem omitir informações: liste palavras ou frases que possam indicar raciocínio ilógico.
Análise de sentimento: Analise o tom e a linguagem para ver se há palavras polarizadoras ou linguagem emocional.
São utilizados dispositivos retóricos?
Procure conteúdo: Qual a probabilidade de que o conteúdo tenha sido escrito por um bot?
Identifique o público-alvo: Quem parece ser o público-alvo deste título?
Informação adequada: A informação apresentada na notícia é abrangente? É confiável?

###Input:título da notícia:{}

### Response:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  #print(row['label'])
  messages = [
      {"role": "system", "content": "É um especialista em Comunicação Política. Recebeu uma instrução que descreve uma tarefa e é combinada com uma entrada que fornece mais contexto. Responda conforme indicado nas instruções."},
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

df_ans=pd.DataFrame(sys_ans)
filename = 'FakeBr.csv'
df_ans.to_csv(filename, index=False)