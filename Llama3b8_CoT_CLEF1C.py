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
data = "CLEF_1C_English"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
n_example = 5
###Dataset###
tsv_directory = '/home/michele.maggini/dataset/CLEF_1C/CLEF_1c'
filename = 'CLEF_1C_Dutch.tsv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path, sep="\t", engine="python")
#df = df.groupby('class_label').head(n_example)

prompt_0_spec = """### Instructie:
Er wordt een tweet gegeven, voorspel of deze schadelijk is voor de maatschappij. Als deze schadelijk is, labelt u deze als '1', als deze niet schadelijk is, labelt u deze als '0'. Het uitvoerlabelformaat is 'integer'.

Stapsgewijze redenering:
Identificatie van trefwoorden en zinnen: Maak een lijst met woorden of zinnen die schadelijke informatie en intenties kunnen bevatten.
Sentimentanalyse: Analyseer de toon en taal om te zien of er onbedoeld of opzettelijk schadelijk gedrag, woorden of emotionele taal is.
Identificatie van de doelgroep: Wie lijkt de doelgroep voor deze tweet te zijn?
Bedreigingen en intimidatie: Directe of verhulde bedreigingen van geweld, fysieke schade of intimidatie. Dit omvat het onthullen van persoonlijke informatie (doxing), stalking of het aanzetten van anderen tot intimidatie.
Vijandige taal: Uitingen van haat, intolerantie of vooroordelen jegens specifieke groepen. Dit kan zich manifesteren als onmenselijke taal, oproepen tot geweld of promotie van discriminerende ideologieën.
Desinformatie en valse informatie: Het verspreiden van valse of misleidende informatie, vaak met de bedoeling om te misleiden of te manipuleren. Dit kan samenzweringstheorieën, nepnieuws of gemanipuleerde media omvatten.
Cyberpesten: Agressief of opzettelijk gedrag waarbij elektronische communicatie wordt gebruikt om een ​​individu te schaden, te vernederen of te bedreigen.
De tweet is: {}

### Reactie:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "Je hebt een instructie ontvangen die een taak beschrijft en deze is gecombineerd met een invoer die meer context biedt. Reageer zoals aangegeven in de instructie."},
      {"role": "user", "content": prompt_0_spec.format(row['tweet_text'])}
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
filename = 'CLEF1C_Dutch.csv'
df_ans.to_csv(filename, index=False)