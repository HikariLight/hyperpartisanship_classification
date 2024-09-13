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
data = "FakeNewsCorpusSpanish"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

###Dataset###
tsv_directory = '/home/michele.maggini/dataset'
filename = 'FakeNewsCorpusSpanish.csv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path)
df = df[['Category', 'Text']]
df['Category'] = df['Category'].replace({'Fake': 1, 'True':0})


prompt_0_spec = """### Instrucción: 
Determinar si la noticia es falsa o verdadera siguiendo los pasos de razonamiento a continuación. Devolverás la explicación y el resultado de la etiqueta como un solo número entero.
El resultado será solo una etiqueta como esta: 'número entero'. Devolverás solo el resultado como un número entero como 'número entero': '0' si el artículo es verdadero, '1' si es falso.
Razonar paso a paso:

Identificar palabras y frases clave que pueden omitir información: lista de palabras o frases que podrían indicar pensamiento ilógico.
Análisis de sentimientos: Analizar el tono y el lenguaje para ver si hay palabras polarizadoras o lenguaje emocional.
¿Se utilizan dispositivos retóricos? (por ejemplo, hipérbole, preguntas cargadas)
Buscar contenido: ¿Cómo es probable que el contenido haya sido escrito por un bot?
Identificar al público objetivo: ¿Quién parece ser el público objetivo de este titular?
Información adecuada: ¿La información se presenta exhaustivamente en la noticia? ¿Es confiable?
Considerar omisiones y énfasis: ¿Qué información se enfatiza y qué podría omitirse?
El resultado es un número entero como 'integer': '0' si el artículo es verdadero, '1' si es falso.
{}

### Respuesta:
"""
sys_ans = []
temp = 0.1


for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "Eres un experto en Comunicación Política. Se te ha proporcionado una instrucción que describe una tarea y se combina con una entrada que da más contexto. Responde como se indica en la instrucción."},
      {"role": "user", "content": prompt_0_spec.format(row['Text'])}
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
filename = '{data}_sub_sys_ans_CoT.csv'
df_ans.to_csv(filename, index=False)