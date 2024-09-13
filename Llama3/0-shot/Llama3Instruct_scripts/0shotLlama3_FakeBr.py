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
data = "Fake.br"
start_time = time.time()
configuration = "10-shot"
n_example = 1000
temp = 0.1
sys_ans = []

tsv_directory = ''
filename = 'Fake.br-Corpus.tsv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path, sep="\t", engine="python")
df = df.groupby('label').head(n_example)
df['label'] = df['label'].map({'true': 0, 'fake': 1})
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
Dada uma manchete, determine se é uma notícia real ou falsa.
A saída consiste apenas num número inteiro com este formato: 'inteiro'. Se o título da notícia for falso, o resultado será '1', se for verdadeiro, '0'.
Vou fornecer alguns exemplos para você aprender as características das notícias falsas.
#example: jornalista encurrala rodrigo maia vai dar golpe temer assim fez dilma jornalista roberto d avila entrevistou presidente camara deputados rodrigo maia inicio entrevista apresentador perguntou maia voce fara michel temer fez dilma perguntou presidente camara respondeu mae passou mensagem perguntou conspirando disse nao mostrei mensagem presidente temer, '1'
#example: lula desembarca rj hostilizado populacao hospede surpreende lula dentro hotel ai ladrao sob gritos lula ladrao lugar prisao desembarcou aeroporto rio janeiro ultima petista nao incomodou elogios ainda colocou sorriso amarelado cara pois lula queria justica ruas tendo leia tambem, '1'
#example: magnata ecologista financia campanha contra trump tom steyer ira destinar us milhoes ajudar democratas tentar maioria congresso eleicoes deste ano tambem financia campanha impeachment trump ja apoio milhoes pessoas magnata ecologista americano tom steyer anunciou nesta destinara us milhoes ajudar democratas recuperar maioria camara representantes eleicoes novembro continuara financiando campanha visando impeachment presidente donald trump steyer revelou ainda nao sera candidato legislativas deste ano omitiu planeja concorrer presidencia magnata informou reforcara apoio campanha internet destituir presidente republicano ja conta quatro milhoes pessoas objetivo donald trump destituido declarou steyer durante entrevista coletiva washington novembro sera renovada totalidade camara representantes terco senado ambos dominados republicanos tom steyer planeja destinar us milhoes organizacao nextgen rising procura mobilizar eleitorado jovem decisivo segundo disse definir proximas eleicoes sera ano luta alma deste pais anos steyer antigo proprietario fundos especulativos fortuna avaliada us bilhao segundo revista forbes, '0'
#example: ceara cid gomes pdt afirmou discurso ontem sobral festa aniversario envolvido esquema corrupcao petrobras ministro stf teori zavascki relata lava jato corno republica rodrigo janot ladrao juiz sergio moro atua investigacoes primeira instancia picareta nome cid gomes aparece lista apreendida lava jato sede odebrecht sob investigacao stf cid aparece lado valor r mil codinome falso ouca audio, '0'
#example: ouca disse general brigada exercito ultimo dia interesse governo nao saibamos situacao ja caos acompanhem sequencia declaracoes bombasticas talvez dificeis aceitar inicio essenciais evolucao exito causa patriotica, '1'
#example: deputado federal afirma plenario lula maior ladrao pais chefe organizacao criminosa, '1'
#example: artistas nordestinos ficam indignados declaracao lula rio grande sul confira video, '1'
#example: eduardo cunha ganha apelido agentes pf angelica, '1'
#example: suplicy participara programa doria web nesta quinta vereador eduardo suplicy participara nesta programa olho olho transmitido redes sociais prefeito joao doria doria ja recebeu quadro aliados personalidades cantor lobao apresentador jose luiz datena basquete oscar cantor roger ultraje rigor jornalista joice hasselman comeco gestao tucano poupou criticas antecessor fernando haddad ultimos meses passou acusar deixar rombo r bilhoes prefeitura haddad nega diz deixou contas cidade ordem suplicy vez vozes criticas doria camara vereadores, '0'

###Input:título da notícia:{}

### Response:
"""



for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "É um especialista em Comunicação Política. Recebeu uma instrução que descreve uma tarefa e é combinada com uma entrada que fornece mais contexto. Responda conforme indicado nas instruções."},
      {"role": "user", "content": prompt.format(row['text'])}
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
print(sys_ans)
gold_ans = df['label'].tolist()
gold_ans = list(map(int, gold_ans))



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
print(f"{data}, {configuration}, prompt with temperature={temp},  n_labels={df.shape}:\n", json.dumps(results, indent=4))
# Compute precision, recall, and f1-score
print(f"{data}, {configuration}, prompt with temperazture={temp}, n_labels={df.shape}:\n", classification_report(gold_ans,sys_ans))

print("My program took", time.time() - start_time, "to run")
