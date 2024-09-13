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
import re
import time
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json

random.seed(42)

start_time = time.time()
configuration = "10-shot"
#n_example = 5
temp = 0.1
sys_ans = []

tsv_directory = ''
filename = 'sub_allsides.csv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path)
#df = df.groupby('bias_rating').head(n_example)

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

device = torch.device('cuda:1')
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map=device, 
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.float16)


prompt = """
### Instruction:
Detect the political leaning of the article. 
You'll return only the output as an integer like 'integer': '0' if the article is left-wing, '1' if the political leaning is from the center, '2' if the article is right-wing'.
Pay attention to right-wing because you are missing them.
I'll provide you with some examples of a headline and its label 'label', so you can learn features from it.
#Example: 'Vaccination foes on the defensive The Disneyland measles outbreak has forced a reckoning on the politics of vaccination: Likely GOP presidential candidates are stumbling over the issue, President Barack Obama has forcefully weighed in, and several states are pushing to make it harder to exempt children from vaccinations.' '0'
#Example: "Democrats Keep Control of Senate as Red Wave Fails to MaterializeDemocrats kept control of the Senate on Saturday after Catherine Cortez Masto won re-election in Nevada, repelling Republican efforts to retake the chamber.\nThe Associated Press called Nevada for Cortez Masto on Saturday, projecting that Republican opponent Adam Laxalt—backed by former president Donald Trump—could not make enough gains to win in one of the most competitive races of the midterm elections.\nMasto's victory allows Democrats to retain control of the Senate with at least 50 seats because of Vice President Kamala Harris' tiebreaking vote, making it harder for Republicans to thwart President Joe Biden's agenda even if Republicans gain...", '1'
#Example: "After serving as a foreign policy advisor for many politicians, Madeleine Albright's first major role was as the United States' ambassador to the United Nations. President Bill Clinton nominated her for the role in 1993.", '2'
#Example: 'The House narrowly approved a sweeping spending bill Thursday night despite deep misgivings among liberals and conservatives alike, sending the measure to the Senate as lawmakers averted a partial government shutdown.', '2'
#Example: 'The spiritual leader of 1.2 billion Catholics, Pope Benedict XVI, surprised the world Monday by saying he will resign at the end of the month "because of advanced age."', '0'
#Example:'The number of Americans who filed for unemployment benefits last week was roughly in line with expectations as the coronavirus pandemic continues to ravage the U.S. economy.\nData released by the Labor Department showed initial weekly jobless claims for the week ending July 25 came in at 1.434 million. Economists polled by Dow Jones expected claims to rise to 1.45 million.\nThis also marks the second consecutive week in which initial claims rose after declining for 15 straight weeks. It is also the 19th straight week in which initial...', '1'
#Example: 'CNN Thursday turned the important battleground state of Wisconsin from "lean Obama" to true "toss up" on its electoral map, in the wake of Mitt Romney\'s naming of House Budget Chairman Paul Ryan, a seven term congressman from the Badger state, as his running mate.', '0'
#Example: 'This was the week of consumer angst. Inflation is pushing prices up, while the stock market started the week trending down.\n“The economy is going through some dramatic adjustments as it opens up following the unprecedented circumstances associated with the pandemic last year,” said Mark Hamrick, senior economic analyst for Bankrate.\nThen there’s the panic over gas.\n“Top off your cars,” a friend texted this week. “Costco and BJ lines are extremely long this morning and smaller stations are running out of gas.”\nIt’s the toilet paper sellout all over...', '0'
#Example: "Democrats kept control of the Senate on Saturday after Catherine Cortez Masto won re-election in Nevada, repelling Republican efforts to retake the chamber.\nThe Associated Press called Nevada for Cortez Masto on Saturday, projecting that Republican opponent Adam Laxalt—backed by former president Donald Trump—could not make enough gains to win in one of the most competitive races of the midterm elections.\nMasto's victory allows Democrats to retain control of the Senate with at least 50 seats because of Vice President Kamala Harris' tiebreaking vote, making it harder for Republicans to thwart President Joe Biden's agenda even if Republicans gain...", '1'
#Example: 'While it does not go as far as proponents would like, the bill is the most sweeping prison reform agreement that Congress has passed in decades.', '0'

### Input:
{}

### Response:
"""



for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "You are an expert in Political Communication Science. You have been provided with an instruction describing a task and it is combined with an input giving more context. Reply as indicated in the instruction."},
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

sys_ans = [1 if item == '1' else 2 if item == '2' else 0 for item in sys_ans]

gold_ans = df['bias_rating'].tolist()





print("gold_ans:", gold_ans)
print("list_answer:", sys_ans)



import evaluate 

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
