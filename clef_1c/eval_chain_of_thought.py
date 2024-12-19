from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset, Features, Value
import argparse
import torch
import time
import json
import re
import wandb
from utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Randomized Few-Shot Eval script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument("--language", type=str, default="English")
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)


main_run = wandb.init(
    project="CLEF2022task1C",
    entity="michelej-m",
    name=f"{args.model_name.split('/')[1]}_CoT",
)
main_run.log({"num_runs": 5})
main_run.log({"language": args.language})

# ---- Model/Tokenizer loading
if args.use_quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
print(model.generation_config)

# ---- Dataset loading/Processing
features = Features(
    {
        "tweet_text": Value("string"),
        "class_label": Value("int64"),
    }
)

data_files = {
    "train": f"./data/CT22_{args.language.lower()}_1C_harmful_train.tsv",
    "test": f"./data/CT22_{args.language.lower()}_1C_harmful_test_gold.tsv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", features=features)
dataset = dataset.rename_column("class_label", "label")
dataset = dataset.rename_column("tweet_text", "text")
print(dataset)

# ---- Inference utils
prompts = {
    "english": {
        "system": "You have received an instruction that describes a task and it has been combined with an input that provides more context. Respond as directed in the instruction.",
"user":"""###Instruction: Given a tweet, predict if it is harmful to society. If it is harmful, label it as '1', if it is not harmful, label it as '0'. The output label format is 'integer'. Separate the reasoning steps with <end>. Always include the final prediction in the output, following the template given in the example.
# Step-by-step reasoning: Identify keywords and phrases: Make a list of words or phrases that may contain harmful information and intent.
# Sentiment analysis: Analyze the tone and language to see if there is unintentional or intentional harmful behavior, words, or emotional language.
# Identify the target audience: Who seems to be the target audience for this tweet?
# Threats and harassment: Direct or veiled threats of violence, physical harm, or harassment. This includes revealing personal information (doxing), stalking, or inciting others to commit harassment.
# Hostile speech: Expressions of hatred, intolerance or prejudice against specific groups. This can manifest as dehumanizing language, calls for violence or promotion of discriminatory ideologies.
# Disinformation and false information: The spreading of false or misleading information, often with the intention to mislead or manipulate. This can include conspiracy theories, fake news or manipulated media.
# Cyberbullying: Aggressive or deliberate behavior using electronic communications to harm, humiliate or threaten an individual. # final prediction:

Example output:
# Step-by-step reasoning: EXPLANATION <end>
# Sentiment analysis: EXPLANATION <end>
# Target audience identification: EXPLANATION <end>
# Threats and intimidation: EXPLANATION <end>
# Hostile language: EXPLANATION <end>
# Disinformation and false information: EXPLANATION<end>
#Cyberbullying: EXPLANATION <end>
# final prediction: integer prediction <end>

###Input: {}

###Response:""",    },
    "bulgarian": {
        "system": "Получихте инструкция, която описва задача и тя е комбинирана с въвеждане, което предоставя повече контекст. Отговорете, както е указано в инструкцията.",
        "user": """###Инструкция: Имайки даден туит, познайте дали е вреден за обществото. Ако е вреден, маркирайте го като „1“, ако не е вреден, маркирайте го като „0“. Форматът на изходния етикет е „цяло число“. Разделете стъпките на разсъждение с <end>. Винаги включвайте крайната прогноза в изхода, като следвате шаблона, даден в примера.
# Разсъждение стъпка по стъпка: Идентифицирайте ключови думи и фрази: Направете списък с думи или фрази, които може да съдържат вредна информация и намерение.
# Анализ на настроението: Анализирайте тона и езика, за да видите дали има неволно или умишлено вредно поведение, думи или емоционален език.
# Идентифицирайте целевата аудитория: Коя изглежда е целевата аудитория за този туит?
# Заплахи и тормоз: Преки или завоалирани заплахи за насилие, физическо увреждане или тормоз. Това включва разкриване на лична информация (доксиране), преследване или подбуждане на други към извършване на тормоз.
# Враждебна реч: Изрази на омраза, нетолерантност или предразсъдъци срещу определени групи. Това може да се прояви като дехуманизиращ език, призиви за насилие или насърчаване на дискриминационни идеологии.
# Дезинформация и невярна информация: Разпространението на невярна или подвеждаща информация, често с намерение да подведе или манипулира. Това може да включва теории на конспирацията, фалшиви новини или манипулирани медии.
# Кибертормоз: Агресивно или умишлено поведение, използващо електронни комуникации, за да нарани, унижи или заплаши дадено лице. # финална прогноза:

Примерен резултат:
# Разсъждение стъпка по стъпка: ОБЯСНЕНИЕ <край>
# Анализ на настроението: ОБЯСНЕНИЕ <край>
# Идентификация на целевата аудитория: ОБЯСНЕНИЕ <край>
# Заплахи и сплашване: ОБЯСНЕНИЕ <край>
# Враждебен език: ОБЯСНЕНИЕ <край>
# Дезинформация и невярна информация: ОБЯСНЕНИЕ<end>
#Кибертормоз: ОБЯСНЕНИЕ <край>
# финална прогноза: целочислена прогноза <край>

###Вход: {}

###Отговор:"""
    },
    "arabic": {
        "system": "لقد تلقيت تعليمات تصف مهمة وتم دمجها مع مدخلات توفر سياقًا أكثر. استجب وفقًا للتوجيهات الواردة في التعليمات.",
"user":"""###التعليمات: إذا تلقيت تغريدة، توقع ما إذا كانت ضارة بالمجتمع. إذا كانت ضارة، ضع علامة عليها بـ "1"، وإذا لم تكن ضارة، ضع علامة عليها بـ "0". تنسيق العلامة الناتج هو "عدد صحيح". افصل خطوات الاستدلال بعلامة <end>. قم دائمًا بتضمين التنبؤ النهائي في الناتج، باتباع النموذج الوارد في المثال.
# الاستدلال خطوة بخطوة: حدد الكلمات والعبارات الرئيسية: قم بعمل قائمة بالكلمات أو العبارات التي قد تحتوي على معلومات ضارة ونوايا ضارة.
# تحليل المشاعر: قم بتحليل النبرة واللغة لمعرفة ما إذا كان هناك سلوك أو كلمات أو لغة عاطفية ضارة غير مقصودة أو مقصودة.
# حدد الجمهور المستهدف: من يبدو أنه الجمهور المستهدف لهذه التغريدة؟
# التهديدات والمضايقات: التهديدات المباشرة أو المبطنة بالعنف أو الأذى الجسدي أو المضايقة. ويشمل ذلك الكشف عن المعلومات الشخصية (التشهير) أو الملاحقة أو تحريض الآخرين على ارتكاب المضايقات.
# الخطاب العدائي: تعبيرات الكراهية أو التعصب أو التحيز ضد # التضليل والمعلومات الكاذبة: نشر معلومات كاذبة أو مضللة، غالبًا بقصد التضليل أو التلاعب. يمكن أن يشمل ذلك نظريات المؤامرة أو الأخبار المزيفة أو وسائل الإعلام التي تم التلاعب بها.
# التنمر الإلكتروني: سلوك عدواني أو متعمد باستخدام الاتصالات الإلكترونية لإيذاء أو إذلال أو تهديد فرد ما. # التنبؤ النهائي:

مثال على الناتج:
# الاستدلال خطوة بخطوة: الشرح <end>
# تحليل المشاعر: الشرح <end>
# تحديد الجمهور المستهدف: الشرح <end>
# التهديدات والترهيب: الشرح <end>
# اللغة العدائية: الشرح <end>
# التضليل والمعلومات الكاذبة: الشرح <end>
# التنمر الإلكتروني: الشرح <end>
# التنبؤ بالأعداد الصحيحة <end>

###الإدخال: {}

###الاستجابة:"""},}

prompt = prompts[args.language.lower()]["user"]
system_prompt = prompts[args.language.lower()]["system"]


def parse_label(model_output):
    match = re.search(r"\b[0-1]\b", model_output)
    return int(match.group()) if match else None


def generate(model, tokenizer, prompt, element, temperature=0.1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.format(element)},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=512, temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


# ---- Inference
dataset_labels = list(set(dataset["train"]["label"]))

start_time = time.time()
irregular_outputs = 0
preds = []
refs = []

for element in dataset["test"]:
    pred = generate(model, tokenizer, prompt, element["text"])

    args.verbose and print("-" * 20)
    args.verbose and print(" > Pred: ", pred)
    args.verbose and print(" > Parsed Pred: ", parse_label(pred))
    args.verbose and print(" > Ref: ", element["label"])
    args.verbose and print("-" * 20)

    if parse_label(pred) is None:
        print(" > Irregular output:  ", pred)

        print("*" * 5, "Trying to resolve irregularity", "*" * 5)
        while True:
            pred = generate(
                model,
                tokenizer,
                prompt,
                element["text"],
                temperature=0.7,
            )
            print(" >> Attempted Pred: ", pred)

            if parse_label(pred) is not None:
                print(" >> Regularized output: ", pred)
                break
        irregular_outputs += 1
        continue

    preds.append(parse_label(pred))
    refs.append(element["label"])

results = compute_metrics(preds, refs)
results["irregular_outputs"] = irregular_outputs
model_outputs = {
    "ground_truth": refs,
    "model_predictions": preds,
}
print(json.dumps(results, indent=4))

print(f" > Inference execution time: {(time.time() - start_time):.2f}s")

for metric in results:
    main_run.log({f"avg_{metric}": results[metric]})

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
main_run.save("results.json")

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)
main_run.save("model_outputs.json")

main_run.finish()
