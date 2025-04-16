### Comparative Analysis of Transformer Architectures and Learning Techniques for Multilingual Political and Fake News Classification

#### Supervised fine-tuning experiment
This experiment focuses on using supervised learning to fine-tune LoRA (Low-Rank Adaptation) adapters for sequence classification across various datasets. We conducted all our tests on a single A100 80GB SXM GPU.

##### Models tested:
| Model name | Model Architecture | Huggingface ID | Note |
|---|---|---|---|
| Llama3-8B | Decoder |  meta-llama/Meta-Llama-3-8B  | Access request required | 
| Llama3-8B-instruct | Decoder |  meta-llama/Meta-Llama-3-8B-Instruct  | Access request required |
| Llama-3.1-8B | Decoder |  meta-llama/Meta-Llama-3.1-8B  | Access request required |
| Phi-2 | Decoder |  microsoft/phi-2  | - |
| POLITICS | Encoder |  launch/POLITICS | - |
| RoBERTa-base | Encoder |  FacebookAI/roberta-base  | - |
| RoBERTa-large | Encoder |  FacebookAI/roberta-large  | - |
| mlx-RoBERTa-base | Encoder | FacebookAI/xlm-roberta-base  | - | 
| mdeBERTa-base | Encoder | microsoft/mdeberta-v3-base  | - | 


##### Result reproduction
To reproduce our findings for any of the datasets, follow these steps:
1. Navigate to the dataset's folder
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the training script: 
    - For encoder models: `python train_encoder_seq_cls.py --model_name MODEL --epochs 3 --runs 5` 
    - For decoder models: `python train_decoder_seq_cls.py --model_name MODEL --epochs 3 --runs 5`

Where MODEL is the model's id on huggingface, described in the table above.

**Note**: for the Clef 1C dataset: you have to also add the "--language LANGUAGE" flag when executing the training scripts. If it's not included, it defaults to English. The languages are: English, Dutch, Bulgarian, Arabic.  
_Example:_ `python train_encoder_seq_cls.py --model_name MODEL --language Bulgarian --epochs 3 --runs 5` 



#### ICL experiment
This experiment investigates the effectiveness of prompts with varying levels of reasoning complexity, utilizing Few-shot learning and Chain-of-Thought (CoT) approaches. The prompts, detailed in the Appendix tables, range from simple to more elaborate reasoning structures. To conduct this experiment, we employed a computing infrastructure consisting of two Tesla P40 GPUs and one NVIDIA GeForce RTX 2080 Ti GPU.

##### Models tested:
| Model name | Model Architecture | Huggingface ID | Note |
|---|---|---|---|
| Llama3-8B-instruct | Decoder |  meta-llama/Meta-Llama-3-8B-Instruct  | Access request required |


##### Result reproduction

To reproduce our findings for any of the datasets, follow these steps:
1. Open the project directory and navigate to either:
The "COT" folder, or
The "Llama3/0-shot/Llama3Instruct_scripts" directory
2. Locate and open the configuration file (likely named something like "config.py" or "settings.py")
3. Find the variable named "tsv_directory" and update its value to the full path where you've downloaded the datasets. For example:tsv_directory = "/home/user/projects/datasets"
4. In the same configuration file, find the section for prompt instructions. Modify these instructions to match the In-Context Learning (ICL) templates provided in the paper's appendix. Ensure you're using the correct template for your chosen method (COT or 0-shot).

   
