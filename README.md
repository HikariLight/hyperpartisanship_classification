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
This experiment focuses on using different prompts (see tables in the Appendix of the paper) with Few-shots and CoT. The computing infrastructure for this experiment included two Tesla P40 GPUs, one NVIDIA GeForce RTX 2080 Ti GPU.

##### Model tested:
| Model name | Model Architecture | Huggingface ID | Note |
| Llama3-8B-instruct | Decoder |  meta-llama/Meta-Llama-3-8B-Instruct  | Access request required |


#####Result reproduction
To reproduce our findings for any of the datasets, follow these steps:
1. Navigate to the COT or Llama3/0-shot/Llama3Instruct_scripts
2. Change the tsv_directory to where datasets have been downloaded
3. Change the Instructions following the ICL templates given in the paper
   
