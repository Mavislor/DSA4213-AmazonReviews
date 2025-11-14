## BERT models

This folder contains the finetuning code for the 4 variants of the BERT model and the evaluation results on the test data.
- `01_full_finetuned_BERT_Amazon.ipynb` contains a BERT model that has been fully finetuned
- `02_LORA_finetuned_BERT_Amazon.ipynb` contains a BERT model that has been LORA finetuned
- `03_prompt_finetuned_BERT_Amazon.ipynb` contains a BERT model that has been prompt finetuned
- `04_pretrained_BERT_Amazon.ipynb` contains the original BERT model as a baseline for comparison
- `05_BERT_variant_evaluation.ipynb` contains the performance of the 4 BERT models quantitatively and qualitatively (error analysis & adversarial testing)
  
Hyperparamter tuning for `learning_rate`, `batch_size` and `weight_decay` was performed on the fully finetuned BERT model. The best set of hyperparameters obtained for the fully finetuned BERT model was also used for the subsequent BERT models for consistency.
