# DSA4213-AmazonReviews
DSA4213 final project proposal (group 43)

[Amazon Reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)


## Repository Structure
- data/
   - raw/
   - processed/             # The final, cleaned data for modeling
      - train.csv
      - test.csv
      - validation.csv     
    - README.md            # Description of the data, source, and any cleaning steps
- notebooks/
  - 01_data_exploration.ipynb
  - 02_baseline_models.ipynb      # NB & Logistic Regression
  - 03_embeddings_experiments.ipynb # GloVe, etc.
  - 04_llm_experiments.ipynb      # BERT, RoBERTa
- src/                       
  - __init__.py            
  - data_preprocessing.py
  - feature_engineering.py
  - baseline_models.py
  - llm_models.py
    - train.py
    - evaluate.py
    - utils.py
- models/                    
  - naive_bayes_model.pkl
  - logistic_regression_model.pkl
  - bert_model/            (folder for model & tokenizer)
- results/
  - metrics/               
    - baseline_model_comparison.csv
    - llm_metrics.json
- config/
  - baseline_config.yaml
  - bert_config.yaml
- requirements.txt           # Pip requirements file
- README.md                  # Project overview, how to run the code
- .gitignore
