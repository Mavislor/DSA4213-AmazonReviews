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
  - 02_data_preprocessing.ipynb
  - 03_baseline_models.ipynb      # NB & Logistic Regression
  - 04_embeddings_experiments.ipynb # GloVe, etc.
  - 05_llm_experiments.ipynb      # BERT, RoBERTa
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
  - nb_model.pkl
  - logistic_model.pkl
  - bert_model/            (folder for model & tokenizer)
- results/
  - figures/               # For plots, confusion matrices
    - nb_confusion_matrix.png
    - roberta_loss_curve.png
  - metrics/               # For saved evaluation metrics
    - baseline_metrics.json
    - llm_metrics.json
- config/
  - baseline_config.yaml
  - bert_config.yaml
- environment.yml            # Conda environment file
- requirements.txt           # Pip requirements file
- README.md                  # Project overview, how to run the code
- .gitignore
