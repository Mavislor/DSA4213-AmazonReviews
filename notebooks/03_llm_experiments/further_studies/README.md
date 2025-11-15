## Sentiment analysis with sarcasm detection
### Objective
Adversarial testing on our finetuned BERT models revealed that the models were unable to correctly predict the sentiment of sarcastic reviews.
In an effort to improve the performance of the finetuned BERT model for sentiment analysis, we finetune a separate BERT model for sarcasm detection and then integrate its predictions for sarcasm into the predictions for sentiments made by the BERT model for sentiment analysis.

### Process
1. As our original dataset on Amazon reviews is not labeled for sarcasm, there is a need to use a new dataset to finetune the separate BERT model for sarcasm detection. We will use the `Sarcasm_Headlines_Datset.json` dataset (obtained from [here](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data)) for this purpose. This is done under `01_sarcasm_BERT.ipynb`.
2. Upon fully finetuning the BERT model for sarcasm detection, we load the two BERT models in `02_sentiment_analysis_with_sarcasm_detection.ipynb`.
3. In `02_sentiment_analysis_with_sarcasm_detection.ipynb`, we first obtain the sentiment predictions for the Amazon reviews test data using the BERT model trained for sentiment analysis. Then we feed the Amazon reviews test data into the BERT model trained for sarcasm detection to get the sarcasm predictions.
4. After obtaining the two sets of predictions for the Amazon reviews test data, we use the sarcasm predictions to edit the sentiment predictions.
   **_The rationale is as such: If a review is predicted to be sarcastic, we will invert the sentiment prediction._**
5. Finally, we evaluate and compare the accuracy of the edited predictions with the original predictions for sentiment analysis.
