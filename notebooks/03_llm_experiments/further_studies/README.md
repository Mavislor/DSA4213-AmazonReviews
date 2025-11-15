## Sentiment analysis with sarcasm detection
Adversarial testing on our finetuned BERT models revealed that the models were unable to correctly predict the sentiment of sarcastic reviews.
In an effort to improve the performance of the finetuned BERT model for sentiment analysis, we finetune a separate BERT model for sarcasm detection and 
then integrate its predictions for sarcasm into the predictions for sentiments made by the BERT model for sentiment analysis.

The rationale is:
If a review is predicted to be sarcastic, we will invert the sentiment prediction.
