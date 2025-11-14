import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, PromptTuningConfig, TaskType
import optuna


class BERTModelTrainer():
    def __init__(self, variant_name):
        self.model_name = variant_name
        self.results = {} # {accuracy:val, precision:val, recall:val, f1-score:val}
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model = None
        self.tokenizer = None

    def load_data(self):
        # Read in the data from csv files
        self.train_df = pd.read_csv('train.csv')
        self.val_df = pd.read_csv('validation.csv')
        self.test_df = pd.read_csv('test.csv')
        print(f"Training: {self.train_df.shape}")
        print(f"Validation: {self.val_df.shape}")
        print(f"Test: {self.test_df.shape}")
    
    def data_preprocessing(self):
        # Maps labels ("negative" -> 0, "positive" -> 1)
        label_to_id_map = {"negative": 0, "positive": 1}
        id_to_label_map = {0: "negative", 1: "positive"}
        self.train_df["label"] = self.train_df["label"].map(label_to_id_map)
        self.val_df["label"] = self.val_df["label"].map(label_to_id_map)
        self.test_df["label"] = self.test_df["label"].map(label_to_id_map)
    
    def load_pretrained_BERT(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tokenize_function(self, text):
        # helper function
        if "prompt-finetuned" in self.model_name:
            # Adjust max_length to account for the virtual tokens preprended to the prompt
            max_seq_length = self.tokenizer.model_max_length - 20
            return self.tokenizer(text["review"], padding="max_length", truncation=True, max_length=max_seq_length)
        return self.tokenizer(text["review"], padding="max_length", truncation=True)
    
    def tokenization(self):
        # Tokenize data
        self.train_df = Dataset.from_pandas(self.train_df)
        self.val_df = Dataset.from_pandas(self.val_df)
        self.test_df = Dataset.from_pandas(self.test_df)
        self.train_df = self.train_df.map(self.tokenize_function, batched=True)
        self.val_df = self.val_df.map(self.tokenize_function, batched=True)
        self.test_df = self.test_df.map(self.tokenize_function, batched=True)

    def compute_metrics(self, pred):
        # helper function
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    def objective(self, trial):
        # helper function
        # Define the hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        weight_decay = trial.suggest_float("weight_decay", 0.001, 0.01, log=True)

        training_args = TrainingArguments(
            output_dir=self.model_name,
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=1,
            weight_decay=weight_decay,
            fp16=True,
            save_strategy="epoch",
            disable_tqdm=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_df,
            eval_dataset=self.val_df
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results["eval_accuracy"]

    def hyperparameter_finetuning(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=20)
        return study.best_params
    
    def train_model(self, best_learning_rate, best_batch_size, best_weight_decay):
        training_args = TrainingArguments(
            output_dir=self.model_name,
            eval_strategy="epoch",
            learning_rate=best_learning_rate,
            per_device_train_batch_size=best_batch_size,
            per_device_eval_batch_size=best_batch_size,
            num_train_epochs=1,
            weight_decay=best_weight_decay,
            fp16=True,
            save_strategy="epoch",
            disable_tqdm=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_df,
            eval_dataset=self.val_df
        )

        trainer.train()
        return trainer
    
    def evaluate(self, trained_model):
        preds_outputs = trained_model.predict(self.test_df)
        metrics = preds_outputs.metrics
        self.results["accuracy"] = metrics["test_accuracy"]
        self.results["precision"] = metrics["test_precision"]
        self.results["recall"] = metrics["test_recall"]
        self.results["f1-score"] = metrics["test_f1"]
        return self.results
    
    def apply_lora(self):
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank of the update matrices.
            lora_alpha=32,  # Scaling factor for the LoRA layers.
            target_modules=["query", "value"],  # Modules to apply LoRA to.
            lora_dropout=0.05,  # Dropout probability for LoRA layers.
            bias="none",  # Bias type.
            task_type=TaskType.SEQ_CLS  # Task type for sequence classification.
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def apply_prompt(self):
        # Prompt tuning configuration
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,       # Number of learnable virtual tokens
            prompt_tuning_init_text="Classify the sentiment of this text:"
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def eval_model(self):
        eval_args = TrainingArguments(
            output_dir='./eval_results',
            do_train=False,  # don't train the model
            do_eval=True,  # evaluate the model
            per_device_eval_batch_size=32,
            report_to='none'
        )

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            eval_dataset=self.test_df
        )

        return trainer


# Build BERT model 1
trainer = BERTModelTrainer("full-finetuned-bert-amazon-reviews")
trainer.load_data()
trainer.data_preprocessing()
trainer.load_pretrained_BERT()
trainer.tokenization()
best_hyperparameters = trainer.hyperparameter_finetuning()
trained_model = trainer.train_model(best_hyperparameters['learning_rate'], 
                                    best_hyperparameters['batch_size'], 
                                    best_hyperparameters['weight_decay'])
results = trainer.evaluate(trained_model)
print(f"Results for full-finetuned-bert-model:\n{results}")

# Build BERT model 2
trainer = BERTModelTrainer("lora-finetuned-bert-amazon-reviews")
trainer.load_data()
trainer.data_preprocessing()
trainer.load_pretrained_BERT()
trainer.apply_lora()
trainer.tokenization()
best_hyperparameters = trainer.hyperparameter_finetuning()
trained_model = trainer.train_model(best_hyperparameters['learning_rate'], 
                                    best_hyperparameters['batch_size'], 
                                    best_hyperparameters['weight_decay'])
results = trainer.evaluate(trained_model)
print(f"Results for lora-finetuned-bert-model:\n{results}")

# Build BERT model 3
trainer = BERTModelTrainer("prompt-finetuned-bert-amazon-reviews")
trainer.load_data()
trainer.data_preprocessing()
trainer.load_pretrained_BERT()
trainer.apply_prompt()
trainer.tokenization()
best_hyperparameters = trainer.hyperparameter_finetuning()
trained_model = trainer.train_model(best_hyperparameters['learning_rate'], 
                                    best_hyperparameters['batch_size'], 
                                    best_hyperparameters['weight_decay'])
results = trainer.evaluate(trained_model)
print(f"Results for prompt-finetuned-bert-model:\n{results}")

# Build BERT model 4
trainer = BERTModelTrainer("original-pretrained-bert-amazon-reviews")
trainer.load_data()
trainer.data_preprocessing()
trainer.load_pretrained_BERT()
trainer.tokenization()
best_hyperparameters = trainer.hyperparameter_finetuning()
trained_model = trainer.eval_model()
results = trainer.evaluate(trained_model)
print(f"Results for original-pretrained-bert-model:\n{results}")