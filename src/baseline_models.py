import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from typing import Tuple, Dict, Any


class BaselineModelTrainer:
    def __init__(self, random_state: int = 4213):
        self.random_state = random_state
        self.tfidf = None
        self.models = {}
        self.results = {}
        
    def initialize_tfidf(self, 
                        max_features: int = 10000,
                        min_df: int = 5,
                        max_df: float = 0.7,
                        ngram_range: Tuple[int, int] = (1, 2),
                        stop_words: str = 'english') -> None:
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True
        )
        
    def initialize_models(self) -> None:
        self.models = {
            'Naive_Bayes': MultinomialNB(),
            'Logistic_Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            )
        }
    
    def load_data(self, 
                  train_path: str, 
                  val_path: str, 
                  test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv('train.csv')
        val_df = pd.read_csv('validation.csv')
        test_df = pd.read_csv('test.csv')
        
        print(f"  Training: {train_df.shape}")
        print(f"  Validation: {val_df.shape}")
        print(f"  Test: {test_df.shape}")
            
        return train_df, val_df, test_df
    
    def prepare_features(self, 
                        train_df: pd.DataFrame, 
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame = None) -> Tuple:
        
        if self.tfidf is None:
            self.initialize_tfidf()
            
        X_train = train_df['review']
        y_train = train_df['label']
        X_val = val_df['review']
        y_val = val_df['label']
        
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_val_tfidf = self.tfidf.transform(X_val)
        
        if test_df is not None:
            X_test = test_df['review']
            y_test = test_df['label']
            X_test_tfidf = self.tfidf.transform(X_test)
        else:
            X_test_tfidf, y_test = None, None
            
        print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
        print(f"Training features shape: {X_train_tfidf.shape}")
        print(f"Validation features shape: {X_val_tfidf.shape}")
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test
    
    def train_models(self, 
                    X_train_tfidf, 
                    y_train, 
                    X_val_tfidf, 
                    y_val) -> Dict[str, Dict[str, Any]]:
       
        if not self.models:
            self.initialize_models()
            
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model
            model.fit(X_train_tfidf, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
        return self.results
    
    def save_models(self, models_dir: str = '../models') -> None:
        os.makedirs(models_dir, exist_ok=True)
        
        if self.tfidf:
            joblib.dump(self.tfidf, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        
        for name, result in self.results.items():
            model = result['model']
            # Convert name to filename-friendly format
            filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, os.path.join(models_dir, filename))
        
        print(f"Models saved to: {models_dir}")
    
    def save_results(self, results_dir: str = '../results/metrics') -> None:
        os.makedirs(results_dir, exist_ok=True)
        
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(results_dir, 'baseline_model_comparison.csv'), index=False)
        
        print(f"Results saved to: {results_dir}")


# Function for quick usage
def run_baseline_experiment(train_path: str, 
                          val_path: str, 
                          test_path: str = None,
                          random_state: int = 4213) -> BaselineModelTrainer:
   
    trainer = BaselineModelTrainer(random_state=random_state)
    
    train_df, val_df, test_df = trainer.load_data(train_path, val_path, test_path)
    
    X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )
  
    trainer.train_models(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    return trainer
