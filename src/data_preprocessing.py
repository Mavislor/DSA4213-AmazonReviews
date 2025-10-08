import bz2
import pandas as pd
from langdetect import detect, LangDetectException, DetectorFactory
from sklearn.model_selection import train_test_split
import os

RANDOM_SEED = 4213

def is_english(review):
    try:
        return detect(review) == 'en'
    except LangDetectException:
        return False

def main():
    print("Starting data preprocessing...")
    
    # Load and clean the data
    file_path = "train_small.ft.txt.bz2"  
    print("Loading data from compressed file...")
    
    data = []
    with bz2.open(file_path, "rt", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:  # Progress indicator
                print(f"Processed {i} lines...")
            label, text = line.strip().split(" ", 1)
            data.append((label, text))

    df = pd.DataFrame(data, columns=["label", "review"])
    df["label"] = df["label"].map({"__label__1": "negative", "__label__2": "positive"})
    print(f"Original number of reviews: {len(df)}")

    # Filter English reviews
    print("Filtering English reviews...")
    DetectorFactory.seed = 0
    df['is_english'] = df['review'].apply(is_english)
    english_df = df[df['is_english']].drop(columns=['is_english'])
    print(f"Number of English reviews: {len(english_df)}")

    # Take a sample for faster experimentation
    print("Sampling data...")
    df_sample = english_df
    print(f"Working sample size: {len(df_sample)}")

    # Check class distribution
    print("\nClass distribution in sample:")
    print(df_sample['label'].value_counts())

    # Split into train (70%), validation (15%), and test (15%)
    print("Splitting data...")
    train_val_df, test_df = train_test_split(
        df_sample, 
        test_size=0.15, 
        random_state=RANDOM_SEED, 
        stratify=df_sample['label']
    )

    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.176,  # 0.15 / 0.85 ≈ 0.176
        random_state=RANDOM_SEED, 
        stratify=train_val_df['label']
    )

    print(f"\nFinal dataset sizes:")
    print(f"Training set: {len(train_df)} reviews")
    print(f"Validation set: {len(val_df)} reviews")
    print(f"Test set: {len(test_df)} reviews")

    os.makedirs('data/processed', exist_ok=True)

    # Save the splits
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/validation.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print("Files created:")
    print("  - data/processed/train.csv")
    print("  - data/processed/validation.csv") 
    print("  - data/processed/test.csv")

if __name__ == "__main__":
    main()
