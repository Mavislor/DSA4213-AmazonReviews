## Dataset Information

### Source
- **Original Dataset**: Amazon Reviews
- **Kaggle URL**: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
- **Original Size**: 3,600,000 reviews
- **File**: `train.ft.txt.bz2` (632MB compressed)

### Our Sample
For GitHub compatibility and faster experimentation, we're using a 20,000 review sample.

**Sample Statistics:**
- Total reviews: 20,000
- Positive reviews: 10,000 (50%)
- Negative reviews: 10,000 (50%)
- Language: English only

### File Structure

#### Raw Data (Not on GitHub - too large)
- `train.ft.txt.bz2` - Original compressed file (632MB)
  - Format: Each line: `__label__1` or `__label__2` followed by review text
  - Labels: `__label__1` = negative, `__label__2` = positive

#### Processed Data
- `data/processed/train.csv` - 14,000 reviews for training
- `data/processed/validation.csv` - 3,000 reviews for validation
- `data/processed/test.csv` - 3,000 reviews for testing

**CSV Format:**
```csv
label,review
positive,"This product is amazing..."
negative,"Terrible quality, would not recommend..."
