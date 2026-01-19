import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your data
df = pd.read_excel(r"C:\Users\User\Documents\Dissertation\BigQuery\BroadMkt\Scraped.xlsx")


# Create text for analysis - use scraped_full_text if available, otherwise headline
def get_text_for_sentiment(row):
    if pd.notna(row['scraped_full_text']) and row['scraped_full_text'].strip():
        return row['scraped_full_text']
    elif pd.notna(row['headline']) and row['headline'].strip():
        return row['headline']
    else:
        return ""


df['analysis_text'] = df.apply(get_text_for_sentiment, axis=1)

# Load FinBERT model
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)
model.eval()
print("Model loaded successfully")


def complete_finbert_analysis(text):
    """
    Complete FinBERT analysis with all metrics
    """
    if not text or not text.strip():
        return {
            'positive_prob': 0.33,
            'negative_prob': 0.33,
            'neutral_prob': 0.34,
            'predicted_label': 'neutral',
            'confidence': 0.34,
            'sentiment_score': 0.0,
            'polarity_strength': 0.66,
            'uncertainty': 1.099  # High uncertainty for empty text
        }

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Move to CPU and convert to numpy
    probs = probabilities.cpu().numpy()

    # Extract probabilities
    positive_prob = float(probs[0])
    negative_prob = float(probs[1])
    neutral_prob = float(probs[2])

    # Predicted label
    labels = ['positive', 'negative', 'neutral']
    predicted_label = labels[np.argmax(probs)]

    # Confidence (highest probability)
    confidence = float(np.max(probs))

    # Sentiment score (positive - negative)
    sentiment_score = positive_prob - negative_prob

    # Polarity strength (1 - neutral)
    polarity_strength = 1 - neutral_prob

    # Uncertainty (entropy)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    uncertainty = -sum(p * np.log(p + epsilon) for p in probs)

    return {
        'positive_prob': positive_prob,
        'negative_prob': negative_prob,
        'neutral_prob': neutral_prob,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'sentiment_score': sentiment_score,
        'polarity_strength': polarity_strength,
        'uncertainty': float(uncertainty)
    }


# Process all texts with progress tracking
print("Processing sentiment analysis...")
print(f"Total texts to process: {len(df)}")

results = []
for i, text in enumerate(df['analysis_text']):
    if i % 50 == 0:
        print(f"Processed {i}/{len(df)} texts ({i / len(df) * 100:.1f}%)")

    result = complete_finbert_analysis(text)
    results.append(result)

print("Processing complete!")

# Add all new columns to dataframe
df['positive_prob'] = [r['positive_prob'] for r in results]
df['negative_prob'] = [r['negative_prob'] for r in results]
df['neutral_prob'] = [r['neutral_prob'] for r in results]
df['predicted_label'] = [r['predicted_label'] for r in results]
df['confidence'] = [r['confidence'] for r in results]
df['sentiment_score'] = [r['sentiment_score'] for r in results]
df['polarity_strength'] = [r['polarity_strength'] for r in results]
df['uncertainty'] = [r['uncertainty'] for r in results]

# Save results
output_path = r"C:\Users\User\Documents\Dissertation\BigQuery\BroadMkt\Complete_FinBERT_Analysis.xlsx"
df.to_excel(output_path, index=False)

print(f"Complete analysis saved to: {output_path}")

# Display sample results
print("\n=== SAMPLE RESULTS ===")
sample_cols = ['headline', 'positive_prob', 'negative_prob', 'neutral_prob',
               'predicted_label', 'confidence', 'sentiment_score', 'polarity_strength', 'uncertainty']

# Show first 5 rows
for i in range(min(5, len(df))):
    row = df.iloc[i]
    print(f"\nSample {i + 1}: {row['headline'][:60]}...")
    print(
        f"  Probabilities: Pos={row['positive_prob']:.3f}, Neg={row['negative_prob']:.3f}, Neu={row['neutral_prob']:.3f}")
    print(f"  Predicted: {row['predicted_label']} (confidence: {row['confidence']:.3f})")
    print(f"  Sentiment Score: {row['sentiment_score']:.3f}")
    print(f"  Polarity Strength: {row['polarity_strength']:.3f}")
    print(f"  Uncertainty: {row['uncertainty']:.3f}")

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Total processed: {len(df)} rows")
print(f"Using scraped text: {sum(df['scraped_full_text'].notna())} rows")
print(f"Using headlines: {sum(df['scraped_full_text'].isna() & df['headline'].notna())} rows")

print(f"\nSentiment Distribution:")
print(
    f"Positive: {sum(df['predicted_label'] == 'positive')} ({sum(df['predicted_label'] == 'positive') / len(df) * 100:.1f}%)")
print(
    f"Negative: {sum(df['predicted_label'] == 'negative')} ({sum(df['predicted_label'] == 'negative') / len(df) * 100:.1f}%)")
print(
    f"Neutral: {sum(df['predicted_label'] == 'neutral')} ({sum(df['predicted_label'] == 'neutral') / len(df) * 100:.1f}%)")

print(f"\nSentiment Score Range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")
print(f"Average Sentiment Score: {df['sentiment_score'].mean():.3f}")
print(f"Average Confidence: {df['confidence'].mean():.3f}")
print(f"Average Polarity Strength: {df['polarity_strength'].mean():.3f}")
print(f"Average Uncertainty: {df['uncertainty'].mean():.3f}")

# High confidence predictions
high_conf = df[df['confidence'] >= 0.8]
print(f"\nHigh Confidence Predictions (≥80%): {len(high_conf)} ({len(high_conf) / len(df) * 100:.1f}%)")

# Clear GPU memory if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")