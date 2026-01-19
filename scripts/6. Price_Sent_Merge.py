import pandas as pd
import numpy as np

# --- Configuration ---
SENTIMENT_FILE_PATH = '/Users/hrishikeshsajeev/Dissertation codes/FInal FIles BERT/Sent daily.csv'
PRICE_DATA_FILE_PATH = '/Users/hrishikeshsajeev/Dissertation codes/FInal FIles BERT/gspc_vix_enhanced.csv'
OUTPUT_FILE_PATH = "final_merged_dataset_enhanced.csv"

# Hyperparameter for the exponential weighting, as described in Ren et al. (2019).
# A higher value means older news decays in importance more quickly.
# This value can be tuned in more advanced experiments.
LAMBDA = 0.5


# --- Main Script ---
def merge_data_with_ren_et_al(price_path, sentiment_path):
    """
    Merges price and sentiment data using the Ren et al. (2019) methodology
    to handle sentiment over non-trading days.
    Now includes all 6 sentiment features.
    """
    try:
        # --- Step 1: Load and Prepare Data ---
        price_df = pd.read_csv(price_path, parse_dates=['Date'], index_col='Date')
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['event_date'], index_col='event_date')
        sentiment_df.index.name = 'Date'
        print("Files read successfully.")
        print(f"Price data shape: {price_df.shape}")
        print(f"Sentiment data shape: {sentiment_df.shape}")
        print(f"Available sentiment columns: {sentiment_df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

    # Ensure sentiment data has no gaps by forward-filling.
    # This creates a complete calendar of the latest known sentiment.
    full_date_range = pd.date_range(start=sentiment_df.index.min(), end=sentiment_df.index.max(), freq='D')
    sentiment_df = sentiment_df.reindex(full_date_range).ffill()
    print(f"Sentiment data after forward-fill: {sentiment_df.shape}")

    # --- Step 2: Calculate Modified Sentiment Features ---
    # Updated to include all 6 sentiment columns
    sentiment_columns = [
        'positive_prob',
        'negative_prob',
        'neutral_prob',
        'confidence',
        'sentiment_score',
        'polarity_strength'
    ]

    # Verify all columns exist in the sentiment data
    missing_columns = [col for col in sentiment_columns if col not in sentiment_df.columns]
    if missing_columns:
        print(f"Warning: Missing sentiment columns: {missing_columns}")
        # Filter to only existing columns
        sentiment_columns = [col for col in sentiment_columns if col in sentiment_df.columns]
        print(f"Using available columns: {sentiment_columns}")

    lagged_sentiment_features = {}

    # Initialize dictionaries to hold the new feature data
    for col in sentiment_columns:
        lagged_sentiment_features[f"{col}_lag1"] = []

    print(f"\nProcessing {len(sentiment_columns)} sentiment features...")

    # Iterate through the trading days to calculate the correct lagged sentiment
    for i in range(1, len(price_df)):
        current_trading_day = price_df.index[i]
        prev_trading_day = price_df.index[i - 1]

        # Calculate the gap in calendar days since the last trading day
        gap = (current_trading_day - prev_trading_day).days

        if gap == 1:
            # Normal trading day: use sentiment from the previous day
            try:
                sentiment_values = sentiment_df.loc[prev_trading_day, sentiment_columns]
            except KeyError:
                # If previous day's sentiment is not available, use forward-fill
                available_sentiment = sentiment_df.loc[:prev_trading_day].last_valid_index()
                if available_sentiment is not None:
                    sentiment_values = sentiment_df.loc[available_sentiment, sentiment_columns]
                else:
                    # Use zeros if no sentiment data available
                    sentiment_values = pd.Series([0] * len(sentiment_columns), index=sentiment_columns)
        else:
            # Post-weekend/holiday: apply the Ren et al. weighted average

            # Get the slice of sentiment for all days in the gap
            # (e.g., Sat, Sun, Mon for a Monday)
            try:
                sentiment_slice = sentiment_df.loc[prev_trading_day + pd.Timedelta(days=1): current_trading_day]

                if len(sentiment_slice) > 0:
                    # Generate exponential weights. Most recent day gets highest weight.
                    weights = np.exp(-LAMBDA * (len(sentiment_slice) - 1 - np.arange(len(sentiment_slice))))

                    # Calculate the weighted average for each sentiment column
                    sentiment_values = np.average(sentiment_slice[sentiment_columns], axis=0, weights=weights)
                else:
                    # If no sentiment data in the gap, use the last available sentiment
                    available_sentiment = sentiment_df.loc[:prev_trading_day].last_valid_index()
                    if available_sentiment is not None:
                        sentiment_values = sentiment_df.loc[available_sentiment, sentiment_columns]
                    else:
                        sentiment_values = pd.Series([0] * len(sentiment_columns), index=sentiment_columns)
            except Exception as e:
                print(f"Error processing gap for {current_trading_day}: {e}")
                # Fallback to zeros
                sentiment_values = pd.Series([0] * len(sentiment_columns), index=sentiment_columns)

        # Append the calculated features
        if isinstance(sentiment_values, pd.Series):
            for col in sentiment_columns:
                lagged_sentiment_features[f"{col}_lag1"].append(sentiment_values[col])
        else:
            # If sentiment_values is a numpy array
            for idx, col in enumerate(sentiment_columns):
                lagged_sentiment_features[f"{col}_lag1"].append(sentiment_values[idx])

    # --- Step 3: Combine and Finalize the Dataset ---
    # Create a new DataFrame from the calculated features
    features_df = pd.DataFrame(lagged_sentiment_features, index=price_df.index[1:])

    # Join the new features with the original price data
    final_df = price_df.join(features_df, how='inner')

    print(f"\nFinal dataset created using Ren et al. methodology.")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
    print(f"\nNew sentiment features added:")
    for col in sentiment_columns:
        print(f"  - {col}_lag1")

    print("\nPreview of final dataset:")
    print(final_df.head())

    print("\nBasic statistics of sentiment features:")
    sentiment_feature_cols = [f"{col}_lag1" for col in sentiment_columns]
    print(final_df[sentiment_feature_cols].describe())

    return final_df


def save_enhanced_dataset(df, output_path):
    """
    Save the enhanced dataset with additional metadata.
    """
    try:
        # Save the main dataset
        df.to_csv(output_path)

        # Create a summary file
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Enhanced Price-Sentiment Dataset Summary\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Created using Ren et al. (2019) methodology\n")
            f.write(f"Lambda parameter: {LAMBDA}\n\n")
            f.write(f"Input files:\n")
            f.write(f"  Price data: {PRICE_DATA_FILE_PATH}\n")
            f.write(f"  Sentiment data: {SENTIMENT_FILE_PATH}\n")
            f.write(f"Output file: {output_path}\n\n")
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Date range: {df.index.min()} to {df.index.max()}\n")
            f.write(f"Total trading days: {len(df)}\n\n")
            f.write("Columns:\n")
            for col in df.columns:
                f.write(f"  {col}\n")

        print(f"Dataset saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")

    except Exception as e:
        print(f"Error saving files: {e}")


# --- Execution ---
if __name__ == "__main__":
    print("Starting enhanced price-sentiment data merger...")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Lambda (decay parameter): {LAMBDA}")
    print(f"  Price data: {PRICE_DATA_FILE_PATH}")
    print(f"  Sentiment data: {SENTIMENT_FILE_PATH}")
    print(f"  Output file: {OUTPUT_FILE_PATH}")
    print()

    final_dataset = merge_data_with_ren_et_al(PRICE_DATA_FILE_PATH, SENTIMENT_FILE_PATH)

    if final_dataset is not None:
        save_enhanced_dataset(final_dataset, OUTPUT_FILE_PATH)
        print(f"\nProcess completed successfully!")
        print(f"Enhanced dataset with 6 sentiment features ready for use.")
    else:
        print(f"\nProcess failed. Please check the error messages above.")