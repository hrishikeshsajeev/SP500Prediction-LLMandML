import pandas as pd
import numpy as np


def engineer_svm_features(file_path):
    """
    Load data and engineer essential features for SVM stock prediction
    """
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Create new dataframe with engineered features
    features_df = pd.DataFrame(index=df.index)

    # 1. Core Price Features
    print("\n1. Engineering core price features...")
    features_df['close_price'] = df['GSPC_Close']

    # Daily returns (you already have this calculated!)
    features_df['daily_return'] = df['Daily_Return']

    # Volume
    features_df['volume'] = df['GSPC_Volume']

    # Volume ratio (current volume / 20-day average volume)
    volume_ma_20 = df['GSPC_Volume'].rolling(window=20).mean()
    features_df['volume_ratio'] = df['GSPC_Volume'] / volume_ma_20

    # Price volatility (you already have Realized_Vol_20!)
    features_df['price_volatility'] = df['Realized_Vol_20']

    # VIX (volatility index - very important for stock prediction)
    features_df['vix_close'] = df['VIX_Close']

    # VRP (Volatility Risk Premium - you already have this!)
    features_df['vrp'] = df['VRP']

    # 2. Technical Indicators
    print("2. Adding technical indicators...")

    # RSI using your RSI_14 column
    features_df['rsi_14'] = df['RSI_14']

    # MACD features using your exact column names
    features_df['macd'] = df['MACD']
    features_df['macd_signal'] = df['MACD_Signal']
    features_df['macd_histogram'] = df['MACD_Histogram']

    # MACD Signal Ratio (MACD / Signal)
    features_df['macd_signal_ratio'] = df['MACD'] / df['MACD_Signal']
    # Replace infinite values with 0
    features_df['macd_signal_ratio'] = features_df['macd_signal_ratio'].replace([np.inf, -np.inf], 0)

    # 3. Moving Average Ratios
    print("3. Adding moving average ratios...")
    # Using your existing calculated ratios
    features_df['price_vs_sma10'] = df['Price_vs_SMA10']
    features_df['price_vs_ema30'] = df['Price_vs_EMA30']

    # 4. Enhanced Sentiment Features (All 6 Columns)
    print("4. Adding enhanced sentiment features...")

    # All 6 sentiment features from FinBERT analysis
    features_df['positive_prob'] = df['positive_prob_lag1']
    features_df['negative_prob'] = df['negative_prob_lag1']
    features_df['neutral_prob'] = df['neutral_prob_lag1']
    features_df['confidence'] = df['confidence_lag1']
    features_df['sentiment_score'] = df['sentiment_score_lag1']
    features_df['polarity_strength'] = df['polarity_strength_lag1']

    # Additional sentiment-derived features
    print("5. Creating sentiment-derived features...")

    # Removed derived features to keep exactly 12 input features
    # This reduces overfitting risk and follows the original request

    # 6. Create Target Variable (for SVM classification)
    print("6. Creating target variable...")

    # Method 1: Simple next-day direction (most common in research)
    # Target = direction of tomorrow's return compared to today
    next_day_return = features_df['daily_return'].shift(-1)  # Tomorrow's return

    # Create binary classification target:
    # +1 = UP (positive return > threshold)
    # -1 = DOWN (negative return < -threshold)
    # 0 = NEUTRAL (small movements between thresholds)

    threshold = 0.005  # 0.5% threshold to filter out noise

    features_df['target_direction'] = np.where(next_day_return > threshold, 1,  # UP
                                               np.where(next_day_return < -threshold, -1, 0))  # DOWN or NEUTRAL

    # Alternative: Pure binary (UP/DOWN only, no neutral)
    # features_df['target_direction'] = np.where(next_day_return > 0, 1, -1)

    print(f"Target distribution:")
    print(f"UP days (+1): {(features_df['target_direction'] == 1).sum()}")
    print(f"DOWN days (-1): {(features_df['target_direction'] == -1).sum()}")
    print(f"NEUTRAL days (0): {(features_df['target_direction'] == 0).sum()}")

    # Show example of target calculation
    print(f"\nExample target calculation (first 10 days):")
    example_df = pd.DataFrame({
        'Date': features_df.index[:10],
        'Today_Return': features_df['daily_return'][:10],
        'Tomorrow_Return': next_day_return[:10],
        'Target_Direction': features_df['target_direction'][:10]
    })
    print(example_df.round(4))

    # 7. Clean the data
    print("7. Cleaning data...")

    # Remove rows with NaN values (mainly from rolling calculations)
    print(f"Rows before cleaning: {len(features_df)}")
    features_df = features_df.dropna()
    print(f"Rows after cleaning: {len(features_df)}")

    # Final feature selection - INPUT FEATURES ONLY (no target!)
    # EXACTLY 12 INPUT FEATURES
    input_features = [
        # Technical/Price features (6)
        'daily_return',  # Daily_Return (price momentum)
        'volume_ratio',  # Volume signal
        'vix_close',  # VIX (market fear/volatility)
        'rsi_14',  # RSI (momentum oscillator)
        'macd_signal_ratio',  # MACD ratio (trend strength)
        'price_vs_sma10',  # Short-term trend

        # Core sentiment features (6)
        'positive_prob',  # Positive probability
        'negative_prob',  # Negative probability
        'neutral_prob',  # Neutral probability
        'confidence',  # Confidence score
        'sentiment_score',  # Overall sentiment score
        'polarity_strength'  # Polarity strength
    ]

    # Keep target separate for SVM training
    target_feature = ['target_direction']

    # Reference features (kept for analysis but not used in SVM)
    reference_features = ['close_price']

    # Combine all for the final dataset
    final_features = input_features + target_feature + reference_features

    # Keep only features that exist in the dataframe
    available_features = [col for col in final_features if col in features_df.columns]
    missing_features = [col for col in final_features if col not in features_df.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")

    features_df_final = features_df[available_features]

    print(f"\n8. Final dataset summary:")
    print(f"Final shape: {features_df_final.shape}")
    print(f"EXACTLY 12 INPUT features for SVM model")
    print(f"TARGET feature (y): {target_feature}")
    print(f"REFERENCE features: {reference_features}")

    print(f"\nINPUT features (X):")
    available_input_features = [col for col in input_features if col in available_features]
    for i, feature in enumerate(available_input_features, 1):
        print(f"  {i:2d}. {feature}")

    # Display basic statistics for input features only
    print(f"\nInput feature statistics:")
    print(features_df_final[available_input_features].describe())

    # Show target distribution
    print(f"\nTarget variable distribution:")
    target_counts = features_df_final['target_direction'].value_counts()
    print(target_counts)
    print(
        f"Class balance: UP={target_counts.get(1, 0)}, DOWN={target_counts.get(-1, 0)}, NEUTRAL={target_counts.get(0, 0)}")

    # Check for any remaining issues
    print(f"\nData quality check:")
    print(f"Missing values per column:")
    missing_values = features_df_final.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    print(f"\nInfinite values per column:")
    inf_values = features_df_final.replace([np.inf, -np.inf], np.nan).isnull().sum() - features_df_final.isnull().sum()
    if inf_values.sum() > 0:
        print(inf_values[inf_values > 0])
    else:
        print("No infinite values found.")

    # Show correlation between sentiment features
    print(f"\nSentiment feature correlations:")
    sentiment_cols = [col for col in ['positive_prob', 'negative_prob', 'neutral_prob', 'confidence', 'sentiment_score',
                                      'polarity_strength'] if col in features_df_final.columns]
    if len(sentiment_cols) > 1:
        sentiment_corr = features_df_final[sentiment_cols].corr()
        print(sentiment_corr.round(3))

    return features_df_final


def save_engineered_features(file_path, output_path=None):
    """
    Load data, engineer features, and save to new file
    """
    if output_path is None:
        output_path = file_path.replace('.csv', '_svm_features_enhanced.csv')

    # Engineer features
    features_df = engineer_svm_features(file_path)

    # Save to new file
    features_df.to_csv(output_path)
    print(f"\nEnhanced SVM-ready dataset saved to: {output_path}")

    return features_df, output_path


# Execute the feature engineering
if __name__ == "__main__":
    input_file = '/Users/hrishikeshsajeev/Dissertation codes/FInal FIles BERT/final_merged_dataset_enhanced.csv'

    print("=== Enhanced SVM Feature Engineering Pipeline ===")
    print(f"Input file: {input_file}")

    try:
        # Engineer and save features
        features_df, output_file = save_engineered_features(input_file)

        print(f"\n=== SUCCESS ===")
        print(f"Enhanced SVM-ready features saved to: {output_file}")
        print(f"Dataset shape: {features_df.shape}")
        print(f"Date range: {features_df.index.min()} to {features_df.index.max()}")

        # Show sample of the data
        print(f"\nFirst 5 rows of engineered features:")
        print(features_df.head())

        print(f"\nLast 5 rows of engineered features:")
        print(features_df.tail())

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()