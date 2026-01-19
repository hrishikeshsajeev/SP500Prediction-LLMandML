import pandas as pd
import numpy as np


def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()


def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()


def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns MACD line, Signal line, and Histogram
    """
    ema_fast = calculate_ema(close_prices, fast)
    ema_slow = calculate_ema(close_prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_rsi(close_prices, window=14):
    """Calculate Relative Strength Index"""
    delta = close_prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_realized_volatility(close_prices, window=20):
    """
    Calculate realized volatility using daily returns
    Annualized volatility (252 trading days)
    """
    returns = close_prices.pct_change()
    realized_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return realized_vol


def calculate_vrp(vix_close, realized_vol):
    """
    Calculate Volatility Risk Premium (VRP)
    VRP = VIX - Realized Volatility
    Positive VRP indicates VIX is higher than realized volatility
    """
    return vix_close - realized_vol


def add_technical_indicators(df):
    """
    Add all technical indicators to the dataframe

    Parameters:
    df (pd.DataFrame): DataFrame with GSPC OHLC and VIX data

    Returns:
    pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying original data
    data = df.copy()

    # Ensure Date column is datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Calculate technical indicators
    print("Calculating technical indicators...")

    # 1. SMA (10-day)
    data['SMA_10'] = calculate_sma(data['GSPC_Close'], 10)

    # 2. EMA (30-day)
    data['EMA_30'] = calculate_ema(data['GSPC_Close'], 30)

    # 3. MACD (12, 26, 9)
    macd, signal, histogram = calculate_macd(data['GSPC_Close'])
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Histogram'] = histogram

    # 4. RSI (14-day)
    data['RSI_14'] = calculate_rsi(data['GSPC_Close'], 14)

    # 5. Realized Volatility (20-day rolling)
    data['Realized_Vol_20'] = calculate_realized_volatility(data['GSPC_Close'], 20)

    # 6. Volatility Risk Premium (VRP)
    data['VRP'] = calculate_vrp(data['VIX_Close'], data['Realized_Vol_20'])

    # Additional useful indicators for context
    # Price position relative to moving averages
    data['Price_vs_SMA10'] = (data['GSPC_Close'] / data['SMA_10'] - 1) * 100
    data['Price_vs_EMA30'] = (data['GSPC_Close'] / data['EMA_30'] - 1) * 100

    # Daily returns
    data['Daily_Return'] = data['GSPC_Close'].pct_change() * 100

    print("Technical indicators calculated successfully!")

    return data


def save_enhanced_data(data, filename='gspc_vix_enhanced.csv'):
    """Save the enhanced dataset with technical indicators"""
    # Remove rows with NaN values (due to rolling calculations)
    clean_data = data.dropna()

    # Save to CSV
    clean_data.to_csv(filename, index=False)
    print(f"Enhanced data saved to {filename}")
    print(f"Clean data shape: {clean_data.shape}")

    # Save to Excel with proper formatting
    excel_filename = filename.replace('.csv', '.xlsx')
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        clean_data.to_excel(writer, sheet_name='Enhanced_Data', index=False)

        # Get worksheet for formatting
        worksheet = writer.sheets['Enhanced_Data']

        # Set column widths
        worksheet.column_dimensions['A'].width = 12  # Date
        for col_letter in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
            if col_letter <= chr(ord('A') + len(clean_data.columns) - 1):
                worksheet.column_dimensions[col_letter].width = 15

    print(f"Enhanced data also saved to {excel_filename}")

    return clean_data


def display_indicators_summary(data):
    """Display summary statistics for technical indicators"""
    indicators = ['SMA_10', 'EMA_30', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                  'RSI_14', 'Realized_Vol_20', 'VRP', 'Price_vs_SMA10', 'Price_vs_EMA30']

    print("\n" + "=" * 80)
    print("TECHNICAL INDICATORS SUMMARY")
    print("=" * 80)

    for indicator in indicators:
        if indicator in data.columns:
            print(f"\n{indicator}:")
            print(f"  Mean: {data[indicator].mean():.2f}")
            print(f"  Std:  {data[indicator].std():.2f}")
            print(f"  Min:  {data[indicator].min():.2f}")
            print(f"  Max:  {data[indicator].max():.2f}")

    print(f"\nData Date Range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Total Records: {len(data)}")


# Load the original data
print("Loading GSPC and VIX data...")
data = pd.read_csv('gspc_vix_data.csv')

print(f"Original data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Add technical indicators
enhanced_data = add_technical_indicators(data)

# Display summary
display_indicators_summary(enhanced_data)

# Save enhanced data
clean_data = save_enhanced_data(enhanced_data)

# Display first few rows of enhanced data
print("\nFirst 5 rows of enhanced data:")
print(clean_data.head())

print("\nLast 5 rows of enhanced data:")
print(clean_data.tail())