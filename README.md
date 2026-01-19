# 📈 Comparitive analysis Two-Stage Regression Model (SVR) vs End to End LLM Pipeline for Stock Market Prediction



---

## Executive Summary

This study investigates whether Large Language Models outperform traditional Machine Learning approaches for portfolio return generation by processing financial sentiment and technical data. The study examines how different information processing designs capture market dynamics. Using S&P500 index data from March 2021 to March 2025, the study implements 2 forecasting architectures. The first is a two stage Support Vector Regression model with technical indicators and sentiment data quantified by using FinBERT. The ML underwent thorough feature engineering across the variables. The second is an end to end LLM model which processes sentiment text directly along with technical indicators. The LLM model underwent detailed prompt engineering to capture relationships within the data.


The first model utilises a **Support Vector Regression (SVR)** architecture to:
1.  Predict future technical indicators.
2.  Forecast daily returns based on predicted indicators and market sentiment.
3.  Generate actionable trading signals (Long/Short/Hold).

The second model utilises a **an End to End LLM** architecture to:
1.  Forecast daily returns based on predicted indicators and market sentiment - derived from NLP.
2.  Generate actionable trading signals (Long/Short/Hold).
---

## Research Context

The forecasting of financial markets remains one of the most contested challenges in the world of finance. While the Efficient Market Hypothesis asserts that returns are unpredictable, behavioural and technological perspectives continue to challenge this view. This study seeks to determine whether an LLM with the same dataset can achieve superior predictive performance compared to a feature-engineered ML model. This study focuses on forecasting the daily movement of the S&P 500 index, a key benchmark for the U.S. stock market.

> **Do end-to-end LLMs outperform traditional machine learning approaches for portfolio return generation on the S&P500 index when processing financial sentiment and technical data?**

---

## Objectives

-   **Data Engineering**: Construct a robust pipeline sourcing relevant news article links through querieng the GDELT database in BigQuery, scrape text from news articles using Newspaper3k and fetch market data from yfinance.
-   **Sentiment Analysis**: Fine-tune **FinBERT** to quantify market sentiment from financial news.

-   **Predictive Modelling - ML**: Build a two-stage SVR:
    -   *Stage 1*: Forecast future market states (RSI, VIX, etc.).
    -   *Stage 2*: Predict asset returns using Stage 1 outputs.
      
-   **Predictive Modelling - LLM**: Build an End to End LLM predictive model:
    -  Considering all technical indicators, market context, volatility environment, and news flow, predict the S&P 500's next trading day percentage return.
-   **Backtesting**: Evaluate trading performance using Sharpe Ratio, Max Drawdown, and Profit Factor.


---

## Data Sources

The project merges structured and unstructured data:
-   **Market Data**: historical price, volume, volatility (VIX).
-   **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA.
-   **Text Data**: Financial news articles scraped and processed for sentiment.

---

## Methodology

### The Pipeline

#### 1. Data Engineering
-   **News Sourcing**: Financial news articles were sourced by querying the **GDELT database** via **Google BigQuery**.
-   **Scraping**: The **Newspaper3k** library was used to scrape full text from the identified URLs.
-   **Market Data**: Historical price data (Open, High, Low, Close, Volume) was fetched using **yfinance**.

#### 2. Predictive Modelling - ML (Two-Stage SVR)
This architecture separates indicator forecasting from return prediction:
1.  **Sentiment Quantification**: **FinBERT** (Financial BERT) processes news headlines to generate sentiment scores.
2.  **Stage 1 (State Forecasting)**: An **SVR** model predicts next-day technical indicators (RSI, VIX) based on current values and sentiment.
3.  **Stage 2 (Return Prediction)**: A second SVR model predicts the daily return using the forecasted indicators from Stage 1.

#### 3. Trading Signal Generation and Portfolio Construction

Trading signals are generated using a rule based threshold system that converts the SVR’s returns forecasts and confidence measures into decisions. This ensures only high confidence forecasts drive portfolio positions and avoids trading on marginal predictions. The three signals are generated when there is a sufficient confidence level.
-	1 for long - predicted returns>positive threshold.
-	-1 for short - predicted returns<negative threshold.
-	0 for neutral - Neutral signals apply when neither conditions are met, contributing zero return. 


#### 4. Predictive Modelling - LLM (End-to-End)
This architecture utilizes **Vertex AI (Gemini 2.5 Pro)** in a reasoning-based approach:
1.  **Prompt Engineering**: A comprehensive prompt feeds the model with a merged dataset combining technical indicators, volatility regimes, and news data.
2.  **Ensemble Prediction**: The model generates multiple predictions (ensemble approach) to reduce variance and hallucination.
3.  **Direct Forecasting**: The LLM directly outputs the predicted return and confidence level, skipping the intermediate indicator forecasting step.

---

## Key Findings & Model Comparison

The study compared the traditional **Support Vector Regression (SVR)** approach against the **Generative AI (Vertex AI/Gemini)** model.

### Performance Summary
| Metric | Traditional SVR Model | Vertex AI (Gemini 2.5) |
| :--- | :--- | :--- |
| **Total Return** | **51.73%** | **83.74%**  |
| **Annualised Return** | 14.76% | **16.86%** |
| **Sharpe Ratio** | 0.924 | **1.015** |
| **Max Drawdown** | **-12.84%** (Safer) | -23.36% (Riskier) |
| **Directional Accuracy** | **55.95%** | 53.60% |
| **Prediction Error (MAE)** | **0.703** | 1.94 |

### 💡 Critical Insights (Discussion)
1.  **Profitability vs. Accuracy Paradox**: The SVR model achieved a lower **MAE (0.703)** compared to the LLM (**1.94**), indicating better daily precision. However, the **LLM model was significantly more profitable** likely due to its ability to understand market sentiment better by leveraging **NLP** (+83.74% vs +51.73%).
2.  **Reasoning over Patterns**: As noted in the study, "The LLM model's natural language processing ability captures market dynamics better, allowing it to set up winning trades when market conditions allow." It effectively acted as a human analyst, prioritizing high-conviction setups over noise.
3.  **Risk Profile**: The SVR model offered a smoother equity curve with a lower Max Drawdown (**-12.84%**), making it suitable for risk-averse strategies. The LLM had a higher drawdown (**-23.36%**) but compensated with superior total returns.
4.  **Hybrid Potential**: The study concludes that a hybrid approach—using ML for stability and LLMs for opportunity capturing—would likely yield the optimal risk-adjusted performance.

---

## Visualizations

### 1. Stage 1: Technical Indicator Prediction
The first stage validates the model's ability to forecast key market drivers like RSI and Volatility. The model's trading signals were backtested against historical data.

<img src="images/plot_2.png" width="800" alt="Cumulative Returns Analysis (SVR)"/>

### 2. Stage 2: Return Prediction
The second stage translates these technical forecasts into concrete return predictions.

<img src="images/plot_3.png" width="800" alt="Trade Distribution"/>


### 3. Cumulative Portfolio Performance of ML Model
The ML model's performance quantified as part of a long-short portfolio.

<img src="images/plot_0.png" width="800" alt="Stage 1 Validation Plots"/>

### 4. Vertex AI Cumulative Returns (Generative AI)
Demonstrating the superior profitability of the reasoning-based model.

<img src="images/vertex_fig_1.png" width="800" alt="Vertex AI Cumulative Returns"/>



---

## Tools & Technologies Used

-   **Languages**: Python, SQL
-   **Libraries**: `pandas`, `numpy`, `scikit-learn` (SVR), `matplotlib`, `seaborn`
-   **Cloud & AI**: Google BigQuery, **Vertex AI (Gemini 2.5 Pro)**

---

## Limitations of the Study

### 1. Data Quality and Sources
The study relied on freely accessible data from Yahoo Finance and GDELT. While suitable for academic research, GDELT's news data can be noisy and less relevant, potentially sensitive sentiment indices to misrepresent market tone.

### 2. Computational Constraints
Experiments were conducted on a personal computer, limiting hyperparameter optimization and temporal scope (2021-2025). This restricted the ability to test across multiple market cycles and utilize more computationally intensive models like LSTMs.

### 3. LLM Reproducibility
Generative AI models exhibit inherent variability. Despite controlled prompts, repeated runs produced varying forecasts, posing a challenge for the reproducibility required in financial forecasting tools.

### 4. Model Choice
The study used SVR due to computational limits, though literature suggests LSTMs may offer superior performance. Similarly, the cost-effective Gemini 2.5 Flash Lite was chosen over more capable reasoning models (like Gemini Pro/Ultra) to manage API costs.

---

## Recommendations for Future Work

### 1. Enhanced Data Sourcing
Future research should leverage high-quality, curated data feeds (e.g., Bloomberg, LSEG) and real-time APIs to improve sentiment measurement and enable timely trade execution.

### 2. High-Performance Computing
Utilizing cloud-based environments (e.g., Google Colab Pro) would allow for broader temporal scopes, larger parameter grids, and the implementation of deep learning architectures like LSTMs.

### 3. Hybrid Model Design
A hybrid approach combining the stability/accuracy of ML models with the reasoning/opportunity-capturing abilities of LLMs offers the most promising path for maximizing risk-adjusted returns.

---
