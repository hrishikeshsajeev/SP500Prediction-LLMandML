import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict


class BalancedFinancialCleaner:
    """
    Balanced TF-IDF cleaner that keeps legitimate financial articles while filtering out irrelevant content.
    Uses multiple scoring methods and smart thresholds.
    """

    def __init__(self, reference_keywords: List[str]):
        self.reference_keywords = reference_keywords
        self.reference_text = " ".join(reference_keywords)

        # Balanced TF-IDF settings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            max_features=5000,
            min_df=1,  # Keep rare financial terms
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            max_df=0.95  # Keep most terms
        )

        # Financial indicators (positive signals)
        self.financial_indicators = {
            'strong': [
                r'\b(earnings|revenue|profit|loss|dividend)\b',
                r'\b(federal reserve|fed|interest rate|monetary policy)\b',
                r'\b(stock|market|nasdaq|dow|s&p|trading|investor)\b',
                r'\b(inflation|gdp|unemployment|recession|economic)\b',
                r'\b(bank|banking|financial|investment|fund)\b',
                r'\b(merger|acquisition|ipo|buyback)\b',
                r'\b(quarter|quarterly|annual|guidance|forecast)\b'
            ],
            'moderate': [
                r'\b(company|corporate|business|industry|sector)\b',
                r'\b(growth|decline|increase|decrease|rise|fall)\b',
                r'\b(analyst|rating|target|recommendation)\b',
                r'\b(cash|debt|capital|financing)\b',
                r'\b(technology|energy|healthcare|consumer)\b'
            ]
        }

        # Non-financial indicators (negative signals)
        self.non_financial_indicators = [
            r'\b(celebrity|actor|actress|singer|musician)\b',
            r'\b(movie|film|television|tv show|series)\b',
            r'\b(sports|football|basketball|baseball|soccer)\b',
            r'\b(wedding|divorce|dating|relationship)\b',
            r'\b(health|medical|doctor|hospital|treatment)\b',
            r'\b(weather|climate|storm|hurricane)\b',
            r'\b(food|recipe|cooking|restaurant|chef)\b',
            r'\b(fashion|style|beauty|makeup)\b',
            r'\b(travel|vacation|tourism|hotel)\b',
            r'\b(crime|murder|theft|assault|arrest)\b',
            r'\b(lawsuit|court|trial|judge)\b',  # Unless combined with financial terms
            r'\b(ski|skiing|crash|accident)\b'  # Specifically for Paltrow case
        ]

    def calculate_financial_score(self, headline: str) -> Dict[str, float]:
        """Calculate multiple scores for financial relevance."""
        if pd.isna(headline):
            headline = ""

        headline_lower = str(headline).lower()

        # Score 1: Strong financial keywords
        strong_score = sum(1 for pattern in self.financial_indicators['strong']
                           if re.search(pattern, headline_lower))

        # Score 2: Moderate financial keywords
        moderate_score = sum(1 for pattern in self.financial_indicators['moderate']
                             if re.search(pattern, headline_lower)) * 0.5

        # Score 3: Non-financial penalty
        penalty_score = sum(1 for pattern in self.non_financial_indicators
                            if re.search(pattern, headline_lower))

        # Combined financial relevance score
        financial_relevance = strong_score + moderate_score - penalty_score

        return {
            'strong_financial': strong_score,
            'moderate_financial': moderate_score,
            'non_financial_penalty': penalty_score,
            'financial_relevance': max(0, financial_relevance)  # Don't go negative
        }

    def preprocess_headline(self, headline: str) -> str:
        """Gentle preprocessing that preserves financial terms."""
        if pd.isna(headline):
            return ""

        headline = str(headline).lower()

        # Remove URLs and HTML, but keep most text
        headline = re.sub(r'<[^>]+>', '', headline)
        headline = re.sub(r'http[s]?://\S+', '', headline)
        headline = re.sub(r'[^\w\s\-\.\,\&]', ' ', headline)  # Keep & for S&P, etc.
        headline = ' '.join(headline.split())

        return headline

    def calculate_tfidf_similarity(self, headlines: List[str]) -> np.ndarray:
        """Calculate TF-IDF similarity scores."""
        processed_headlines = [self.preprocess_headline(headline) for headline in headlines]
        all_texts = [self.reference_text] + processed_headlines

        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        reference_vector = tfidf_matrix[0:1]
        headline_vectors = tfidf_matrix[1:]

        similarity_scores = cosine_similarity(reference_vector, headline_vectors).flatten()
        return similarity_scores

    def smart_filter(self, df: pd.DataFrame, tfidf_threshold: float = 0.02) -> pd.DataFrame:
        """Apply smart filtering using multiple criteria."""

        print("Applying smart multi-criteria filtering...")

        # Calculate financial scores for all headlines
        financial_data = df['headline'].apply(self.calculate_financial_score)

        # Extract scores into separate columns
        df['strong_financial'] = [data['strong_financial'] for data in financial_data]
        df['moderate_financial'] = [data['moderate_financial'] for data in financial_data]
        df['non_financial_penalty'] = [data['non_financial_penalty'] for data in financial_data]
        df['financial_relevance'] = [data['financial_relevance'] for data in financial_data]

        # Apply graduated filtering criteria
        conditions = [
            # Tier 1: Strong financial content (keep regardless of TF-IDF)
            (df['strong_financial'] >= 2),

            # Tier 2: Good financial content with decent TF-IDF
            (df['strong_financial'] >= 1) & (df['tfidf_similarity'] >= tfidf_threshold),

            # Tier 3: Moderate financial content with good TF-IDF
            (df['financial_relevance'] >= 1) & (df['tfidf_similarity'] >= tfidf_threshold * 1.5),

            # Tier 4: High TF-IDF but no obvious non-financial penalties
            (df['tfidf_similarity'] >= tfidf_threshold * 3) & (df['non_financial_penalty'] == 0)
        ]

        # Combine all conditions
        keep_mask = pd.Series(False, index=df.index)
        for condition in conditions:
            keep_mask |= condition

        return df[keep_mask].copy()

    def analyze_filtering_results(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame):
        """Analyze what was kept vs removed."""

        print(f"\n=== FILTERING ANALYSIS ===")
        print(f"Original articles: {len(original_df)}")
        print(f"Kept articles: {len(filtered_df)}")
        print(f"Removed articles: {len(original_df) - len(filtered_df)}")
        print(f"Retention rate: {len(filtered_df) / len(original_df) * 100:.1f}%")

        # Analyze what was kept
        print(f"\n=== KEPT ARTICLES BREAKDOWN ===")
        kept_stats = {
            'Strong financial (2+ keywords)': len(filtered_df[filtered_df['strong_financial'] >= 2]),
            'Good financial (1+ strong keyword)': len(filtered_df[filtered_df['strong_financial'] >= 1]),
            'Moderate financial relevance': len(filtered_df[filtered_df['financial_relevance'] >= 1]),
            'High TF-IDF, no penalties': len(
                filtered_df[(filtered_df['tfidf_similarity'] >= 0.06) & (filtered_df['non_financial_penalty'] == 0)])
        }

        for category, count in kept_stats.items():
            print(f"{category}: {count} articles")

        # Show sample of kept articles
        print(f"\n=== SAMPLE OF KEPT ARTICLES ===")
        sample_kept = filtered_df.nlargest(5, 'tfidf_similarity')
        for i, (idx, row) in enumerate(sample_kept.iterrows()):
            print(f"{i + 1}. TF-IDF: {row['tfidf_similarity']:.4f}, Financial: {row['financial_relevance']:.1f}")
            print(f"   Headline: {str(row['headline'])[:120]}...")
            print()

        # Analyze what was removed
        removed_df = original_df[~original_df.index.isin(filtered_df.index)]
        if len(removed_df) > 0:
            print(f"\n=== SAMPLE OF REMOVED ARTICLES ===")
            # Show worst offenders (high penalty score)
            sample_removed = removed_df.nlargest(5, 'non_financial_penalty')
            for i, (idx, row) in enumerate(sample_removed.iterrows()):
                print(f"{i + 1}. TF-IDF: {row['tfidf_similarity']:.4f}, Penalty: {row['non_financial_penalty']}")
                print(f"   Headline: {str(row['headline'])[:120]}...")
                print()

    def clean_financial_csv(self,
                            input_file: str,
                            output_file: str,
                            tfidf_threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main cleaning function with balanced approach."""

        print(f"Loading CSV: {input_file}")
        df = pd.read_csv(input_file)

        print(f"Original dataset: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        if 'headline' not in df.columns:
            raise ValueError("'headline' column not found")

        # Calculate TF-IDF scores
        print("Calculating TF-IDF similarity scores...")
        headlines = df['headline'].fillna('').tolist()
        similarity_scores = self.calculate_tfidf_similarity(headlines)
        df['tfidf_similarity'] = similarity_scores

        # Apply smart filtering
        filtered_df = self.smart_filter(df, tfidf_threshold)

        # Analyze results
        self.analyze_filtering_results(df, filtered_df)

        # Prepare output (remove temporary scoring columns)
        output_columns = [col for col in filtered_df.columns
                          if col not in ['strong_financial', 'moderate_financial',
                                         'non_financial_penalty', 'financial_relevance', 'tfidf_similarity']]

        output_df = filtered_df[output_columns].copy()

        # Sort by original TF-IDF score (best first)
        if 'tfidf_similarity' in filtered_df.columns:
            output_df = output_df.loc[filtered_df.sort_values('tfidf_similarity', ascending=False).index]

        # Save results
        output_df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")

        return output_df, df


def main():
    """Run balanced financial news cleaning."""

    # Comprehensive but balanced keyword list
    FINANCIAL_KEYWORDS = [
        # Core financial terms
        "earnings", "revenue", "profit", "loss", "dividend", "eps",
        "federal reserve", "fed", "interest rate", "monetary policy", "fomc",
        "stock market", "nasdaq", "dow jones", "s&p 500", "market rally",
        "inflation", "gdp", "unemployment", "recession", "economic growth",
        "bank", "banking", "financial sector", "investment", "investor",

        # Corporate finance
        "merger", "acquisition", "ipo", "buyback", "guidance", "forecast",
        "quarterly", "annual", "cash flow", "debt", "financing",

        # Market terms
        "trading", "bull market", "bear market", "volatility", "correction",
        "analyst rating", "price target", "upgrade", "downgrade",

        # Economic indicators
        "consumer spending", "retail sales", "manufacturing", "housing market",
        "employment", "jobs report", "labor market", "inflation rate",

        # Sectors
        "technology", "energy", "healthcare", "financial services",
        "consumer discretionary", "industrials", "materials"
    ]

    # File paths
    input_file = r'C:\Users\User\Documents\Dissertation\BigQuery\BroadMkt\21-23\BroadMarket_21_23B_Mkt.csv'
    output_file = r'C:\Users\User\Documents\Dissertation\BigQuery\BroadMkt\21-23\BroadMarket_21_23B_Mkt_balanced_clean.csv'

    # Initialize balanced cleaner
    cleaner = BalancedFinancialCleaner(FINANCIAL_KEYWORDS)

    try:
        print("Starting balanced financial news cleaning...")

        cleaned_df, original_df = cleaner.clean_financial_csv(
            input_file=input_file,
            output_file=output_file,
            tfidf_threshold=0.02  # Lower threshold, but smart filtering
        )

        print(f"\nBalanced cleaning completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()