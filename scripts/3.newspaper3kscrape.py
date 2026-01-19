import pandas as pd
from newspaper import Article, Config
from tqdm import tqdm
import logging

# --- Configuration ---

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
INPUT_PATHNAME = '/Users/hrishikeshsajeev/Dissertation codes/BigQuery/BroadMkt/BroadMarket_cleaned_003.csv'
PRIMARY_OUTPUT_PATHNAME = '/Users/hrishikeshsajeev/Dissertation codes/BigQuery/BroadMkt/BroadMarket_scraped_professional.csv'
HOST_REPORT_PATHNAME = '/Users/hrishikeshsajeev/Dissertation codes/BigQuery/BroadMkt/scraping_host_report.csv'

# Column names from your input CSV
URL_COLUMN_NAME = 'url'
HOST_COLUMN_NAME = 'website_host'


# --- Core Functions ---

def load_data(filepath):
    """Loads data from a specified CSV file."""
    try:
        logging.info(f"Loading data from {filepath}...")
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {filepath}")
        return None


def get_execution_mode():
    """Prompts the user to select the execution mode."""
    while True:
        mode = input("Select execution mode ('full' or 'test'): ").lower().strip()
        if mode == 'full':
            return 'full', 0
        elif mode == 'test':
            try:
                batch_size = int(input("Enter batch size for test run: "))
                if batch_size > 0:
                    return 'test', batch_size
                else:
                    logging.warning("Batch size must be a positive integer.")
            except ValueError:
                logging.warning("Invalid input. Please enter an integer.")
        else:
            logging.warning("Invalid mode selected. Please enter 'full' or 'test'.")


def scrape_articles(df, url_column):
    """Scrapes the full text of articles from a list of URLs in a DataFrame."""
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    config.request_timeout = 10

    article_texts = []
    logging.info(f"Initiating scraping for {len(df)} articles.")

    for url in tqdm(df[url_column], desc="Scraping Progress"):
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            article_texts.append(article.text)
        except Exception as e:
            logging.debug(f"Failed to process {url}: {e}")
            article_texts.append('')

    return article_texts


def generate_host_report(df_results, host_column, output_path):
    """Analyzes results and creates a detailed CSV report with a grand total row."""
    logging.info("Generating detailed website host scraping report...")

    if host_column not in df_results.columns:
        logging.error(f"Host column '{host_column}' not found. Skipping host report.")
        return

    # Group by host and aggregate to get counts
    host_summary = df_results.groupby(host_column).agg(
        total_articles=('url', 'size'),
        successful_scrapes=('scraped_full_text', lambda x: (x != '').sum())
    ).reset_index()

    host_summary['failed_scrapes'] = host_summary['total_articles'] - host_summary['successful_scrapes']

    # Sort by the number of articles processed
    host_summary = host_summary.sort_values(by='total_articles', ascending=False)

    # Create a grand total row
    total_row = {
        host_column: '--- GRAND TOTAL ---',
        'total_articles': host_summary['total_articles'].sum(),
        'successful_scrapes': host_summary['successful_scrapes'].sum(),
        'failed_scrapes': host_summary['failed_scrapes'].sum()
    }
    total_df = pd.DataFrame([total_row])

    # Append the total row to the summary DataFrame
    final_report = pd.concat([host_summary, total_df], ignore_index=True)

    try:
        final_report.to_csv(output_path, index=False)
        logging.info(f"Host report with grand total successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save host report: {e}")


# --- Main Execution Block ---

def main():
    """Main function to orchestrate the data loading, scraping, and saving process."""
    df_full = load_data(INPUT_PATHNAME)
    if df_full is None:
        return

    mode, batch_size = get_execution_mode()
    if mode == 'test':
        df_to_scrape = df_full.head(batch_size).copy()
        logging.info(f"Test mode: Processing first {len(df_to_scrape)} articles.")
    else:
        df_to_scrape = df_full.copy()
        logging.info(f"Full mode: Processing all {len(df_to_scrape)} articles.")

    # Execute scraping and add results to the DataFrame
    scraped_texts = scrape_articles(df_to_scrape, URL_COLUMN_NAME)
    df_to_scrape['scraped_full_text'] = scraped_texts

    # Save primary results file
    try:
        df_to_scrape.to_csv(PRIMARY_OUTPUT_PATHNAME, index=False)
        logging.info(f"Primary scraping results saved to {PRIMARY_OUTPUT_PATHNAME}")
    except Exception as e:
        logging.error(f"Failed to save primary results: {e}")

    # Generate the detailed host report with a grand total row
    generate_host_report(df_to_scrape, HOST_COLUMN_NAME, HOST_REPORT_PATHNAME)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()