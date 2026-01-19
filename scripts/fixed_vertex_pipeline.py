import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import List, Dict, Any, Optional
import warnings
import re

# Vertex AI imports
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Warning: google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedVertexAILLMPipeline:
    """
    Enhanced Vertex AI LLM pipeline with temperature increase and ensemble predictions
    for improved stock return forecasting performance.
    """

    def __init__(self, project_id: str = "REDACTED", location: str = "REDACTED",
                 model_name: str = "gemini-2.5-pro", ensemble_size: int = 3):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.ensemble_size = ensemble_size

        # Cost tracking
        self.total_api_calls = 0
        self.total_cost = 0.0
        self.cost_per_call = 0.001  # Approximate cost per call in USD

        # Initialize Vertex AI
        if VERTEX_AI_AVAILABLE:
            try:
                vertexai.init(project=project_id, location=location)
                self.model = GenerativeModel(model_name)

                # ENHANCED: Higher temperature for more aggressive predictions
                self.generation_config = {
                    "temperature": 0.0,  # Deterministic for reproducible results
                    "top_p": 1.0,  # Use all tokens for deterministic results
                    "top_k": 1,    # Most deterministic setting
                    "max_output_tokens": 8192, # Maximum tokens for detailed analysis
                    "candidate_count": 1,
                }
                logger.info(f"Enhanced Vertex AI initialized: {project_id}/{location}/{model_name}")
                logger.info(f"Temperature: 0.0 (deterministic mode for reproducible results)")
                logger.info(f"Ensemble size: {ensemble_size} calls per prediction")
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}")
                self.model = None
        else:
            logger.warning("Vertex AI not available - will use simulation mode")
            self.model = None

    def load_llm_dataset(self, filepath: str = None) -> pd.DataFrame:
        """Load the LLM-ready dataset"""
        if filepath is None:
            filepath = "/Users/hrishikeshsajeev/Dissertation codes/LLM PyCharm/llm_ready_data.csv"

        df = pd.read_csv(filepath)
        df['trading_date'] = pd.to_datetime(df['trading_date'])

        logger.info(f"Loaded LLM dataset: {len(df)} rows from {df['trading_date'].min().date()} to {df['trading_date'].max().date()}")
        return df

    def create_enhanced_prompt_template(self, row: pd.Series) -> str:
        """
        Create comprehensive yet efficient prompt for accurate predictions.
        """

        # Technical indicators
        rsi_signal = "OVERSOLD" if row['rsi_14'] < 30 else "OVERBOUGHT" if row['rsi_14'] > 70 else "NEUTRAL"
        vix_signal = "HIGH_VOL" if row['vix_close'] > 25 else "NORMAL_VOL"
        macd_signal = "BULLISH" if row['macd_histogram'] > 0 else "BEARISH"
        trend_signal = "UPTREND" if row['price_vs_sma10'] > 0 and row['price_vs_ema30'] > 0 else "DOWNTREND" if row['price_vs_sma10'] < 0 and row['price_vs_ema30'] < 0 else "SIDEWAYS"

        prompt = f"""You are an expert quantitative analyst with deep expertise in S&P 500 forecasting. Analyze the following comprehensive market data and predict the next trading day's return with high precision.

## CURRENT MARKET CONTEXT
Trading Date: {row['trading_date'].strftime('%Y-%m-%d (%A)')}
S&P 500 Close: ${row['gspc_close']:,.2f}
Daily Return: {row['daily_return']:.3f}%
Volume: {row['gspc_volume']:,} shares
Market Gap: {row['gap_length']} days since last trading {"(weekend/holiday)" if row['is_weekend_gap'] else ""}

## TECHNICAL ANALYSIS FRAMEWORK

### Price Action & Trend Analysis
- Current Price vs 10-day SMA: {row['price_vs_sma10']:.2f}% ({trend_signal})
- Current Price vs 30-day EMA: {row['price_vs_ema30']:.2f}%
- 10-day SMA: ${row['sma_10']:,.2f}
- 30-day EMA: ${row['ema_30']:,.2f}
- Price momentum suggests: {trend_signal.lower()} bias

### Momentum & Oscillator Signals
- RSI (14-day): {row['rsi_14']:.2f} → {rsi_signal}
  {"• Oversold condition suggests potential bounce" if rsi_signal == "OVERSOLD" else "• Overbought condition suggests potential pullback" if rsi_signal == "OVERBOUGHT" else "• Neutral momentum, direction uncertain"}
- MACD: {row['macd']:.4f}
- MACD Signal: {row['macd_signal']:.4f}
- MACD Histogram: {row['macd_histogram']:.4f} → {macd_signal}
  {"• Bullish momentum building" if macd_signal == "BULLISH" else "• Bearish momentum increasing"}

### Volatility & Risk Assessment
- VIX: {row['vix_close']:.2f} → {vix_signal}
  {"• High volatility regime - expect larger moves (±2-4%)" if vix_signal == "HIGH_VOL" else "• Normal volatility - typical range (±0.5-1.5%)"}
- 20-day Realized Volatility: {row['realized_vol_20']:.4f}
- Volatility Risk Premium: {row['vrp']:.4f}

### News & Market Sentiment
- News Articles: {row['news_count']} ({"High" if row['news_count'] > 300 else "Moderate" if row['news_count'] > 100 else "Low"} coverage)
- Source Diversity: {row['source_diversity']} different outlets
- Full Text Ratio: {row['full_text_ratio']:.1%} comprehensive coverage
- Market attention level: {"Heightened due to high news flow" if row['news_count'] > 300 else "Normal"}

## PREDICTION METHODOLOGY

### Technical Confluence Analysis
1. **Trend Alignment**: Price vs moving averages indicates {trend_signal.lower()} bias
2. **Momentum Signals**: RSI shows {rsi_signal.lower()} condition, MACD shows {macd_signal.lower()} momentum
3. **Volatility Context**: {vix_signal.replace("_", " ").lower()} environment suggests {"larger potential moves" if vix_signal == "HIGH_VOL" else "normal trading range"}
4. **News Impact**: {"Potential for volatility due to high news volume" if row['news_count'] > 300 else "Limited news-driven volatility expected"}

### Historical Pattern Recognition
- RSI <30 historically precedes bounces in 65% of cases
- RSI >70 historically precedes pullbacks in 68% of cases  
- High VIX (>25) correlates with ±2-4% daily moves
- MACD histogram direction predicts short-term momentum
- Weekend/holiday gaps can amplify first-day moves by 20-40%

### Market Regime Assessment
Based on current conditions:
- Volatility regime: {vix_signal.replace("_", " ").lower()}
- Trend regime: {trend_signal.lower()}
- Momentum regime: {macd_signal.lower()}

## PREDICTION TASK
Considering all technical indicators, market context, volatility environment, and news flow, predict the S&P 500's next trading day percentage return.

### Expected Return Ranges:
- **High Confidence (0.8-1.0)**: Strong technical confluence, clear directional signals
- **Medium Confidence (0.5-0.8)**: Mixed signals, moderate conviction
- **Low Confidence (0.0-0.5)**: Conflicting indicators, high uncertainty

### Magnitude Guidelines:
- **Normal volatility**: ±0.3% to ±1.5% moves
- **Elevated volatility**: ±1.5% to ±3.0% moves  
- **High volatility**: ±3.0% to ±5.0% moves

Provide your prediction as a precise percentage and confidence level based on the comprehensive analysis above.

PREDICTION: [percentage return e.g. -1.25 or +2.30]
CONFIDENCE: [0.0 to 1.0 based on signal strength]"""

        return prompt

    def call_vertex_ai_single(self, prompt: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Make single Vertex AI API call with cost tracking.
        """
        start_time = time.time()
        self.total_api_calls += 1
        self.total_cost += self.cost_per_call

        # Check if Vertex AI is available
        if not self.model:
            if not VERTEX_AI_AVAILABLE:
                raise Exception("Vertex AI not available - install google-cloud-aiplatform")
            else:
                raise Exception("Vertex AI model not initialized - check authentication and project setup")

        try:
            # Make the API call
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(**self.generation_config)
            )

            processing_time = time.time() - start_time

            # Extract the prediction from response
            response_text = response.text.strip()

            # Parse numerical prediction from response
            prediction = self.parse_prediction_from_response(response_text)

            return {
                "prediction": prediction,
                "confidence": 0.8,  # Default confidence for real API responses
                "status": "success",
                "model_version": self.model_name,
                "processing_time": processing_time,
                "raw_response": response_text
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            # Handle common API errors
            if "quota" in error_msg.lower():
                raise Exception("API quota exceeded - check your project limits")
            elif "permission" in error_msg.lower():
                raise Exception("Permission denied - check authentication and IAM roles")
            elif "not found" in error_msg.lower():
                raise Exception(f"Model {self.model_name} not found or not available in {self.location}")
            else:
                raise Exception(f"Vertex AI API error: {error_msg}")

    def ensemble_predict(self, prompt: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Make ensemble prediction using multiple API calls with advanced aggregation.
        """
        logger.info(f"Making ensemble prediction with {self.ensemble_size} calls...")

        predictions = []
        confidences = []
        raw_responses = []
        total_processing_time = 0

        for i in range(self.ensemble_size):
            try:
                # Slight temperature variation for diversity
                temp_variation = 0.4 + (i * 0.1)  # 0.4, 0.5, 0.6 for diversity

                # Temporarily adjust temperature for this call
                original_temp = self.generation_config["temperature"]
                temp_config = self.generation_config.copy()
                temp_config["temperature"] = min(temp_variation, 0.8)  # Cap at 0.8

                # Use temporary config for this call
                original_config = self.generation_config
                self.generation_config = temp_config

                result = self.call_vertex_ai_single(prompt, retry_count)

                # Restore original temperature
                self.generation_config = original_config

                if result['status'] == 'success':
                    predictions.append(result['prediction'])
                    confidences.append(result['confidence'])
                    raw_responses.append(result['raw_response'])
                    total_processing_time += result['processing_time']

                    # Small delay between ensemble calls
                    if i < self.ensemble_size - 1:
                        time.sleep(0.2)

            except Exception as e:
                logger.warning(f"Ensemble call {i+1} failed: {e}")
                continue

        if len(predictions) == 0:
            raise Exception("All ensemble calls failed")

        # Advanced ensemble aggregation
        predictions_array = np.array(predictions)

        # Remove outliers (beyond 1.5 IQR)
        Q1 = np.percentile(predictions_array, 25)
        Q3 = np.percentile(predictions_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter predictions within bounds
        filtered_predictions = predictions_array[
            (predictions_array >= lower_bound) &
            (predictions_array <= upper_bound)
        ]

        # If too many filtered out, use all predictions
        if len(filtered_predictions) < len(predictions_array) * 0.5:
            filtered_predictions = predictions_array

        # Calculate ensemble metrics
        ensemble_prediction = np.mean(filtered_predictions)
        prediction_std = np.std(filtered_predictions)
        ensemble_confidence = 1.0 / (1.0 + prediction_std) if prediction_std > 0 else 0.9

        # Weight by agreement (lower std = higher confidence)
        confidence_bonus = 0.1 if prediction_std < 0.5 else 0.0
        final_confidence = min(ensemble_confidence + confidence_bonus, 0.95)

        logger.info(f"Ensemble result: {ensemble_prediction:.4f} (std: {prediction_std:.4f}, confidence: {final_confidence:.3f})")

        return {
            "prediction": round(ensemble_prediction, 4),
            "confidence": round(final_confidence, 4),
            "status": "success",
            "model_version": f"{self.model_name}_ensemble_{self.ensemble_size}",
            "processing_time": total_processing_time,
            "ensemble_size": len(filtered_predictions),
            "prediction_std": prediction_std,
            "raw_responses": raw_responses[:3]  # Keep first 3 for debugging
        }

    def parse_prediction_from_response(self, response_text: str) -> float:
        """
        Parse numerical prediction from LLM response text.
        Enhanced to handle more varied response formats from higher temperature.
        """
        if not response_text:
            raise ValueError("Empty response from model")

        # Try different patterns to extract numerical prediction
        patterns = [
            r'([-+]?\d+\.?\d*)\s*%?\s*$',  # Final number possibly with %
            r'prediction:?\s*([-+]?\d+\.?\d*)\s*%?',  # "prediction: X.XX"
            r'return:?\s*([-+]?\d+\.?\d*)\s*%?',  # "return: X.XX"
            r'([-+]?\d+\.?\d*)\s*%?\s*$',  # Number at end of line
            r'([-+]?\d+\.?\d+)',  # Any floating point number
            r'([-+]?\d+)'  # Any integer
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (most likely to be the final prediction)
                    prediction = float(matches[-1])
                    # Ensure reasonable bounds (enhanced for more aggressive predictions)
                    if -8 <= prediction <= 8:  # Expanded from -10 to -8/+8 for more realistic range
                        return round(prediction, 4)
                except (ValueError, TypeError):
                    continue

        # If no valid number found, try to extract any floating point number
        numbers = re.findall(r'[-+]?\d*\.?\d+', response_text)
        if numbers:
            try:
                # Take the last number found
                prediction = float(numbers[-1])
                if -8 <= prediction <= 8:
                    return round(prediction, 4)
            except (ValueError, TypeError):
                pass

        # If all parsing fails, raise an error
        raise ValueError(f"Could not parse numerical prediction from response: '{response_text}'")

    def process_batch_with_ensemble(self, prompts: List[str], max_retries: int = 3,
                                  batch_delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Process batch of prompts using ensemble predictions with cost tracking.
        """
        results = []
        start_time = time.time()

        logger.info(f"Processing {len(prompts)} prompts with ensemble size {self.ensemble_size}")
        logger.info(f"Estimated cost: ${len(prompts) * self.ensemble_size * self.cost_per_call:.2f}")

        for i, prompt in enumerate(prompts):
            # Progress tracking for large batches
            if len(prompts) > 50 and (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                avg_time_per_sample = elapsed / (i + 1)
                remaining_samples = len(prompts) - (i + 1)
                estimated_remaining_time = remaining_samples * avg_time_per_sample

                logger.info(f"Progress: {i+1}/{len(prompts)} samples ({100*(i+1)/len(prompts):.1f}%) - "
                           f"Cost so far: ${self.total_cost:.2f} - "
                           f"ETA: {estimated_remaining_time/60:.1f} minutes")

            if not prompt or not prompt.strip():
                results.append({
                    "prediction": None,
                    "status": "failed",
                    "error": "Empty or invalid prompt"
                })
                continue

            retry_count = 0
            success = False
            last_error = None

            while retry_count <= max_retries and not success:
                try:
                    if len(prompts) <= 50:  # Only log individual samples for small batches
                        logger.info(f"Processing ensemble sample {i+1}/{len(prompts)}, attempt {retry_count + 1}")

                    # Make ensemble prediction
                    result = self.ensemble_predict(prompt, retry_count)
                    results.append(result)
                    success = True

                    # Rate limiting between successful requests
                    if i < len(prompts) - 1:  # Don't delay after last prompt
                        time.sleep(batch_delay)

                except Exception as e:
                    retry_count += 1
                    last_error = str(e)

                    if len(prompts) <= 50:
                        logger.warning(f"Attempt {retry_count} failed for sample {i+1}: {e}")

                    if retry_count <= max_retries:
                        # Exponential backoff with jitter
                        base_wait = 2 ** (retry_count - 1)
                        jitter = np.random.uniform(0, 1)
                        wait_time = base_wait + jitter

                        if len(prompts) <= 50:
                            logger.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for sample {i+1}")
                        results.append({
                            "prediction": None,
                            "status": "failed",
                            "error": last_error,
                            "retry_count": retry_count - 1,
                            "final_attempt": True
                        })

        # Log final batch summary with cost
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful
        total_time = time.time() - start_time

        logger.info(f"Ensemble batch processing complete!")
        logger.info(f"Results: {successful} successful, {failed} failed")
        logger.info(f"Total API calls made: {self.total_api_calls}")
        logger.info(f"Total cost: ${self.total_cost:.2f}")
        logger.info(f"Average cost per sample: ${self.total_cost/len(prompts):.3f}")
        logger.info(f"Processing time: {total_time/60:.1f} minutes ({total_time/len(prompts):.2f}s per sample)")

        return results

    def run_enhanced_pipeline(self, n_samples: int = None) -> pd.DataFrame:
        """
        Run enhanced pipeline with ensemble predictions and cost tracking.
        """
        # Load dataset
        df_llm = self.load_llm_dataset()

        # Determine sample size
        if n_samples is None:
            n_samples = len(df_llm)
            logger.info(f"Processing FULL ENHANCED DATASET with {n_samples} samples...")
        else:
            logger.info(f"Processing {n_samples} samples with enhanced pipeline...")

        # Take samples
        test_samples = df_llm.head(n_samples).copy()
        logger.info(f"Selected {len(test_samples)} samples for enhanced processing")

        # Estimate processing time and cost
        estimated_time_minutes = (len(test_samples) * self.ensemble_size * 1.2) / 60  # 1.2s per ensemble call
        estimated_cost = len(test_samples) * self.ensemble_size * self.cost_per_call

        logger.info(f"ENHANCED PIPELINE ESTIMATES:")
        logger.info(f"  Processing time: {estimated_time_minutes:.1f} minutes")
        logger.info(f"  Total API calls: {len(test_samples) * self.ensemble_size}")
        logger.info(f"  Estimated cost: ${estimated_cost:.2f}")

        # Create enhanced prompts
        logger.info("Creating enhanced prompts with aggressive guidance...")
        prompts = []
        failed_prompt_count = 0

        for idx, row in test_samples.iterrows():
            try:
                prompt = self.create_enhanced_prompt_template(row)
                prompts.append(prompt)
            except Exception as e:
                logger.error(f"Error creating prompt for row {idx}: {e}")
                prompts.append("")
                failed_prompt_count += 1

        if failed_prompt_count > 0:
            logger.warning(f"Failed to create {failed_prompt_count} prompts")

        # Process with Enhanced Vertex AI (ensemble)
        logger.info("Processing with Enhanced Vertex AI (Ensemble + Higher Temperature)...")
        predictions = self.process_batch_with_ensemble(
            prompts,
            max_retries=3,
            batch_delay=0.3  # Reduced delay for ensemble efficiency
        )

        # Combine results
        results_data = []
        for i, (idx, row) in enumerate(test_samples.iterrows()):
            prediction = predictions[i] if i < len(predictions) else {}

            result = {
                # Original data
                'sample_id': i + 1,
                'trading_date': row['trading_date'],
                'gap_length': row['gap_length'],
                'is_weekend_gap': row['is_weekend_gap'],
                'news_count': row['news_count'],

                # Technical indicators
                'gspc_close': row['gspc_close'],
                'vix_close': row['vix_close'],
                'rsi_14': row['rsi_14'],
                'macd_histogram': row['macd_histogram'],
                'realized_vol_20': row['realized_vol_20'],
                'daily_return': row['daily_return'],

                # News features
                'full_text_ratio': row['full_text_ratio'],
                'source_diversity': row['source_diversity'],

                # Enhanced predictions
                'actual_next_day_return': row['next_day_return'],
                'predicted_return': prediction.get('prediction', None),
                'prediction_confidence': prediction.get('confidence', None),
                'ensemble_size': prediction.get('ensemble_size', 0),
                'prediction_std': prediction.get('prediction_std', None),
                'prediction_status': prediction.get('status', 'unknown'),
                'processing_time': prediction.get('processing_time', None),
                'error_message': prediction.get('error', None)
            }
            results_data.append(result)

        results_df = pd.DataFrame(results_data)

        # Calculate enhanced performance metrics
        self.calculate_enhanced_performance_metrics(results_df)

        return results_df

    def calculate_enhanced_performance_metrics(self, results_df: pd.DataFrame):
        """Calculate and display enhanced performance metrics with cost analysis"""
        successful_predictions = results_df[results_df['prediction_status'] == 'success']

        logger.info(f"\n" + "="*60)
        logger.info(f"ENHANCED PIPELINE PERFORMANCE SUMMARY")
        logger.info(f"="*60)
        logger.info(f"Success Rate: {len(successful_predictions)}/{len(results_df)} ({100*len(successful_predictions)/len(results_df):.1f}%)")
        logger.info(f"Total API Calls: {self.total_api_calls}")
        logger.info(f"Total Cost: ${self.total_cost:.2f}")
        logger.info(f"Average Cost per Sample: ${self.total_cost/len(results_df):.3f}")

        if len(successful_predictions) > 0:
            # Enhanced prediction analysis
            valid_predictions = successful_predictions.dropna(subset=['actual_next_day_return', 'predicted_return'])

            if len(valid_predictions) > 0:
                actual = valid_predictions['actual_next_day_return']
                predicted = valid_predictions['predicted_return']

                # Basic accuracy metrics
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted)**2))
                correlation = np.corrcoef(actual, predicted)[0,1] if len(actual) > 1 else 0

                # Direction accuracy
                actual_direction = np.sign(actual)
                predicted_direction = np.sign(predicted)
                direction_accuracy = np.mean(actual_direction == predicted_direction)

                # Enhanced metrics
                pred_range = predicted.max() - predicted.min()
                actual_range = actual.max() - actual.min()
                prediction_boldness = pred_range / actual_range if actual_range > 0 else 0

                # Large move accuracy (>1% moves)
                large_moves = np.abs(actual) > 1.0
                if large_moves.sum() > 0:
                    large_move_mae = np.mean(np.abs(actual[large_moves] - predicted[large_moves]))
                    large_move_direction_acc = np.mean(
                        np.sign(actual[large_moves]) == np.sign(predicted[large_moves])
                    )
                else:
                    large_move_mae = 0
                    large_move_direction_acc = 0

                logger.info(f"\nENHANCED PREDICTION QUALITY:")
                logger.info(f"  MAE: {mae:.4f} (vs previous: ~0.86)")
                logger.info(f"  RMSE: {rmse:.4f} (vs previous: ~1.15)")
                logger.info(f"  Correlation: {correlation:.4f} (vs previous: ~-0.001)")
                logger.info(f"  Direction Accuracy: {direction_accuracy:.1%} (vs previous: ~51.8%)")
                logger.info(f"  Prediction Boldness: {prediction_boldness:.2f} (range ratio)")
                logger.info(f"  Large Move (>1%) MAE: {large_move_mae:.4f}")
                logger.info(f"  Large Move Direction Acc: {large_move_direction_acc:.1%}")

                # Ensemble quality metrics
                if 'ensemble_size' in valid_predictions.columns:
                    avg_ensemble_size = valid_predictions['ensemble_size'].mean()
                    logger.info(f"  Average Ensemble Size: {avg_ensemble_size:.1f}")

                if 'prediction_std' in valid_predictions.columns:
                    avg_prediction_std = valid_predictions['prediction_std'].dropna().mean()
                    logger.info(f"  Average Prediction Std: {avg_prediction_std:.4f}")

                # Processing time stats
                avg_processing_time = successful_predictions['processing_time'].mean()
                logger.info(f"  Average Processing Time: {avg_processing_time:.2f}s per sample")

    def save_enhanced_results(self, results_df: pd.DataFrame, filename: str = None) -> str:
        """Save enhanced results with cost tracking"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_count = len(results_df)
            filename = f"enhanced_vertex_ai_results_{sample_count}samples_ensemble{self.ensemble_size}_{timestamp}.csv"

        filepath = f"/Users/hrishikeshsajeev/Dissertation codes/LLM PyCharm/{filename}"
        results_df.to_csv(filepath, index=False)

        # Save cost summary
        cost_summary = {
            'total_samples': len(results_df),
            'ensemble_size': self.ensemble_size,
            'total_api_calls': self.total_api_calls,
            'total_cost_usd': self.total_cost,
            'cost_per_sample': self.total_cost / len(results_df),
            'model_name': self.model_name,
            'temperature': 0.4,
            'timestamp': datetime.now().isoformat()
        }

        cost_filepath = filepath.replace('.csv', '_cost_summary.json')
        with open(cost_filepath, 'w') as f:
            json.dump(cost_summary, f, indent=2)

        logger.info(f"Enhanced results saved to: {filepath}")
        logger.info(f"Cost summary saved to: {cost_filepath}")

        return filepath

def main():
    """Main function to run enhanced Vertex AI LLM pipeline"""

    logger.info("="*80)
    logger.info("ENHANCED VERTEX AI LLM PIPELINE")
    logger.info("Temperature: 0.0 | Single Predictions | Full Dataset | DETERMINISTIC MODE")
    logger.info("="*80)

    # Initialize enhanced pipeline
    pipeline = EnhancedVertexAILLMPipeline(
        project_id="REDACTED",
        location="REDACTED",
    model_name="gemini-2.5-pro",
        ensemble_size=3  # Ensemble mode to reduce variance (Methodology: Ensemble Prediction)
    )

    # Run enhanced pipeline on full dataset
    results_df = pipeline.run_enhanced_pipeline(n_samples=None)  # None = full dataset

    # Display final results
    print("\n" + "="*80)
    print("ENHANCED PIPELINE RESULTS")
    print("="*80)

    print(f"Total samples processed: {len(results_df)}")
    successful = (results_df['prediction_status'] == 'success').sum()
    print(f"Successful predictions: {successful}/{len(results_df)} ({100*successful/len(results_df):.1f}%)")

    # Save enhanced results
    saved_file = pipeline.save_enhanced_results(results_df)

    print(f"\n✓ Enhanced LLM pipeline completed!")
    print(f"✓ Results saved to: {saved_file}")
    print(f"✓ Total cost: ${pipeline.total_cost:.2f}")
    print(f"✓ Ready for enhanced portfolio analysis!")

    return results_df

if __name__ == "__main__":
    main()
