from llm_portfolio_strategy import LLMPortfolioStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class EnhancedLLMPortfolioStrategy(LLMPortfolioStrategy):
    """
    Enhanced LLM Portfolio Strategy using ensemble predictions with temperature increase.
    """
    
    def __init__(self):
        super().__init__()
        self.cost_tracking = {}
        
    def load_enhanced_predictions(self, filepath):
        """Load enhanced LLM predictions with ensemble results"""
        print("Loading enhanced LLM predictions...")
        df = pd.read_csv(filepath)
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        df = df.set_index('trading_date')
        
        print(f"Loaded {len(df)} enhanced predictions")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Load cost summary if available
        cost_file = filepath.replace('.csv', '_cost_summary.json')
        try:
            with open(cost_file, 'r') as f:
                self.cost_tracking = json.load(f)
            print(f"Cost tracking loaded: ${self.cost_tracking.get('total_cost_usd', 0):.2f}")
        except FileNotFoundError:
            print("No cost summary found")
        
        # Add derived features
        df = self._add_enhanced_derived_features(df)
        
        return df
    
    def _add_enhanced_derived_features(self, df):
        """Add enhanced derived features for better analysis"""
        df = df.copy()
        
        # Enhanced prediction confidence based on ensemble metrics
        if 'prediction_std' in df.columns:
            df['enhanced_confidence'] = 1.0 / (1.0 + df['prediction_std'].fillna(1.0))
        else:
            df['enhanced_confidence'] = df['prediction_confidence'].fillna(0.8)
        
        # Prediction boldness metric
        df['prediction_magnitude'] = np.abs(df['predicted_return'])
        df['is_bold_prediction'] = (df['prediction_magnitude'] > df['prediction_magnitude'].quantile(0.7)).astype(int)
        
        # Enhanced error and direction metrics
        df['prediction_error'] = abs(df['predicted_return'] - df['actual_next_day_return'])
        df['predicted_direction'] = np.sign(df['predicted_return'])
        df['actual_direction'] = np.sign(df['actual_next_day_return'])
        df['correct_direction'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
        
        # Large move analysis (>1% moves)
        df['large_actual_move'] = (np.abs(df['actual_next_day_return']) > 1.0).astype(int)
        df['large_predicted_move'] = (np.abs(df['predicted_return']) > 1.0).astype(int)
        
        # Volatility and regime features
        df['return_volatility'] = df['daily_return'].rolling(20).std()
        df['regime'] = (df['vix_close'] > df['vix_close'].rolling(60).quantile(0.7)).astype(int)
        
        return df
    
    def run_enhanced_threshold_optimization(self, df):
        """
        Enhanced threshold optimization with focus on ensemble prediction quality
        """
        print("\n" + "=" * 80)
        print("ENHANCED LLM PORTFOLIO THRESHOLD OPTIMIZATION")
        print("Using Ensemble Predictions with Temperature 0.4")
        print("=" * 80)
        
        # Enhanced threshold ranges - test higher thresholds for better ensemble predictions
        return_thresholds = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]  # Extended range
        confidence_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]  # More granular
        
        best_sharpe = -np.inf
        best_params = {}
        results_summary = []
        
        for ret_thresh in return_thresholds:
            for conf_thresh in confidence_thresholds:
                print(f"\nTesting thresholds: Return={ret_thresh:.3f}, Confidence={conf_thresh:.3f}")
                
                # Set thresholds
                self.return_threshold = ret_thresh
                self.confidence_threshold = conf_thresh
                
                # Generate signals using enhanced confidence
                signals_df = self.generate_enhanced_trading_signals(df)
                
                if len(signals_df) > 0 and signals_df['trading_signal'].notna().sum() > 0:
                    # Calculate performance
                    performance = self.calculate_enhanced_portfolio_performance(signals_df)
                    
                    results_summary.append({
                        'return_threshold': ret_thresh,
                        'confidence_threshold': conf_thresh,
                        'sharpe_ratio': performance['sharpe_ratio'],
                        'total_return': performance['total_return'],
                        'max_drawdown': performance['max_drawdown'],
                        'win_rate': performance['win_rate'],
                        'total_trades': performance['total_trades'],
                        'prediction_accuracy': performance['prediction_accuracy'],
                        'large_move_accuracy': performance.get('large_move_accuracy', 0)
                    })
                    
                    if performance['sharpe_ratio'] > best_sharpe:
                        best_sharpe = performance['sharpe_ratio']
                        best_params = {
                            'return_threshold': ret_thresh,
                            'confidence_threshold': conf_thresh,
                            'performance': performance
                        }
                    
                    print(f"  Sharpe: {performance['sharpe_ratio']:.3f} | Return: {performance['total_return']:.3f} | Trades: {performance['total_trades']}")
        
        # Export enhanced results
        if results_summary:
            df_results = pd.DataFrame(results_summary)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_llm_threshold_optimization_{timestamp}.csv"
            df_results.to_csv(f"/Users/hrishikeshsajeev/Dissertation codes/LLM PyCharm/{filename}", index=False)
            print(f"\nEnhanced threshold optimization results exported to: {filename}")
        
        # Display optimal results
        print(f"\n" + "=" * 60)
        print("ENHANCED OPTIMAL THRESHOLD RESULTS")
        print("=" * 60)
        print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
        print(f"Optimal Return Threshold: {best_params['return_threshold']:.4f}")
        print(f"Optimal Confidence Threshold: {best_params['confidence_threshold']:.4f}")
        
        if 'performance' in best_params:
            perf = best_params['performance']
            print(f"\nOptimal Enhanced Strategy Performance:")
            print(f"  Total Return: {perf['total_return']:.4f}")
            print(f"  Annualized Return: {perf['annualized_return']:.4f}")
            print(f"  Volatility: {perf['portfolio_volatility']:.4f}")
            print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
            print(f"  Maximum Drawdown: {perf['max_drawdown']:.4f}")
            print(f"  Win Rate: {perf['win_rate']:.4f}")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Large Move Accuracy: {perf.get('large_move_accuracy', 0):.4f}")
        
        # Set optimal thresholds
        self.return_threshold = best_params['return_threshold']
        self.confidence_threshold = best_params['confidence_threshold']
        self.optimal_thresholds = best_params
        
        return best_params, results_summary
    
    def generate_enhanced_trading_signals(self, df):
        """
        Generate trading signals using enhanced ensemble predictions
        """
        df = df.copy()
        
        # Use enhanced confidence from ensemble predictions
        confidence_proxy = df['enhanced_confidence'].fillna(df['prediction_confidence'].fillna(0.5))
        
        # Generate signals using dual-threshold system
        long_condition = (df['predicted_return'] > self.return_threshold) & \
                        (confidence_proxy > self.confidence_threshold)
        
        short_condition = (df['predicted_return'] < -self.return_threshold) & \
                         (confidence_proxy > self.confidence_threshold)
        
        # Create trading signals
        df['trading_signal'] = 0  # Default to neutral
        df.loc[long_condition, 'trading_signal'] = 1   # Long
        df.loc[short_condition, 'trading_signal'] = -1  # Short
        
        # Enhanced threshold indicators
        df['return_threshold_met'] = (np.abs(df['predicted_return']) > self.return_threshold).astype(int)
        df['confidence_threshold_met'] = (confidence_proxy > self.confidence_threshold).astype(int)
        
        # Calculate portfolio returns
        df['actual_return'] = df['actual_next_day_return']
        df['portfolio_return'] = df['trading_signal'] * df['actual_return']
        df['profitable_trade'] = (df['portfolio_return'] > 0).astype(int)
        
        # Store enhanced signals
        self.trading_signals = df.copy()
        
        return df
    
    def calculate_enhanced_portfolio_performance(self, signals_df):
        """
        Calculate enhanced portfolio performance with additional metrics
        """
        # Get base performance
        performance = super().calculate_portfolio_performance(signals_df)
        
        # Add enhanced metrics
        active_signals = signals_df[signals_df['trading_signal'] != 0].copy()
        
        if len(active_signals) > 0:
            # Large move performance (>1% actual moves)
            large_moves = active_signals[active_signals['large_actual_move'] == 1]
            if len(large_moves) > 0:
                large_move_accuracy = large_moves['correct_direction'].mean()
                large_move_returns = large_moves['portfolio_return'].mean()
                performance['large_move_accuracy'] = large_move_accuracy
                performance['large_move_avg_return'] = large_move_returns
            
            # Bold prediction performance
            bold_predictions = active_signals[active_signals['is_bold_prediction'] == 1]
            if len(bold_predictions) > 0:
                bold_accuracy = bold_predictions['correct_direction'].mean()
                bold_returns = bold_predictions['portfolio_return'].mean()
                performance['bold_prediction_accuracy'] = bold_accuracy
                performance['bold_prediction_avg_return'] = bold_returns
            
            # Enhanced confidence analysis
            if 'enhanced_confidence' in active_signals.columns:
                high_confidence = active_signals[active_signals['enhanced_confidence'] > 0.8]
                if len(high_confidence) > 0:
                    high_conf_accuracy = high_confidence['correct_direction'].mean()
                    high_conf_returns = high_confidence['portfolio_return'].mean()
                    performance['high_confidence_accuracy'] = high_conf_accuracy
                    performance['high_confidence_avg_return'] = high_conf_returns
        
        return performance
    
    def create_enhanced_performance_comparison(self):
        """Create comparison with original LLM performance"""
        if not hasattr(self, 'optimal_thresholds') or 'performance' not in self.optimal_thresholds:
            print("Run enhanced optimization first")
            return
        
        enhanced_perf = self.optimal_thresholds['performance']
        
        print("\n" + "=" * 80)
        print("ENHANCED vs ORIGINAL LLM PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Original performance (from previous analysis)
        original_perf = {
            'sharpe_ratio': 0.1539,
            'total_return': 9.9936,
            'annualized_return': 2.5593,
            'win_rate': 0.5183,
            'total_trades': 984
        }
        
        print(f"{'Metric':<25} {'Original':<12} {'Enhanced':<12} {'Improvement':<15}")
        print("-" * 65)
        
        metrics = ['sharpe_ratio', 'total_return', 'annualized_return', 'win_rate']
        for metric in metrics:
            orig_val = original_perf.get(metric, 0)
            enh_val = enhanced_perf.get(metric, 0)
            
            if orig_val != 0:
                improvement = ((enh_val - orig_val) / abs(orig_val)) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric:<25} {orig_val:<12.4f} {enh_val:<12.4f} {improvement_str:<15}")
        
        # Cost analysis
        if self.cost_tracking:
            print(f"\nCOST ANALYSIS:")
            print(f"Total Cost: ${self.cost_tracking.get('total_cost_usd', 0):.2f}")
            print(f"Cost per Sample: ${self.cost_tracking.get('cost_per_sample', 0):.4f}")
            print(f"Ensemble Size: {self.cost_tracking.get('ensemble_size', 1)}")
        
        # Performance improvement summary
        sharpe_improvement = ((enhanced_perf['sharpe_ratio'] - original_perf['sharpe_ratio']) / 
                             original_perf['sharpe_ratio']) * 100
        
        print(f"\n🎯 KEY IMPROVEMENT:")
        print(f"Sharpe Ratio: {original_perf['sharpe_ratio']:.4f} → {enhanced_perf['sharpe_ratio']:.4f}")
        print(f"Improvement: {sharpe_improvement:+.1f}%")
        
        if enhanced_perf['sharpe_ratio'] > 0.5:  # Arbitrary threshold for "good"
            print(" ENHANCED LLM now shows competitive performance!")
        else:
            print("  Further improvements needed to beat Traditional ML")

def main():
    """Main function to run enhanced LLM portfolio analysis"""
    print("=" * 80)
    print("ENHANCED LLM PORTFOLIO STRATEGY ANALYSIS")
    print("Testing Ensemble 3 Results - 20250819_175339")
    print("=" * 80)
    
    # Initialize enhanced strategy
    strategy = EnhancedLLMPortfolioStrategy()
    
    # Use the specific test file requested by user
    predictions_file = "/Users/hrishikeshsajeev/Dissertation codes/LLM PyCharm/enhanced_vertex_ai_results_984samples_ensemble3_20250819_175339.csv"
    
    try:
        print(f"Using ensemble predictions: {predictions_file}")
        
        # Verify file exists
        import os
        if not os.path.exists(predictions_file):
            print("Predictions file not found. Please check the file path!")
            return
        
        # Load enhanced predictions
        df = strategy.load_enhanced_predictions(predictions_file)
        
        # Run enhanced threshold optimization
        best_params, threshold_results = strategy.run_enhanced_threshold_optimization(df)
        
        # Generate final enhanced signals
        final_signals = strategy.generate_enhanced_trading_signals(df)
        
        # Create performance comparison
        strategy.create_enhanced_performance_comparison()
        
        # Save enhanced results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        enhanced_results_file = f"enhanced_llm_portfolio_results_{timestamp}.csv"
        final_signals.to_csv(f"/Users/hrishikeshsajeev/Dissertation codes/LLM PyCharm/{enhanced_results_file}")
        
        print(f"\n✓ Enhanced LLM portfolio analysis complete!")
        print(f"✓ Results saved to: {enhanced_results_file}")
        print(f"✓ Ready for comparison with Traditional ML!")
        
        return final_signals, best_params
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please run enhanced_vertex_ai_pipeline.py first to generate enhanced predictions")
        return None, None

if __name__ == "__main__":
    main()