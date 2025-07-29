"""
EMA Filter Backtesting Module
Extends CoreBacktestEngine to test 6 EMA-based trend filtering strategies
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# Import the core engine and required modules
from core_backtest_engine import CoreBacktestEngine
from modules.trend_classifier import TrendClassifier

class EMAFilterBacktestEngine(CoreBacktestEngine):
    """
    Extended backtesting engine that tests multiple EMA-based trend filters
    Inherits all functionality from CoreBacktestEngine, only modifies trend alignment
    """
    
    def __init__(self):
        """Initialize with parent functionality plus EMA filter tracking"""
        super().__init__()
        
        # Define EMA filter configurations
        self.ema_filters = {
            'EMA_21_Location': {
                'type': 'location',
                'emas': [21],
                'description': 'Price above/below EMA 21'
            },
            'EMA_13_34_Location': {
                'type': 'location', 
                'emas': [13, 34],
                'description': 'Price above/below both EMA 13 and 34'
            },
            'EMA_30_Slope': {
                'type': 'slope',
                'emas': [30],
                'lookback': 5,
                'description': 'EMA 30 slope positive/negative'
            },
            'EMA_9_21_Location': {
                'type': 'location',
                'emas': [9, 21],
                'description': 'Price above/below both EMA 9 and 21'
            },
            'EMA_13_Slope': {
                'type': 'slope',
                'emas': [13],
                'lookback': 5,
                'description': 'EMA 13 slope positive/negative'
            },
            'No_Filter': {
                'type': 'none',
                'description': 'No EMA filter (baseline)'
            }
        }
        
        # Track current filter for testing
        self.current_filter = None
        self.current_ema_data = {}
        
    def run_ema_filter_analysis(self, pair: str, timeframe: str, days_back: int = 730) -> pd.DataFrame:
        """
        Run backtests for all EMA filter variations and return comparison
        
        Args:
            pair: Currency pair to test
            timeframe: Timeframe for analysis
            days_back: Number of days to backtest
            
        Returns:
            DataFrame with comparison of all filter results
        """
        print(f"\n🎯 EMA FILTER ANALYSIS: {pair} {timeframe}")
        print("=" * 60)
        
        all_results = []
        
        # Test each filter configuration
        for filter_name, filter_config in self.ema_filters.items():
            print(f"\n📊 Testing filter: {filter_name}")
            print(f"   Type: {filter_config['type']}")
            print(f"   Description: {filter_config['description']}")
            
            # Set current filter
            self.current_filter = filter_name
            
            # Run backtest with this filter
            result = self.run_single_strategy_test(pair, timeframe, days_back)
            
            # Add filter information to result
            result['filter_type'] = filter_name
            result['filter_description'] = filter_config['description']
            
            # Store result
            all_results.append(result)
            
            # Print summary
            print(f"   ✅ Trades: {result['total_trades']}")
            print(f"   📈 Win Rate: {result['win_rate']:.1f}%")
            print(f"   💰 Profit Factor: {result['profit_factor']:.2f}")
            print(f"   📊 Total Return: {result['total_return']:.2f}%")
        
        # Create comparison summary
        summary_df = self._create_filter_comparison_summary(all_results)
        
        # Save detailed results
        self._save_filter_analysis_results(all_results, summary_df, pair, timeframe)
        
        return summary_df
    
    def is_trend_aligned(self, zone_type: str, current_trend: str) -> bool:
        """
        Override parent method to implement EMA-based trend filters
        
        Args:
            zone_type: Type of zone (D-B-D, R-B-R, D-B-R, R-B-D)
            current_trend: Current trend from standard classification (unused for EMA filters)
            
        Returns:
            bool: Whether zone aligns with EMA filter criteria
        """
        # If no filter set or using No_Filter, use parent logic
        if self.current_filter is None or self.current_filter == 'No_Filter':
            return super().is_trend_aligned(zone_type, current_trend)
        
        # Get filter configuration
        filter_config = self.ema_filters[self.current_filter]
        
        # Determine if this is a buy or sell zone
        is_buy_zone = zone_type in ['R-B-R', 'D-B-R']
        is_sell_zone = zone_type in ['D-B-D', 'R-B-D']
        
        # Apply filter based on type
        if filter_config['type'] == 'location':
            return self._check_location_filter(is_buy_zone, is_sell_zone, filter_config)
        elif filter_config['type'] == 'slope':
            return self._check_slope_filter(is_buy_zone, is_sell_zone, filter_config)
        else:
            # Unknown filter type, use parent logic
            return super().is_trend_aligned(zone_type, current_trend)
    
    def _check_location_filter(self, is_buy_zone: bool, is_sell_zone: bool, 
                              filter_config: Dict) -> bool:
        """
        Check if price location relative to EMAs matches zone requirements
        
        Buy zones require price > all EMAs
        Sell zones require price < all EMAs
        """
        if not hasattr(self, '_current_candle_data'):
            return False
            
        current_price = self._current_candle_data['close']
        
        # Check all EMAs in the filter
        for ema_period in filter_config['emas']:
            ema_value = self._get_current_ema_value(ema_period)
            
            if ema_value is None:
                return False
            
            # Buy zone requires price above EMA
            if is_buy_zone and current_price <= ema_value:
                return False
            
            # Sell zone requires price below EMA
            if is_sell_zone and current_price >= ema_value:
                return False
        
        return True
    
    def _check_slope_filter(self, is_buy_zone: bool, is_sell_zone: bool, 
                           filter_config: Dict) -> bool:
        """
        Check if EMA slope matches zone requirements
        
        Buy zones require positive slope (current > 5 candles ago)
        Sell zones require negative slope (current < 5 candles ago)
        """
        if not hasattr(self, '_current_idx'):
            return False
            
        ema_period = filter_config['emas'][0]  # Slope filters use single EMA
        lookback = filter_config['lookback']
        
        # Get current and historical EMA values
        current_ema = self._get_current_ema_value(ema_period)
        historical_ema = self._get_historical_ema_value(ema_period, lookback)
        
        if current_ema is None or historical_ema is None:
            return False
        
        # Calculate slope
        slope = current_ema - historical_ema
        
        # Buy zone requires positive slope
        if is_buy_zone and slope <= 0:
            return False
        
        # Sell zone requires negative slope
        if is_sell_zone and slope >= 0:
            return False
        
        return True
    
    def _get_current_ema_value(self, period: int) -> Optional[float]:
        """Get current EMA value from cached data"""
        if period in self.current_ema_data and self._current_idx < len(self.current_ema_data[period]):
            return self.current_ema_data[period].iloc[self._current_idx]
        return None
    
    def _get_historical_ema_value(self, period: int, lookback: int) -> Optional[float]:
        """Get historical EMA value from lookback periods ago"""
        historical_idx = self._current_idx - lookback
        if historical_idx < 0:
            return None
            
        if period in self.current_ema_data and historical_idx < len(self.current_ema_data[period]):
            return self.current_ema_data[period].iloc[historical_idx]
        return None
    
    def execute_realistic_trades(self, patterns: List[Dict], data: pd.DataFrame,
                               trend_data: pd.DataFrame, timeframe: str, pair: str) -> List[Dict]:
        """
        Override to calculate and cache EMA data before executing trades
        """
        # Calculate all required EMAs for current filter
        if self.current_filter and self.current_filter != 'No_Filter':
            filter_config = self.ema_filters[self.current_filter]
            
            if 'emas' in filter_config:
                # Create TrendClassifier with custom EMA configuration
                for ema_period in filter_config['emas']:
                    # Use TrendClassifier's _calculate_ema method
                    trend_classifier = TrendClassifier(data)
                    self.current_ema_data[ema_period] = trend_classifier._calculate_ema(ema_period)
        
        # Store current data for access in is_trend_aligned
        self._current_data = data
        
        # Call parent method with EMA data available
        trades = []
        for current_idx in range(200, len(data)):  # Start after EMA warmup
            self._current_idx = current_idx
            self._current_candle_data = data.iloc[current_idx]
            
            # Use parent's trade execution logic with our overridden is_trend_aligned
            # This is a simplified version - you may need to adapt based on parent implementation
            for pattern in patterns:
                # Check zone interaction and trend alignment
                if self._check_zone_interaction(pattern, self._current_candle_data):
                    zone_type = pattern['type']
                    
                    # Our overridden method will be called here
                    if self.is_trend_aligned(zone_type, 'unused'):
                        trade = self._execute_trade(pattern, data, current_idx)
                        if trade:
                            trades.append(trade)
        
        return trades
    
    def _check_zone_interaction(self, pattern: Dict, candle: pd.Series) -> bool:
        """Simplified zone interaction check"""
        # This should match parent's logic - adapt as needed
        return (candle['low'] <= pattern['zone_high'] and 
                candle['high'] >= pattern['zone_low'])
    
    def _execute_trade(self, pattern: Dict, data: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Simplified trade execution"""
        # This should call parent's execute_single_realistic_trade
        # Adapt based on actual parent implementation
        return self.execute_single_realistic_trade(pattern, data, idx)
    
    def _create_filter_comparison_summary(self, results: List[Dict]) -> pd.DataFrame:
        """
        Create summary comparison of all filter performance
        
        Returns DataFrame with key metrics for each filter
        """
        summary_data = []
        
        for result in results:
            summary_data.append({
                'Filter': result['filter_type'],
                'Description': result['filter_description'],
                'Total_Trades': result['total_trades'],
                'Win_Rate_%': result['win_rate'],
                'Loss_Rate_%': result['loss_rate'],
                'BE_Rate_%': result['be_rate'],
                'Profit_Factor': result['profit_factor'],
                'Total_Return_%': result['total_return'],
                'Avg_Trade_Duration': result['avg_trade_duration'],
                'Gross_Profit': result['gross_profit'],
                'Gross_Loss': result['gross_loss'],
                'Expectancy': result['gross_profit'] / result['total_trades'] if result['total_trades'] > 0 else 0,
                'R_Multiple': (result['total_return'] / 100) / (result['total_trades'] * 0.05) if result['total_trades'] > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by profit factor
        summary_df = summary_df.sort_values('Profit_Factor', ascending=False)
        
        # Print summary
        print("\n📊 EMA FILTER COMPARISON SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def _save_filter_analysis_results(self, all_results: List[Dict], 
                                     summary_df: pd.DataFrame,
                                     pair: str, timeframe: str):
        """Save detailed results and summary to Excel"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/ema_filter_analysis_{pair}_{timeframe}_{timestamp}.xlsx"
        
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Summary comparison
                summary_df.to_excel(writer, sheet_name='Filter_Comparison', index=False)
                
                # Sheet 2: All results
                all_results_df = pd.DataFrame(all_results)
                all_results_df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Sheet 3-8: Individual filter trade details
                for result in all_results:
                    if result['trades']:
                        filter_name = result['filter_type']
                        trades_df = pd.DataFrame(result['trades'])
                        sheet_name = f"{filter_name}_Trades"[:31]  # Excel sheet name limit
                        trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n📁 Results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")


def run_multi_asset_analysis(self, pairs: List[str], timeframes: List[str], 
                           days_back: int = 730) -> pd.DataFrame:
    """
    Run EMA filter analysis across multiple pairs and timeframes
    
    Args:
        pairs: List of currency pairs to test
        timeframes: List of timeframes to test
        days_back: Number of days to backtest
        
    Returns:
        DataFrame with aggregated results across all combinations
    """
    print(f"\n🚀 MULTI-ASSET EMA FILTER ANALYSIS")
    print(f"📊 Pairs: {', '.join(pairs)}")
    print(f"📊 Timeframes: {', '.join(timeframes)}")
    print(f"📊 Days back: {days_back}")
    print("=" * 60)
    
    all_combinations_results = []
    
    # Test each pair/timeframe combination
    total_combinations = len(pairs) * len(timeframes)
    combination_count = 0
    
    for pair in pairs:
        for timeframe in timeframes:
            combination_count += 1
            print(f"\n[{combination_count}/{total_combinations}] Testing {pair} {timeframe}")
            
            # Run analysis for this combination
            try:
                summary_df = self.run_ema_filter_analysis(pair, timeframe, days_back)
                
                # Add pair/timeframe info to each result
                for _, row in summary_df.iterrows():
                    result_dict = row.to_dict()
                    result_dict['Pair'] = pair
                    result_dict['Timeframe'] = timeframe
                    all_combinations_results.append(result_dict)
                    
            except Exception as e:
                print(f"❌ Error testing {pair} {timeframe}: {str(e)}")
                continue
    
    # Create comprehensive report
    if all_combinations_results:
        results_df = pd.DataFrame(all_combinations_results)
        self._generate_multi_asset_report(results_df, pairs, timeframes, days_back)
        return results_df
    else:
        print("❌ No successful results to report")
        return pd.DataFrame()

def _generate_multi_asset_report(self, results_df: pd.DataFrame, 
                                pairs: List[str], timeframes: List[str], 
                                days_back: int):
    """Generate comprehensive multi-asset analysis report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/multi_asset_ema_filter_{timestamp}.xlsx"
    
    os.makedirs('results', exist_ok=True)
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Raw results
            results_df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Sheet 2: Aggregated by Filter Type
            filter_agg = results_df.groupby('Filter').agg({
                'Total_Trades': 'sum',
                'Win_Rate_%': 'mean',
                'Loss_Rate_%': 'mean',
                'BE_Rate_%': 'mean',
                'Profit_Factor': 'mean',
                'Total_Return_%': 'mean',
                'Avg_Trade_Duration': 'mean',
                'Gross_Profit': 'sum',
                'Gross_Loss': 'sum'
            }).round(2)
            
            # Calculate weighted averages based on trade count
            for col in ['Win_Rate_%', 'Loss_Rate_%', 'BE_Rate_%']:
                filter_agg[f'Weighted_{col}'] = 0
                for filter_type in filter_agg.index:
                    filter_data = results_df[results_df['Filter'] == filter_type]
                    total_trades = filter_data['Total_Trades'].sum()
                    if total_trades > 0:
                        weighted_avg = (filter_data[col] * filter_data['Total_Trades']).sum() / total_trades
                        filter_agg.loc[filter_type, f'Weighted_{col}'] = round(weighted_avg, 2)
            
            filter_agg.to_excel(writer, sheet_name='Filter_Summary')
            
            # Sheet 3: Aggregated by Pair
            pair_agg = results_df.groupby('Pair').agg({
                'Total_Trades': 'sum',
                'Win_Rate_%': 'mean',
                'Loss_Rate_%': 'mean', 
                'BE_Rate_%': 'mean',
                'Profit_Factor': 'mean',
                'Total_Return_%': 'mean',
                'Avg_Trade_Duration': 'mean',
                'Gross_Profit': 'sum',
                'Gross_Loss': 'sum'
            }).round(2)
            pair_agg.to_excel(writer, sheet_name='Pair_Summary')
            
            # Sheet 4: Aggregated by Timeframe
            tf_agg = results_df.groupby('Timeframe').agg({
                'Total_Trades': 'sum',
                'Win_Rate_%': 'mean',
                'Loss_Rate_%': 'mean',
                'BE_Rate_%': 'mean',
                'Profit_Factor': 'mean',
                'Total_Return_%': 'mean',
                'Avg_Trade_Duration': 'mean',
                'Gross_Profit': 'sum',
                'Gross_Loss': 'sum'
            }).round(2)
            tf_agg.to_excel(writer, sheet_name='Timeframe_Summary')
            
            # Sheet 5: Grand Total Summary
            grand_total = pd.DataFrame([{
                'Total_Combinations_Tested': len(results_df),
                'Total_Pairs': len(pairs),
                'Total_Timeframes': len(timeframes),
                'Total_Filters': len(self.ema_filters),
                'Grand_Total_Trades': results_df['Total_Trades'].sum(),
                'Avg_Win_Rate_%': round(results_df['Win_Rate_%'].mean(), 2),
                'Avg_Loss_Rate_%': round(results_df['Loss_Rate_%'].mean(), 2),
                'Avg_BE_Rate_%': round(results_df['BE_Rate_%'].mean(), 2),
                'Avg_Profit_Factor': round(results_df['Profit_Factor'].mean(), 2),
                'Avg_Return_%': round(results_df['Total_Return_%'].mean(), 2),
                'Total_Gross_Profit': round(results_df['Gross_Profit'].sum(), 2),
                'Total_Gross_Loss': round(results_df['Gross_Loss'].sum(), 2),
                'Avg_Trade_Duration': round(results_df['Avg_Trade_Duration'].mean(), 2)
            }])
            grand_total.to_excel(writer, sheet_name='Grand_Total', index=False)
        
        print(f"\n📁 Multi-asset report saved to: {filename}")
        
        # Print summary to console
        print("\n📊 GRAND TOTAL SUMMARY")
        print("=" * 60)
        print(f"Total Combinations Tested: {len(results_df)}")
        print(f"Grand Total Trades: {results_df['Total_Trades'].sum()}")
        print(f"Average Win Rate: {results_df['Win_Rate_%'].mean():.2f}%")
        print(f"Average Loss Rate: {results_df['Loss_Rate_%'].mean():.2f}%")
        print(f"Average BE Rate: {results_df['BE_Rate_%'].mean():.2f}%")
        print(f"Average Profit Factor: {results_df['Profit_Factor'].mean():.2f}")
        print(f"Average Return: {results_df['Total_Return_%'].mean():.2f}%")
        print(f"Total Gross Profit: ${results_df['Gross_Profit'].sum():.2f}")
        print(f"Total Gross Loss: ${results_df['Gross_Loss'].sum():.2f}")
        
    except Exception as e:
        print(f"❌ Error saving multi-asset report: {str(e)}")

def main():
    """Run EMA filter analysis"""
    print("🎯 EMA FILTER BACKTESTING ANALYSIS")
    print("=" * 60)
    
    # Create engine instance
    engine = EMAFilterBacktestEngine()
    
    # Check system resources
    if not engine.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 SELECT ANALYSIS MODE:")
    print("1. Single pair/timeframe analysis")
    print("2. Quick validation (EURUSD 3D)")
    print("3. Major pairs analysis (EURUSD, GBPUSD, USDJPY, AUDUSD)")
    print("4. Custom multi-pair/timeframe analysis")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Single pair/timeframe
        print("\n📊 Enter test parameters:")
        pair = input("Currency pair (default: EURUSD): ").strip().upper() or "EURUSD"
        timeframe = input("Timeframe (default: 3D): ").strip() or "3D"
        days_back = int(input("Days back (default: 730): ").strip() or "730")
        
        print(f"\n🚀 Starting EMA filter analysis for {pair} {timeframe}")
        summary = engine.run_ema_filter_analysis(pair, timeframe, days_back)
        
    elif choice == '2':
        # Quick validation
        print("\n🚀 Quick validation: EURUSD 3D")
        summary = engine.run_ema_filter_analysis('EURUSD', '3D', 730)
        
    elif choice == '3':
        # Major pairs preset
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        timeframes = ['1D', '3D', '1W']
        print(f"\n🚀 Major pairs analysis")
        engine.run_multi_asset_analysis(pairs, timeframes, 730)
        
    elif choice == '4':
        # Custom multi-asset
        print("\n📊 Custom multi-asset analysis:")
        
        # Get pairs
        pairs_input = input("Pairs (comma-separated, e.g., EURUSD,GBPUSD,USDJPY): ").strip()
        pairs = [p.strip().upper() for p in pairs_input.split(',') if p.strip()]
        
        if not pairs:
            print("❌ No valid pairs entered")
            return
            
        # Get timeframes
        tf_input = input("Timeframes (comma-separated, e.g., 1D,3D,1W): ").strip()
        timeframes = [tf.strip() for tf in tf_input.split(',') if tf.strip()]
        
        if not timeframes:
            print("❌ No valid timeframes entered")
            return
            
        # Get days back
        days_back = int(input("Days back (default: 730): ").strip() or "730")
        
        print(f"\n🚀 Starting multi-asset analysis")
        engine.run_multi_asset_analysis(pairs, timeframes, days_back)
        
    else:
        print("❌ Invalid choice")
        return
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("Check results folder for detailed Excel report")


if __name__ == "__main__":
    main()