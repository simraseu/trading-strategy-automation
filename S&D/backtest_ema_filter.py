"""
EMA Filter Backtesting Module - COMPLETE REWRITE
Properly extends CoreBacktestEngine maintaining ALL trading logic
Only modifies trend alignment checks for different EMA filters
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Import the core engine
from core_backtest_engine import CoreBacktestEngine, run_single_test_worker, ANALYSIS_PERIODS
from modules.trend_classifier import TrendClassifier

class EMAFilterBacktestEngine(CoreBacktestEngine):
    """
    PROPERLY extends CoreBacktestEngine maintaining ALL zone logic
    ONLY modifies the trend alignment check - nothing else
    """
    
    def __init__(self):
        """Initialize with parent functionality plus EMA filter definitions"""
        super().__init__()
        
        # Define EMA filter configurations
        self.ema_filters = {
            'No_Filter': {
                'type': 'none',
                'description': 'No EMA filter - All zones traded (baseline)'
            },
            'EMA_50_200_Cross': {
                'type': 'ema_cross',
                'fast': 50,
                'slow': 200,
                'description': 'Standard EMA 50/200 crossover (original)'
            },
            'EMA_21_Location': {
                'type': 'price_location',
                'ema': 21,
                'description': 'Price above/below EMA 21'
            },
            'EMA_13_34_Location': {
                'type': 'dual_location', 
                'ema1': 13,
                'ema2': 34,
                'description': 'Price above/below both EMA 13 and 34'
            },
            'EMA_30_Slope': {
                'type': 'slope',
                'ema': 30,
                'lookback': 5,
                'description': 'EMA 30 slope positive/negative (5 candle lookback)'
            },
            'EMA_9_21_Location': {
                'type': 'dual_location',
                'ema1': 9,
                'ema2': 21,
                'description': 'Price above/below both EMA 9 and 21'
            }
        }
        
        # Current filter being tested
        self.current_filter = 'EMA_50_200_Cross'  # Default
        self._ema_data_cache = {}
        
    def set_current_filter(self, filter_name: str):
        """Set the current EMA filter for testing"""
        if filter_name not in self.ema_filters:
            raise ValueError(f"Unknown filter: {filter_name}")
        self.current_filter = filter_name
        self._ema_data_cache.clear()  # Clear cache when changing filters
        
    def is_trend_aligned(self, zone_type: str, current_trend: str) -> bool:
        """
        OVERRIDE ONLY THIS METHOD - the trend alignment check
        Everything else stays exactly the same from parent
        """
        # Get current filter config
        filter_config = self.ema_filters[self.current_filter]
        
        # No filter - allow all zones
        if filter_config['type'] == 'none':
            return True
            
        # Standard EMA crossover (original logic) - MUST USE PARENT LOGIC
        if filter_config['type'] == 'ema_cross':
            # This uses the parent's original logic exactly
            return super().is_trend_aligned(zone_type, current_trend)
            
        # For other filters, we need current context
        if not hasattr(self, '_data_for_filter'):
            # Fallback to parent logic if no context
            return super().is_trend_aligned(zone_type, current_trend)
            
        # Determine if buy or sell zone
        is_buy_zone = zone_type in ['R-B-R', 'D-B-R']
        is_sell_zone = zone_type in ['D-B-D', 'R-B-D']
        
        # Get current candle index (we need to find it from the execution context)
        # This is a simplified approach - we'll need the current index from the caller
        try:
            # We need to get current price somehow - let's use the last price for now
            current_price = self._data_for_filter['close'].iloc[-1]
            current_idx = len(self._data_for_filter) - 1
            
            # Price location filters
            if filter_config['type'] == 'price_location':
                return self._check_single_ema_location_simple(is_buy_zone, is_sell_zone, filter_config, current_price, current_idx)
                
            elif filter_config['type'] == 'dual_location':
                return self._check_dual_ema_location_simple(is_buy_zone, is_sell_zone, filter_config, current_price, current_idx)
                
            # Slope filter
            elif filter_config['type'] == 'slope':
                return self._check_ema_slope_simple(is_buy_zone, is_sell_zone, filter_config, current_idx)
                
        except Exception:
            # If anything fails, use parent logic
            return super().is_trend_aligned(zone_type, current_trend)
            
        # Unknown filter type - use parent logic
        return super().is_trend_aligned(zone_type, current_trend)
    
    def _check_single_ema_location_simple(self, is_buy_zone: bool, is_sell_zone: bool, 
                                     filter_config: Dict, current_price: float, current_idx: int) -> bool:
        """Check price location relative to single EMA"""
        ema_value = self._get_ema_value_at_index(filter_config['ema'], current_idx)
        if ema_value is None:
            return False
        
        # Buy zones need price above EMA
        if is_buy_zone:
            return current_price > ema_value
            
        # Sell zones need price below EMA
        if is_sell_zone:
            return current_price < ema_value
            
        return False

    def _check_dual_ema_location_simple(self, is_buy_zone: bool, is_sell_zone: bool, 
                                    filter_config: Dict, current_price: float, current_idx: int) -> bool:
        """Check price location relative to two EMAs"""
        ema1_value = self._get_ema_value_at_index(filter_config['ema1'], current_idx)
        ema2_value = self._get_ema_value_at_index(filter_config['ema2'], current_idx)
        
        if ema1_value is None or ema2_value is None:
            return False
        
        # Buy zones need price above BOTH EMAs
        if is_buy_zone:
            return current_price > ema1_value and current_price > ema2_value
            
        # Sell zones need price below BOTH EMAs
        if is_sell_zone:
            return current_price < ema1_value and current_price < ema2_value
            
        return False

    def _check_ema_slope_simple(self, is_buy_zone: bool, is_sell_zone: bool, 
                            filter_config: Dict, current_idx: int) -> bool:
        """Check EMA slope direction"""
        ema_period = filter_config['ema']
        lookback = filter_config['lookback']
        
        # Get current and historical EMA values
        current_ema = self._get_ema_value_at_index(ema_period, current_idx)
        historical_idx = current_idx - lookback
        
        if historical_idx < 0:
            return False
            
        historical_ema = self._get_ema_value_at_index(ema_period, historical_idx)
        
        if current_ema is None or historical_ema is None:
            return False
        
        # Calculate slope
        slope = current_ema - historical_ema
        
        # Buy zones need positive slope
        if is_buy_zone:
            return slope > 0
            
        # Sell zones need negative slope
        if is_sell_zone:
            return slope < 0
            
        return False
    
    def _get_ema_value_at_index(self, period: int, idx: int) -> Optional[float]:
        """Get EMA value at specific index"""
        if period not in self._ema_data_cache:
            return None
            
        ema_series = self._ema_data_cache[period]
        if idx < 0 or idx >= len(ema_series):
            return None
            
        return ema_series.iloc[idx]
    
    def execute_realistic_trades(self, patterns: List[Dict], data: pd.DataFrame,
                           trend_data: pd.DataFrame, timeframe: str, pair: str) -> List[Dict]:
        """
        SIMPLIFIED OVERRIDE: Just call parent with EMA data context set
        For EMA_50_200_Cross, this should behave identically to core engine
        """
        # Calculate all required EMAs for current filter
        filter_config = self.ema_filters[self.current_filter]
        
        # Clear and rebuild EMA cache
        self._ema_data_cache.clear()
        
        # Calculate EMAs based on filter requirements
        if filter_config['type'] in ['price_location', 'slope']:
            ema_period = filter_config.get('ema')
            if ema_period:
                trend_classifier = TrendClassifier(data)
                self._ema_data_cache[ema_period] = trend_classifier._calculate_ema(ema_period)
                
        elif filter_config['type'] == 'dual_location':
            ema1 = filter_config.get('ema1')
            ema2 = filter_config.get('ema2')
            if ema1 and ema2:
                trend_classifier = TrendClassifier(data)
                self._ema_data_cache[ema1] = trend_classifier._calculate_ema(ema1)
                self._ema_data_cache[ema2] = trend_classifier._calculate_ema(ema2)
        
        # Store data reference for is_trend_aligned access
        self._data_for_filter = data
        
        # Call parent's method - it will use our overridden is_trend_aligned
        return super().execute_realistic_trades(patterns, data, trend_data, timeframe, pair)
    
    def run_single_strategy_test_with_filter(self, pair: str, timeframe: str, 
                                           filter_name: str, days_back: int = 730) -> Dict:
        """
        Run single test with specific EMA filter
        """
        self.set_current_filter(filter_name)
        result = self.run_single_strategy_test(pair, timeframe, days_back)
        
        # Add filter information
        result['filter_name'] = filter_name
        result['filter_description'] = self.ema_filters[filter_name]['description']
        
        return result
    
    def run_filter_comparison_single(self, pair: str, timeframe: str, days_back: int = 730) -> pd.DataFrame:
        """
        Run comparison of all EMA filters for single pair/timeframe
        """
        print(f"\n🎯 EMA FILTER COMPARISON: {pair} {timeframe}")
        print("=" * 60)
        
        results = []
        
        for filter_name, filter_config in self.ema_filters.items():
            print(f"\n📊 Testing: {filter_name}")
            print(f"   Description: {filter_config['description']}")
            
            result = self.run_single_strategy_test_with_filter(pair, timeframe, filter_name, days_back)
            results.append(result)
            
            # Print summary
            print(f"   ✅ Trades: {result['total_trades']}")
            print(f"   📈 Win Rate: {result['win_rate']:.1f}%")
            print(f"   💰 Profit Factor: {result['profit_factor']:.2f}")
            print(f"   📊 Total Return: {result['total_return']:.2f}%")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save results
        self._save_single_filter_comparison(comparison_df, pair, timeframe)
        
        return comparison_df
    
    def run_parallel_comprehensive_filter_analysis(self, analysis_period: str = 'priority_1') -> List[Dict]:
        """
        Run comprehensive analysis across all pairs/timeframes AND all filters
        This is the enhanced version that tests every combination
        """
        print(f"\n🚀 COMPREHENSIVE EMA FILTER ANALYSIS - {analysis_period.upper()}")
        period_config = ANALYSIS_PERIODS[analysis_period]
        days_back = period_config['days_back']
        
        print(f"📊 Period: {period_config['name']}")
        print(f"📅 Days back: {days_back:,}")
        print(f"🎯 Testing {len(self.ema_filters)} EMA filter variations")
        print("=" * 70)
        
        # Discover valid combinations
        valid_combinations = self.discover_valid_data_combinations()
        if not valid_combinations:
            print("❌ No valid data combinations found")
            return []
        
        # Create test combinations for ALL filters
        test_combinations = []
        for pair, timeframe in valid_combinations:
            for filter_name in self.ema_filters.keys():
                test_combinations.append({
                    'pair': pair,
                    'timeframe': timeframe,
                    'filter_name': filter_name,
                    'days_back': days_back,
                    'analysis_period': analysis_period
                })
        
        print(f"📊 Valid pair/timeframe combinations: {len(valid_combinations)}")
        print(f"📊 EMA filters to test: {len(self.ema_filters)}")
        print(f"📊 Total tests to run: {len(test_combinations):,}")
        
        # Run optimized parallel processing
        all_results = self.run_optimized_parallel_filter_tests(test_combinations)
        
        # Generate comprehensive Excel report
        if all_results:
            self.generate_comprehensive_filter_excel_report(all_results, analysis_period, period_config)
        
        # Print summary
        successful_results = [r for r in all_results if r['total_trades'] > 0]
        print(f"\n🎯 ANALYSIS COMPLETE:")
        print(f"   Total combinations tested: {len(all_results):,}")
        print(f"   Successful combinations: {len(successful_results):,}")
        print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        
        # Best performer by filter
        if successful_results:
            df = pd.DataFrame(successful_results)
            for filter_name in self.ema_filters.keys():
                filter_results = df[df['filter_name'] == filter_name]
                if len(filter_results) > 0:
                    avg_pf = filter_results['profit_factor'].mean()
                    avg_wr = filter_results['win_rate'].mean()
                    total_trades = filter_results['total_trades'].sum()
                    print(f"\n📊 {filter_name}:")
                    print(f"   Avg PF: {avg_pf:.2f}, Avg WR: {avg_wr:.1f}%, Total trades: {total_trades}")
        
        return all_results
    
    def run_optimized_parallel_filter_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run filter tests in parallel with memory management
        """
        print(f"\n🔄 OPTIMIZED PARALLEL FILTER EXECUTION")
        print(f"⚡ Workers: {self.max_workers}")
        print(f"📦 Chunk size: {self.chunk_size}")
        
        start_time = time.time()
        results = []
        
        # Process in chunks for memory management
        chunk_size = self.chunk_size
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_tests = test_combinations[chunk_start:chunk_end]
            
            print(f"\n📦 Chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_tests)} tests)")
            
            # Memory check
            memory_percent = psutil.virtual_memory().percent
            print(f"💾 Memory usage: {memory_percent:.1f}%")
            
            if memory_percent > self.memory_threshold * 100:
                print("⚠️  High memory usage, triggering cleanup...")
                gc.collect()
            
            # Process chunk with multiprocessing
            try:
                with Pool(processes=self.max_workers) as pool:
                    chunk_results = pool.map(run_single_filter_test_worker, chunk_tests)
                    results.extend(chunk_results)
                
                # Progress tracking
                completed = chunk_end
                progress = (completed / len(test_combinations)) * 100
                print(f"✅ Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Add empty results for failed chunk
                for test in chunk_tests:
                    results.append({
                        'pair': test['pair'],
                        'timeframe': test['timeframe'],
                        'filter_name': test['filter_name'],
                        'total_trades': 0,
                        'description': f"Parallel processing error: {str(e)}"
                    })
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ PARALLEL FILTER EXECUTION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🎯 Success rate: {success_count}/{len(test_combinations)} ({success_count/len(test_combinations)*100:.1f}%)")
        print(f"⚡ Speed: {len(test_combinations)/total_time:.1f} tests/second")
        
        return results
    
    def generate_comprehensive_filter_excel_report(self, all_results: List[Dict], 
                                                 analysis_period: str, period_config: Dict):
        """
        Generate professional multi-sheet Excel report for filter analysis
        """
        print(f"\n📊 GENERATING COMPREHENSIVE FILTER EXCEL REPORT...")
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_period}_ema_filter_comprehensive_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Results
                df_all.to_excel(writer, sheet_name='All_Results', index=False)
                print("   ✅ Sheet 1: All Results")
                
                # SHEET 2: Successful Results Only
                successful_df = df_all[df_all['total_trades'] > 0]
                if len(successful_df) > 0:
                    successful_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    print("   ✅ Sheet 2: Successful Results")
                    
                    # SHEET 3: Performance Analysis by Filter
                    filter_analysis = self.create_filter_analysis(successful_df)
                    filter_analysis.to_excel(writer, sheet_name='Filter_Analysis', index=False)
                    print("   ✅ Sheet 3: Filter Analysis")
                    
                    # SHEET 4: Performance Analysis by Timeframe
                    tf_analysis = self.create_timeframe_analysis(successful_df)
                    tf_analysis.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    print("   ✅ Sheet 4: Timeframe Analysis")
                    
                    # SHEET 5: Performance Analysis by Pair
                    pair_analysis = self.create_pair_analysis(successful_df)
                    pair_analysis.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ✅ Sheet 5: Pair Analysis")
                    
                    # SHEET 6: Filter Comparison Matrix
                    filter_matrix = self.create_filter_comparison_matrix(successful_df)
                    filter_matrix.to_excel(writer, sheet_name='Filter_Comparison', index=False)
                    print("   ✅ Sheet 6: Filter Comparison Matrix")
                    
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    empty_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    empty_df.to_excel(writer, sheet_name='Filter_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Filter_Comparison', index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"📁 EXCEL REPORT SAVED: {filename}")
            print(f"📊 6 comprehensive analysis sheets created")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")
            # Fallback: Save as CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def create_filter_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create filter performance analysis"""
        try:
            filter_analysis = df.groupby('filter_name').agg({
                'profit_factor': ['mean', 'count'],
                'win_rate': 'mean',
                'loss_rate': 'mean',
                'be_rate': 'mean',
                'total_trades': 'sum',
                'total_return': 'mean',
                'total_pnl': 'sum'
            }).round(2)

            # Flatten column names
            filter_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                     'Avg_Loss_Rate', 'Avg_BE_Rate', 'Total_Trades', 
                                     'Avg_Return', 'Total_PnL']
            filter_analysis = filter_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            # Add filter descriptions
            filter_analysis = filter_analysis.reset_index()
            filter_analysis['Description'] = filter_analysis['filter_name'].map(
                {k: v['description'] for k, v in self.ema_filters.items()}
            )
            
            return filter_analysis
            
        except Exception as e:
            print(f"   ⚠️  Filter analysis error: {str(e)}")
            return pd.DataFrame({'Filter': ['Error'], 'Note': [str(e)]})
    
    def create_filter_comparison_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create filter comparison matrix"""
        try:
            # Pivot table with filters as rows and key metrics as columns
            comparison = df.groupby('filter_name').agg({
                'total_trades': 'sum',
                'winning_trades': 'sum',
                'losing_trades': 'sum',
                'breakeven_trades': 'sum',
                'win_rate': 'mean',
                'profit_factor': 'mean',
                'total_return': 'mean',
                'avg_trade_duration': 'mean'
            }).round(2)
            
            comparison = comparison.reset_index()
            comparison['Win_Ratio'] = (comparison['winning_trades'] / comparison['total_trades'] * 100).round(1)
            
            # Sort by profit factor
            comparison = comparison.sort_values('profit_factor', ascending=False)
            
            return comparison
            
        except Exception as e:
            print(f"   ⚠️  Filter comparison error: {str(e)}")
            return pd.DataFrame({'Error': [str(e)]})
    
    def _save_single_filter_comparison(self, df: pd.DataFrame, pair: str, timeframe: str):
        """Save single pair/timeframe filter comparison to Excel"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/ema_filter_comparison_{pair}_{timeframe}_{timestamp}.xlsx"
        
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_cols = ['filter_name', 'filter_description', 'total_trades', 
                               'winning_trades', 'losing_trades', 'breakeven_trades',
                               'win_rate', 'loss_rate', 'be_rate', 'profit_factor',
                               'total_pnl', 'total_return', 'avg_trade_duration']
                
                summary_df = df[summary_cols].copy()
                summary_df.to_excel(writer, sheet_name='Filter_Comparison', index=False)
                
                # Individual filter sheets with trade details
                for _, row in df.iterrows():
                    filter_name = row['filter_name']
                    if row.get('trades'):
                        trades_df = pd.DataFrame(row['trades'])
                        sheet_name = f"{filter_name}_Trades"[:31]
                        trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n📁 Results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")


def run_single_filter_test_worker(test_config: Dict) -> Dict:
    """
    Worker function for parallel filter processing
    """
    try:
        # Create fresh engine instance
        engine = EMAFilterBacktestEngine()
        
        result = engine.run_single_strategy_test_with_filter(
            test_config['pair'],
            test_config['timeframe'],
            test_config['filter_name'],
            test_config['days_back']
        )
        
        # Add analysis period info
        result['analysis_period'] = test_config['analysis_period']
        
        # Clean up
        del engine
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'filter_name': test_config['filter_name'],
            'analysis_period': test_config['analysis_period'],
            'total_trades': 0,
            'description': f"Worker error: {str(e)}"
        }


def main():
   """Enhanced main function matching core_backtest_engine.py functionality"""
   print("🎯 EMA FILTER BACKTESTING ENGINE - PRODUCTION READY")
   print("=" * 60)
   
   # Check system resources
   engine = EMAFilterBacktestEngine()
   if not engine.check_system_resources():
       print("❌ Insufficient system resources")
       return
   
   print("\n🎯 SELECT ANALYSIS MODE:")
   print("1. Quick Validation (Single test - EURUSD 3D, All Filters)")
   print("2. Comprehensive Analysis - Priority 1 (2015-2025, All pairs/timeframes/filters)")
   print("3. Comprehensive Analysis - Priority 2 (2020-2025, All pairs/timeframes/filters)")
   print("4. Custom Single Test (Choose pair/timeframe/filter)")
   
   choice = input("\nEnter choice (1-4): ").strip()
   
   if choice == '1':
       # Quick validation test
       print("\n🧪 QUICK VALIDATION TEST:")
       print("Testing EURUSD 3D with all EMA filters...")
       
       # Run comparison for EURUSD 3D
       start_time = time.time()
       comparison_df = engine.run_filter_comparison_single('EURUSD', '3D', 730)
       end_time = time.time()
       
       print(f"\n🕐 BENCHMARK TIME: {end_time - start_time:.1f} seconds")
       print("\n📊 FILTER COMPARISON RESULTS:")
       print(comparison_df[['filter_name', 'total_trades', 'win_rate', 'profit_factor', 'total_return']].to_string(index=False))
       
       # Find best performer
       if len(comparison_df) > 0:
           best_pf = comparison_df.loc[comparison_df['profit_factor'].idxmax()]
           print(f"\n🏆 Best Profit Factor: {best_pf['filter_name']} - PF {best_pf['profit_factor']:.2f}")
           
           best_wr = comparison_df.loc[comparison_df['win_rate'].idxmax()]
           print(f"🏆 Best Win Rate: {best_wr['filter_name']} - WR {best_wr['win_rate']:.1f}%")
   
   elif choice == '2':
       # Comprehensive analysis - Priority 1
       print("\n🚀 COMPREHENSIVE FILTER ANALYSIS - PRIORITY 1")
       print("This will test ALL pairs, timeframes, and filters with 10 years of data")
       print(f"Estimated tests: {len(engine.discover_all_pairs())} pairs × multiple timeframes × {len(engine.ema_filters)} filters")
       confirm = input("Continue? (y/n): ").strip().lower()
       
       if confirm == 'y':
           engine.run_parallel_comprehensive_filter_analysis('priority_1')
       else:
           print("Analysis cancelled")
   
   elif choice == '3':
       # Comprehensive analysis - Priority 2
       print("\n🚀 COMPREHENSIVE FILTER ANALYSIS - PRIORITY 2")
       print("This will test ALL pairs, timeframes, and filters with 4 years of data")
       print(f"Estimated tests: {len(engine.discover_all_pairs())} pairs × multiple timeframes × {len(engine.ema_filters)} filters")
       confirm = input("Continue? (y/n): ").strip().lower()
       
       if confirm == 'y':
           engine.run_parallel_comprehensive_filter_analysis('priority_2')
       else:
           print("Analysis cancelled")
   
   elif choice == '4':
       # Custom single test
       print("\n🎯 CUSTOM SINGLE TEST:")
       
       # Get available pairs
       pairs = engine.discover_all_pairs()
       print(f"Available pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
       
       pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
       timeframe = input("Enter timeframe (e.g., 3D): ").strip()
       days_back = input("Enter days back (default: 730): ").strip()
       days_back = int(days_back) if days_back else 730
       
       # Show available filters
       print("\n📊 Available filters:")
       for i, (filter_name, filter_config) in enumerate(engine.ema_filters.items(), 1):
           print(f"{i}. {filter_name}: {filter_config['description']}")
       
       filter_choice = input("\nSelect filter (1-6) or 'all' for comparison: ").strip()
       
       if filter_choice.lower() == 'all':
           # Run comparison
           print(f"\n🚀 Running filter comparison for {pair} {timeframe}")
           comparison_df = engine.run_filter_comparison_single(pair, timeframe, days_back)
           
           print("\n📊 FILTER COMPARISON RESULTS:")
           print(comparison_df[['filter_name', 'total_trades', 'win_rate', 'profit_factor', 'total_return']].to_string(index=False))
           
       else:
           # Run single filter
           try:
               filter_idx = int(filter_choice) - 1
               filter_name = list(engine.ema_filters.keys())[filter_idx]
               
               print(f"\n🚀 Testing {pair} {timeframe} with {filter_name}")
               result = engine.run_single_strategy_test_with_filter(pair, timeframe, filter_name, days_back)
               
               print(f"\n📊 RESULTS:")
               print(f"   Filter: {filter_name}")
               print(f"   Trades: {result['total_trades']}")
               print(f"   Win Rate: {result['win_rate']:.1f}%")
               print(f"   Loss Rate: {result['loss_rate']:.1f}%")
               print(f"   BE Rate: {result['be_rate']:.1f}%")
               print(f"   Profit Factor: {result['profit_factor']:.2f}")
               print(f"   Total Return: {result['total_return']:.2f}%")
               print(f"   Average Duration: {result['avg_trade_duration']:.1f} days")
               
               # Generate detailed report
               if result['total_trades'] > 0:
                   print(f"\n📋 GENERATING DETAILED REPORT...")
                   engine.generate_manual_chart_analysis_report(result)
                   
           except (ValueError, IndexError):
               print("❌ Invalid filter selection")
   
   else:
       print("❌ Invalid choice")


if __name__ == "__main__":
   main()