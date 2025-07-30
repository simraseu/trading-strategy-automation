"""
Distance Validation Backtesting Engine - Extends Core Framework
Tests impact of different 2.5x distance validation requirements on zone quality
Inherits from CoreBacktestEngine - NO code duplication, ONLY overrides validation logic
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import warnings
warnings.filterwarnings('ignore')

# Import your existing engine
from core_backtest_engine import CoreBacktestEngine, run_single_test_worker

# Set process priority for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

# DISTANCE VALIDATION CONFIGURATIONS
DISTANCE_VALIDATION_CONFIGS = {
    'minimal_1x': {
        'name': 'Minimal 1.0x Distance',
        'multiplier': 1.0,
        'description': 'Zone must achieve 1.0x range distance before invalidation'
    },
    'conservative_1_5x': {
        'name': 'Conservative 1.5x Distance', 
        'multiplier': 1.5,
        'description': 'Zone must achieve 1.5x range distance before invalidation'
    },
    'moderate_2x': {
        'name': 'Moderate 2.0x Distance',
        'multiplier': 2.0, 
        'description': 'Zone must achieve 2.0x range distance before invalidation'
    },
    'current_2_5x': {
        'name': 'Current 2.5x Distance (Baseline)',
        'multiplier': 2.5,
        'description': 'Zone must achieve 2.5x range distance before invalidation'
    },
    'aggressive_3x': {
        'name': 'Aggressive 3.0x Distance',
        'multiplier': 3.0,
        'description': 'Zone must achieve 3.0x range distance before invalidation'
    },
    'ultra_aggressive_4x': {
        'name': 'Ultra Aggressive 4.0x Distance',
        'multiplier': 4.0,
        'description': 'Zone must achieve 4.0x range distance before invalidation'
    }
}

class DistanceValidationBacktester(CoreBacktestEngine):
    """
    EXTENDS CoreBacktestEngine to test different distance validation requirements
    
    CRITICAL: Only overrides zone validation logic - everything else identical
    - Same zone detection and formation logic
    - Same invalidation rules (50% wick penetration)  
    - Same trend filtering and trade management
    - ONLY changes: The distance requirement for zone validation
    """
    
    def __init__(self, distance_multiplier: float = 2.5):
        """
        Initialize with specific distance validation multiplier
        
        Args:
            distance_multiplier: Distance multiplier for zone validation (1.0-4.0)
        """
        super().__init__()  # Initialize parent class
        
        self.distance_multiplier = distance_multiplier
        self.validation_config = None
        
        # Find matching config
        for config_key, config in DISTANCE_VALIDATION_CONFIGS.items():
            if config['multiplier'] == distance_multiplier:
                self.validation_config = config
                break
        
        if not self.validation_config:
            self.validation_config = {
                'name': f'Custom {distance_multiplier}x Distance',
                'multiplier': distance_multiplier,
                'description': f'Zone must achieve {distance_multiplier}x range distance before invalidation'
            }
        
        print(f"🎯 Distance Validation Engine initialized:")
        print(f"   Validation: {self.validation_config['name']}")
        print(f"   Multiplier: {self.distance_multiplier}x")
        print(f"   Description: {self.validation_config['description']}")
    
    def track_zone_validation_realtime(self, zone: Dict, data: pd.DataFrame, start_idx: int) -> Dict:
        """
        OVERRIDDEN: Track zone validation with CUSTOM distance multiplier
        
        CRITICAL CHANGE: Uses self.distance_multiplier instead of hardcoded 2.5
        Everything else identical to parent implementation
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        zone_type = zone['type']
        
        # Calculate targets using CUSTOM multiplier
        if zone_type in ['D-B-D', 'R-B-D']:  # Supply zones
            validation_target = zone_low - (self.distance_multiplier * zone_range)  # CUSTOM MULTIPLIER
            invalidation_level = zone_low + (zone_range * 0.50)
        else:  # Demand zones
            validation_target = zone_high + (self.distance_multiplier * zone_range)  # CUSTOM MULTIPLIER
            invalidation_level = zone_high - (zone_range * 0.50)
        
        # Track price movement (identical to parent)
        for idx in range(start_idx, len(data)):
            candle = data.iloc[idx]
            
            # Check validation first
            if zone_type in ['D-B-D', 'R-B-D']:  # Supply
                if candle['low'] <= validation_target:
                    return {
                        'validated': True,
                        'validation_idx': idx,
                        'invalidated': False,
                        'invalidation_idx': None
                    }
                if candle['high'] >= invalidation_level:
                    return {
                        'validated': False,
                        'validation_idx': None,
                        'invalidated': True,
                        'invalidation_idx': idx
                    }
            else:  # Demand
                if candle['high'] >= validation_target:
                    return {
                        'validated': True,
                        'validation_idx': idx,
                        'invalidated': False,
                        'invalidation_idx': None
                    }
                if candle['low'] <= invalidation_level:
                    return {
                        'validated': False,
                        'validation_idx': None,
                        'invalidated': True,
                        'invalidation_idx': idx
                    }
        
        # Neither validated nor invalidated
        return {
            'validated': False,
            'validation_idx': None,
            'invalidated': False,
            'invalidation_idx': None
        }
    
    def run_distance_validation_analysis(self, pairs: List[str], timeframes: List[str], 
                                       days_back: int = 730) -> List[Dict]:
        """
        Run comprehensive distance validation analysis across multiple configurations
        
        Args:
            pairs: List of currency pairs to test
            timeframes: List of timeframes to test  
            days_back: Days of historical data to analyze
            
        Returns:
            List of results with distance validation performance
        """
        print(f"\n🧪 DISTANCE VALIDATION ANALYSIS")
        print(f"📊 Testing {len(DISTANCE_VALIDATION_CONFIGS)} distance configurations")
        print(f"💰 Pairs: {', '.join(pairs)}")
        print(f"📈 Timeframes: {', '.join(timeframes)}")
        print(f"📅 Historical period: {days_back} days")
        print("=" * 80)
        
        all_results = []
        
        # Test each distance configuration
        for config_key, config in DISTANCE_VALIDATION_CONFIGS.items():
            print(f"\n🎯 TESTING: {config['name']}")
            print(f"   Multiplier: {config['multiplier']}x")
            print(f"   Expected impact: {'Fewer, higher-quality zones' if config['multiplier'] > 2.5 else 'More zones, potentially lower success rates'}")
            
            # Create test combinations for this configuration
            test_combinations = []
            for pair in pairs:
                for timeframe in timeframes:
                    test_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'days_back': days_back,
                        'distance_multiplier': config['multiplier'],
                        'config_name': config['name'],
                        'config_key': config_key
                    })
            
            # Run parallel tests for this configuration
            config_results = self.run_optimized_distance_tests(test_combinations)
            
            # Add configuration metadata
            for result in config_results:
                result['distance_config'] = config_key
                result['distance_multiplier'] = config['multiplier']
                result['validation_method'] = config['name']
            
            all_results.extend(config_results)
            
            # Progress summary
            successful_results = [r for r in config_results if r['total_trades'] > 0]
            print(f"   ✅ {config['name']}: {len(successful_results)}/{len(config_results)} successful combinations")
            
            if successful_results:
                avg_trades = sum(r['total_trades'] for r in successful_results) / len(successful_results)
                avg_pf = sum(r['profit_factor'] for r in successful_results) / len(successful_results)
                print(f"   📊 Average trades: {avg_trades:.1f}, Average PF: {avg_pf:.2f}")
        
        # Generate comprehensive comparison report
        if all_results:
            self.generate_distance_validation_report(all_results, days_back)
        
        return all_results
    
    def run_optimized_distance_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run distance validation tests with parallel processing and memory management
        """
        print(f"🔄 Processing {len(test_combinations)} tests with {self.max_workers} workers...")
        
        start_time = time.time()
        results = []
        
        # Process in chunks for memory management
        chunk_size = self.chunk_size
        total_chunks = (len(test_combinations) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(test_combinations))
            chunk_tests = test_combinations[chunk_start:chunk_end]
            
            # Memory check
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.memory_threshold * 100:
                gc.collect()
            
            # Process chunk with multiprocessing
            try:
                with Pool(processes=self.max_workers) as pool:
                    chunk_results = pool.map(run_distance_validation_worker, chunk_tests)
                    results.extend(chunk_results)
                
                completed = chunk_end
                progress = (completed / len(test_combinations)) * 100
                print(f"   ✅ Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Add empty results for failed chunk
                for test in chunk_tests:
                    results.append(self.create_empty_result(test['pair'], test['timeframe'], f"Processing error: {str(e)}"))
            
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"   ⏱️  Completed in {total_time:.1f}s")
        print(f"   🎯 Success rate: {success_count}/{len(test_combinations)} ({success_count/len(test_combinations)*100:.1f}%)")
        
        return results
    
    def generate_distance_validation_report(self, all_results: List[Dict], days_back: int):
        """
        Generate comprehensive Excel report comparing distance validation performance
        """
        print(f"\n📊 GENERATING DISTANCE VALIDATION COMPARISON REPORT...")
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/distance_validation_analysis_{days_back}days_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Results
                df_all.to_excel(writer, sheet_name='All_Results', index=False)
                print("   ✅ Sheet 1: All Results")
                
                # SHEET 2: Distance Configuration Comparison
                successful_df = df_all[df_all['total_trades'] > 0]
                if len(successful_df) > 0:
                    distance_comparison = self.create_distance_comparison_analysis(successful_df)
                    distance_comparison.to_excel(writer, sheet_name='Distance_Comparison', index=False)
                    print("   ✅ Sheet 2: Distance Configuration Comparison")
                    
                    # SHEET 3: Performance by Distance Level
                    distance_performance = self.create_distance_performance_analysis(successful_df)
                    distance_performance.to_excel(writer, sheet_name='Distance_Performance', index=False)
                    print("   ✅ Sheet 3: Performance by Distance Level")
                    
                    # SHEET 4: Pair Analysis by Distance
                    pair_distance_analysis = self.create_pair_distance_analysis(successful_df)
                    pair_distance_analysis.to_excel(writer, sheet_name='Pair_Distance_Analysis', index=False)
                    print("   ✅ Sheet 4: Pair Analysis by Distance")
                    
                    # SHEET 5: Timeframe Analysis by Distance
                    tf_distance_analysis = self.create_timeframe_distance_analysis(successful_df)
                    tf_distance_analysis.to_excel(writer, sheet_name='Timeframe_Distance_Analysis', index=False)
                    print("   ✅ Sheet 5: Timeframe Analysis by Distance")
                    
                    # SHEET 6: Trade Quality Impact Analysis
                    quality_impact = self.create_trade_quality_impact_analysis(successful_df)
                    quality_impact.to_excel(writer, sheet_name='Trade_Quality_Impact', index=False)
                    print("   ✅ Sheet 6: Trade Quality Impact Analysis")
                
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    for sheet_name in ['Distance_Comparison', 'Distance_Performance', 'Pair_Distance_Analysis', 
                                     'Timeframe_Distance_Analysis', 'Trade_Quality_Impact']:
                        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"\n📁 DISTANCE VALIDATION REPORT SAVED:")
            print(f"   File: {filename}")
            print(f"   📊 6 comprehensive analysis sheets created")
            print(f"   🎯 Compare impact of different distance validation requirements")
            
        except Exception as e:
            print(f"❌ Error creating distance validation report: {str(e)}")
            # Fallback: Save as CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def create_distance_comparison_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance configuration comparison analysis"""
        try:
            comparison = df.groupby('distance_config').agg({
                'total_trades': 'sum',
                'winning_trades': 'sum', 
                'losing_trades': 'sum',
                'breakeven_trades': 'sum',
                'profit_factor': 'mean',
                'win_rate': 'mean',
                'loss_rate': 'mean',
                'be_rate': 'mean',
                'total_return': 'mean',
                'avg_trade_duration': 'mean',
                'distance_multiplier': 'first',
                'validation_method': 'first'
            }).round(2)
            
            # Calculate additional metrics
            comparison['total_strategy_count'] = df.groupby('distance_config').size()
            comparison['successful_strategies'] = df.groupby('distance_config').apply(lambda x: (x['total_trades'] > 0).sum())
            comparison['success_rate'] = (comparison['successful_strategies'] / comparison['total_strategy_count'] * 100).round(1)
            
            # Sort by distance multiplier
            comparison = comparison.sort_values('distance_multiplier')
            
            return comparison.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Distance comparison analysis error: {str(e)}")
            return pd.DataFrame({'Distance_Config': ['Error'], 'Note': [str(e)]})
    
    def create_distance_performance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance analysis by distance level"""
        try:
            # Group by distance multiplier ranges
            df['distance_category'] = df['distance_multiplier'].apply(lambda x: 
                'Low (1.0-1.5x)' if x <= 1.5 else
                'Medium (2.0-2.5x)' if x <= 2.5 else
                'High (3.0-4.0x)'
            )
            
            performance = df.groupby('distance_category').agg({
                'total_trades': 'sum',
                'profit_factor': 'mean',
                'win_rate': 'mean',
                'total_return': 'mean',
                'distance_multiplier': ['min', 'max'],
                'validation_method': lambda x: ', '.join(x.unique())
            }).round(2)
            
            # Flatten column names
            performance.columns = ['Total_Trades', 'Avg_Profit_Factor', 'Avg_Win_Rate', 
                                 'Avg_Return', 'Min_Multiplier', 'Max_Multiplier', 'Methods']
            
            return performance.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Distance performance analysis error: {str(e)}")
            return pd.DataFrame({'Distance_Category': ['Error'], 'Note': [str(e)]})
    
    def create_pair_distance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pair analysis across distance configurations"""
        try:
            pair_analysis = df.pivot_table(
                index='pair',
                columns='distance_config', 
                values=['total_trades', 'profit_factor', 'win_rate'],
                aggfunc='mean'
            ).round(2)
            
            # Flatten column names
            pair_analysis.columns = ['_'.join(col).strip() for col in pair_analysis.columns.values]
            
            return pair_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Pair distance analysis error: {str(e)}")
            return pd.DataFrame({'Pair': ['Error'], 'Note': [str(e)]})
    
    def create_timeframe_distance_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create timeframe analysis across distance configurations"""
        try:
            tf_analysis = df.pivot_table(
                index='timeframe',
                columns='distance_config',
                values=['total_trades', 'profit_factor', 'win_rate'],
                aggfunc='mean'
            ).round(2)
            
            # Flatten column names
            tf_analysis.columns = ['_'.join(col).strip() for col in tf_analysis.columns.values]
            
            return tf_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Timeframe distance analysis error: {str(e)}")
            return pd.DataFrame({'Timeframe': ['Error'], 'Note': [str(e)]})
    
    def create_trade_quality_impact_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trade quality impact of different distance requirements"""
        try:
            quality_metrics = []
            
            for config in df['distance_config'].unique():
                config_data = df[df['distance_config'] == config]
                
                if len(config_data) > 0:
                    avg_trades_per_strategy = config_data['total_trades'].mean()
                    strategies_with_trades = (config_data['total_trades'] > 0).sum()
                    total_strategies = len(config_data)
                    
                    quality_metrics.append({
                        'Distance_Config': config,
                        'Distance_Multiplier': config_data['distance_multiplier'].iloc[0],
                        'Validation_Method': config_data['validation_method'].iloc[0],
                        'Total_Strategies_Tested': total_strategies,
                        'Strategies_With_Trades': strategies_with_trades,
                        'Strategy_Success_Rate': round((strategies_with_trades / total_strategies) * 100, 1),
                        'Avg_Trades_Per_Strategy': round(avg_trades_per_strategy, 1),
                        'Total_Trades_All_Strategies': config_data['total_trades'].sum(),
                        'Avg_Profit_Factor': round(config_data['profit_factor'].mean(), 2),
                        'Avg_Win_Rate': round(config_data['win_rate'].mean(), 1),
                        'Quality_Score': round(config_data['profit_factor'].mean() * config_data['win_rate'].mean() / 100, 2)
                    })
            
            quality_df = pd.DataFrame(quality_metrics)
            quality_df = quality_df.sort_values('Distance_Multiplier')
            
            return quality_df
            
        except Exception as e:
            print(f"   ⚠️  Trade quality impact analysis error: {str(e)}")
            return pd.DataFrame({'Analysis': ['Error'], 'Note': [str(e)]})


# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTION FOR DISTANCE VALIDATION
# ============================================================================

def run_distance_validation_worker(test_config: Dict) -> Dict:
    """
    Worker function for distance validation parallel processing
    Each worker creates its own engine instance with specific distance multiplier
    """
    try:
        # Create fresh engine instance with specific distance multiplier
        engine = DistanceValidationBacktester(distance_multiplier=test_config['distance_multiplier'])
        
        result = engine.run_single_strategy_test(
            test_config['pair'],
            test_config['timeframe'],
            test_config['days_back']
        )
        
        # Add distance validation metadata
        result['distance_config'] = test_config['config_key']
        result['distance_multiplier'] = test_config['distance_multiplier']
        result['validation_method'] = test_config['config_name']
        
        # Clean up
        del engine
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'distance_config': test_config['config_key'], 
            'distance_multiplier': test_config['distance_multiplier'],
            'validation_method': test_config['config_name'],
            'total_trades': 0,
            'description': f"Distance validation worker error: {str(e)}"
        }


def main():
    """Enhanced main function for distance validation testing"""
    print("🎯 DISTANCE VALIDATION BACKTESTING ENGINE")
    print("=" * 60)
    
    # Check system resources
    engine = DistanceValidationBacktester()
    if not engine.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 SELECT DISTANCE VALIDATION ANALYSIS MODE:")
    print("1. Quick Validation (EURUSD 3D, all distance configs)")
    print("2. Single Distance Test (Custom pair/timeframe)")
    print("3. Comprehensive Distance Analysis (Multiple pairs/timeframes)")
    print("4. Distance Impact Comparison (All configs vs baseline)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Quick validation across all distance configurations
        print("\n🧪 QUICK DISTANCE VALIDATION TEST:")
        print("Testing EURUSD 3D with all distance configurations...")
        
        results = engine.run_distance_validation_analysis(['EURUSD'], ['3D'], 730)
        
        print(f"\n📊 DISTANCE VALIDATION RESULTS:")
        for config_key, config in DISTANCE_VALIDATION_CONFIGS.items():
            config_results = [r for r in results if r.get('distance_config') == config_key]
            if config_results:
                result = config_results[0]  # Should only be one result
                print(f"\n   {config['name']} ({config['multiplier']}x):")
                print(f"     Trades: {result['total_trades']}")
                print(f"     Win Rate: {result['win_rate']:.1f}%")
                print(f"     Profit Factor: {result['profit_factor']:.2f}")
                if result['total_trades'] == 0:
                    print(f"     Issue: {result.get('description', 'No trades found')}")
    
    elif choice == '2':
        # Single distance test
        print("\n🎯 SINGLE DISTANCE TEST:")
        
        # Show available configurations
        print("Available distance configurations:")
        for i, (key, config) in enumerate(DISTANCE_VALIDATION_CONFIGS.items(), 1):
            print(f"   {i}. {config['name']} ({config['multiplier']}x)")
        
        config_choice = input("\nSelect configuration (1-6): ").strip()
        try:
            config_keys = list(DISTANCE_VALIDATION_CONFIGS.keys())
            selected_config = DISTANCE_VALIDATION_CONFIGS[config_keys[int(config_choice) - 1]]
            
            pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
            timeframe = input("Enter timeframe (e.g., 3D): ").strip()
            days_back = int(input("Enter days back (e.g., 730): ").strip())
            
            # Create engine with selected multiplier
            test_engine = DistanceValidationBacktester(selected_config['multiplier'])
            result = test_engine.run_single_strategy_test(pair, timeframe, days_back)
            
            print(f"\n📊 {selected_config['name']} RESULTS:")
            print(f"   Pair: {result['pair']} {result['timeframe']}")
            print(f"   Distance Requirement: {selected_config['multiplier']}x")
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
            print(f"   Total Return: {result['total_return']:.2f}%")
            
            if result['total_trades'] == 0:
                print(f"   Issue: {result.get('description', 'No trades found')}")
                
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    elif choice == '3':
        # Comprehensive distance analysis
        print("\n🚀 COMPREHENSIVE DISTANCE ANALYSIS")
        print("This will test ALL distance configurations across multiple pairs/timeframes")
        
        # Get available pairs
        pairs = engine.discover_all_pairs()
        print(f"\nAvailable pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
        
        selected_pairs_input = input("Enter pairs (comma-separated, or 'all' for all pairs): ").strip()
        if selected_pairs_input.lower() == 'all':
            selected_pairs = pairs
        else:
            selected_pairs = [p.strip().upper() for p in selected_pairs_input.split(',')]
        
        timeframes_input = input("Enter timeframes (comma-separated, e.g., 3D,1W): ").strip()
        selected_timeframes = [tf.strip() for tf in timeframes_input.split(',')]
        
        days_back = int(input("Enter days back (e.g., 730): ").strip())
        
        confirm = input(f"\nTest {len(selected_pairs)} pairs × {len(selected_timeframes)} timeframes × 6 distance configs = {len(selected_pairs) * len(selected_timeframes) * 6} total tests? (y/n): ").strip().lower()
        
        if confirm == 'y':
            results = engine.run_distance_validation_analysis(selected_pairs, selected_timeframes, days_back)
            
            # Summary by distance configuration
            print(f"\n📊 DISTANCE VALIDATION SUMMARY:")
            for config_key, config in DISTANCE_VALIDATION_CONFIGS.items():
                config_results = [r for r in results if r.get('distance_config') == config_key and r['total_trades'] > 0]
                if config_results:
                    avg_trades = sum(r['total_trades'] for r in config_results) / len(config_results)
                    avg_pf = sum(r['profit_factor'] for r in config_results) / len(config_results) 
                    avg_wr = sum(r['win_rate'] for r in config_results) / len(config_results)
                    print(f"   {config['name']}: {len(config_results)} successful strategies, Avg trades: {avg_trades:.1f}, PF: {avg_pf:.2f}, WR: {avg_wr:.1f}%")
        else:
            print("Analysis cancelled")
    
    elif choice == '4':
        # Distance impact comparison
        print("\n📊 DISTANCE IMPACT COMPARISON")
        print("Compare all distance configurations against 2.5x baseline")
        
        pair = input("Enter pair for comparison (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = int(input("Enter days back (e.g., 730): ").strip())
        
        results = engine.run_distance_validation_analysis([pair], [timeframe], days_back)
        
        print(f"\n📊 DISTANCE IMPACT COMPARISON - {pair} {timeframe}:")
        print("-" * 80)
        
        baseline_result = None 
        for result in results:
            if result.get('distance_config') == 'current_2_5x':
                baseline_result = result
                break
        
        if baseline_result:
            baseline_trades = baseline_result['total_trades']
            baseline_pf = baseline_result['profit_factor']
            baseline_wr = baseline_result['win_rate']
            
            print(f"📋 BASELINE (2.5x): {baseline_trades} trades, PF {baseline_pf:.2f}, WR {baseline_wr:.1f}%")
            print("-" * 80)
            
            for config_key, config in DISTANCE_VALIDATION_CONFIGS.items():
                if config_key == 'current_2_5x':
                    continue  # Skip baseline
                    
                config_results = [r for r in results if r.get('distance_config') == config_key]
                if config_results:
                    result = config_results[0]
                    
                    # Calculate differences from baseline
                    trades_diff = result['total_trades'] - baseline_trades
                    pf_diff = result['profit_factor'] - baseline_pf
                    wr_diff = result['win_rate'] - baseline_wr
                    
                    trades_pct = (trades_diff / baseline_trades * 100) if baseline_trades > 0 else 0
                    
                    print(f"{config['name']} ({config['multiplier']}x):")
                    print(f"   Trades: {result['total_trades']} ({trades_diff:+d}, {trades_pct:+.1f}%)")
                    print(f"   Profit Factor: {result['profit_factor']:.2f} ({pf_diff:+.2f})")
                    print(f"   Win Rate: {result['win_rate']:.1f}% ({wr_diff:+.1f}%)")
                    
                    if result['total_trades'] == 0:
                        print(f"   Status: No valid zones at {config['multiplier']}x requirement")
                    elif trades_diff > 0:
                        print(f"   Impact: More opportunities, {'better' if pf_diff > 0 else 'lower'} quality")
                    else:
                        print(f"   Impact: Fewer opportunities, {'better' if pf_diff > 0 else 'lower'} quality")
                    print()
        else:
            print("❌ No baseline (2.5x) result found for comparison")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()