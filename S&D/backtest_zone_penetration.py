"""
Zone Penetration Backtesting Extension
Tests different wick penetration thresholds for forex zone trading
Inherits from CoreBacktestEngine for full functionality
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
import psutil
import warnings
warnings.filterwarnings('ignore')

# Import parent class and required modules
from core_backtest_engine import CoreBacktestEngine, ANALYSIS_PERIODS
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

# Set process priority for maximum CPU usage
try:
    if os.name == 'nt':  # Windows
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
except:
    pass

class ZonePenetrationBacktester(CoreBacktestEngine):
    """
    Zone penetration testing extension of CoreBacktestEngine
    Tests 7 different wick penetration thresholds for zone validation
    """
    
    def __init__(self):
        """Initialize penetration strategies and inherit parent functionality"""
        super().__init__()
        
        # Define exactly 7 penetration strategies
        self.penetration_strategies = {
            'Fresh_0pct': {
                'wick_threshold': 0.0, 
                'description': 'Fresh zones - 0% wick penetration allowed'
            },
            'Light_20pct': {
                'wick_threshold': 0.20, 
                'description': 'Light testing - 20% wick penetration allowed'
            },
            'Moderate_40pct': {
                'wick_threshold': 0.40, 
                'description': 'Moderate testing - 40% wick penetration allowed'
            },
            'Current_50pct': {
                'wick_threshold': 0.50, 
                'description': 'Current baseline - 50% wick penetration allowed'
            },
            'Heavy_60pct': {
                'wick_threshold': 0.60, 
                'description': 'Heavy testing - 60% wick penetration allowed'
            },
            'Ultra_Heavy_80pct': {
                'wick_threshold': 0.80, 
                'description': 'Ultra heavy testing - 80% wick penetration allowed'
            },
            'No_Filter': {
                'wick_threshold': None, 
                'description': 'No penetration filtering - all zones valid'
            }
        }
        
        print(f"🧪 Zone Penetration Backtester initialized:")
        print(f"   7 penetration strategies loaded")
        print(f"   Inheriting from CoreBacktestEngine")
        print(f"   CPU workers: {self.max_workers}")
    
    def check_zone_testing_with_penetration(self, zone: Dict, data: pd.DataFrame, 
                                      wick_threshold: Optional[float]) -> Tuple[bool, str]:
        """
        Check zone testing with specific wick penetration threshold
        FIXED: Prevents look-ahead bias by only checking penetration UP TO trade execution point
        
        Args:
            zone: Zone dictionary with boundaries and formation data
            data: Full OHLC DataFrame
            wick_threshold: Wick penetration threshold (0.0-1.0) or None for no filter
            
        Returns:
            Tuple of (is_valid: bool, reason: str)
        """
        try:
            # NO FILTER: All zones valid
            if wick_threshold is None:
                return True, "No penetration filtering applied"
            
            zone_end_idx = zone['end_idx']
            zone_high = zone['zone_high']
            zone_low = zone['zone_low']
            zone_size = zone_high - zone_low
            zone_type = zone['type']
            
            # Edge case: Zero zone size
            if zone_size <= 0:
                return False, "Invalid zone - zero or negative size"
            
            # CRITICAL FIX: Only check penetration from zone formation to CURRENT evaluation point
            # This prevents look-ahead bias by not checking the entire future dataset
            
            # For backtesting, we need to determine the "current" evaluation point
            # This should be the point where we're deciding whether to trade the zone
            # NOT the entire future dataset
            
            # Calculate reasonable evaluation window (e.g., 50 candles after zone formation)
            evaluation_window = 50  # Adjust this based on your typical trade duration
            max_check_idx = min(zone_end_idx + evaluation_window, len(data) - 1)
            
            # Check candles AFTER zone formation but only within reasonable evaluation window
            if zone_end_idx >= max_check_idx:
                return True, "Zone untested - no evaluation period available"
            
            # Test penetration within evaluation window only
            for i in range(zone_end_idx + 1, max_check_idx + 1):
                candle = data.iloc[i]
                
                if zone_type in ['R-B-R', 'D-B-R']:  # Demand zones
                    # Test wick penetration DOWNWARD through zone
                    wick_test_level = zone_low + (zone_size * wick_threshold)
                    if candle['low'] < wick_test_level:
                        return False, f"Demand zone tested - wick penetrated {wick_threshold*100:.0f}% level at candle {i}"
                        
                elif zone_type in ['D-B-D', 'R-B-D']:  # Supply zones
                    # Test wick penetration UPWARD through zone
                    wick_test_level = zone_high - (zone_size * wick_threshold)
                    if candle['high'] > wick_test_level:
                        return False, f"Supply zone tested - wick penetrated {wick_threshold*100:.0f}% level at candle {i}"
                
                else:
                    return False, f"Unknown zone type: {zone_type}"
            
            # Zone passed penetration test within evaluation window
            return True, f"Zone untested - passed {wick_threshold*100:.0f}% wick penetration test (eval window: {evaluation_window} candles)"
            
        except KeyError as e:
            return False, f"Missing zone data: {str(e)}"
        except Exception as e:
            return False, f"Penetration test error: {str(e)}"
    
    def execute_backtest_with_penetration_strategy(self, data: pd.DataFrame, patterns: Dict,
                                         trend_data: pd.DataFrame, pair: str, 
                                         timeframe: str, strategy_name: str) -> Dict:
        """
        Execute backtest with walk-forward penetration testing
        FIXED: Tests penetration status at the time of potential trade execution
        """
        try:
            strategy_config = self.penetration_strategies[strategy_name]
            wick_threshold = strategy_config['wick_threshold']
            
            # Get all patterns (momentum + reversal)
            all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                        patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', []))
            
            # FIXED: Filter patterns using walk-forward penetration testing
            valid_patterns = []
            for pattern in all_patterns:
                if pattern.get('end_idx') is not None:
                    # For walk-forward testing, check penetration status at zone formation + buffer
                    # This simulates real-time decision making
                    evaluation_point = pattern['end_idx'] + 10  # 10 candles after formation
                    
                    if evaluation_point < len(data):
                        limited_data = data.iloc[:evaluation_point + 1]  # Only data up to evaluation point
                        is_valid, reason = self.check_zone_testing_with_penetration(
                            pattern, limited_data, wick_threshold
                        )
                        if is_valid:
                            valid_patterns.append(pattern)
                    else:
                        # Zone too recent, assume valid for testing
                        valid_patterns.append(pattern)
            
            print(f"      {strategy_name}: {len(valid_patterns)}/{len(all_patterns)} zones passed walk-forward penetration test")
            
            if not valid_patterns:
                return self.create_empty_result_with_strategy(pair, timeframe, strategy_name, 
                                                            f"No zones passed {strategy_name} walk-forward penetration test")
            
            # Execute trades using parent class method (DO NOT MODIFY TRADE LOGIC)
            trades = self.execute_realistic_trades(valid_patterns, data, trend_data, timeframe)
            
            # Calculate performance with strategy info
            result = self.calculate_performance_metrics(trades, pair, timeframe)
            result['strategy'] = strategy_name
            result['strategy_description'] = strategy_config['description']
            result['wick_threshold'] = wick_threshold
            result['zones_tested'] = len(all_patterns)
            result['zones_passed'] = len(valid_patterns)
            result['zone_filter_rate'] = (len(all_patterns) - len(valid_patterns)) / len(all_patterns) * 100 if all_patterns else 0
            
            return result
            
        except Exception as e:
            return self.create_empty_result_with_strategy(pair, timeframe, strategy_name, f"Strategy error: {str(e)}")
    
    def run_single_penetration_test(self, pair: str, timeframe: str, days_back: int = 730) -> List[Dict]:
        """
        Run all 7 penetration strategies on single pair/timeframe
        
        Args:
            pair: Currency pair
            timeframe: Timeframe
            days_back: Days of historical data
            
        Returns:
            List of results for all 7 strategies
        """
        try:
            print(f"\n🧪 PENETRATION TEST: {pair} {timeframe} ({days_back} days)")
            print("=" * 60)
            
            # Load data using parent class method
            data = self.load_data_with_validation(pair, timeframe, days_back)
            if data is None:
                return [self.create_empty_result_with_strategy(pair, timeframe, strategy, "Insufficient data") 
                       for strategy in self.penetration_strategies.keys()]
            
            # Initialize components using parent class pattern
            candle_classifier = CandleClassifier(data)
            classified_data = candle_classifier.classify_all_candles()
            
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            trend_classifier = TrendClassifier(data)
            trend_data = trend_classifier.classify_trend_with_filter()
            
            print(f"   📊 Found {patterns['total_patterns']} total zones")
            
            # Test all 7 penetration strategies
            results = []
            for strategy_name in self.penetration_strategies.keys():
                result = self.execute_backtest_with_penetration_strategy(
                    data, patterns, trend_data, pair, timeframe, strategy_name
                )
                results.append(result)
            
            # Print strategy comparison
            self.print_penetration_summary(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Single test error: {str(e)}"
            return [self.create_empty_result_with_strategy(pair, timeframe, strategy, error_msg) 
                   for strategy in self.penetration_strategies.keys()]
    
    def run_comprehensive_penetration_analysis(self, analysis_period: str = 'priority_1') -> List[Dict]:
        """
        Run comprehensive penetration analysis across all pairs/timeframes/strategies
        
        Args:
            analysis_period: Analysis period key from ANALYSIS_PERIODS
            
        Returns:
            List of all test results
        """
        print(f"\n🚀 COMPREHENSIVE PENETRATION ANALYSIS - {analysis_period.upper()}")
        period_config = ANALYSIS_PERIODS[analysis_period]
        days_back = period_config['days_back']
        
        print(f"📊 Period: {period_config['name']}")
        print(f"📅 Days back: {days_back:,}")
        print(f"🧪 Testing 7 penetration strategies")
        print("=" * 70)
        
        # Discover valid data combinations using parent class method
        valid_combinations = self.discover_valid_data_combinations()
        if not valid_combinations:
            print("❌ No valid data combinations found")
            return []
        
        # Create test combinations for all strategies
        test_combinations = []
        for pair, timeframe in valid_combinations:
            for strategy_name in self.penetration_strategies.keys():
                test_combinations.append({
                    'pair': pair,
                    'timeframe': timeframe,
                    'strategy': strategy_name,
                    'days_back': days_back,
                    'analysis_period': analysis_period
                })
        
        total_tests = len(test_combinations)
        print(f"📊 Valid combinations: {len(valid_combinations)}")
        print(f"📊 Total tests (7 strategies × combinations): {total_tests:,}")
        
        # Run optimized parallel processing
        all_results = self.run_optimized_parallel_penetration_tests(test_combinations)
        
        # Generate comprehensive Excel report
        if all_results:
            self.generate_penetration_excel_report(all_results, analysis_period, period_config)
        
        # Print final summary
        successful_results = [r for r in all_results if r['total_trades'] > 0]
        print(f"\n🎯 PENETRATION ANALYSIS COMPLETE:")
        print(f"   Total tests: {len(all_results):,}")
        print(f"   Successful tests: {len(successful_results):,}")
        print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        
        if successful_results:
            # Best strategy overall
            best = max(successful_results, key=lambda x: x['profit_factor'])
            print(f"   🏆 Best: {best['strategy']} - {best['pair']} {best['timeframe']} - PF {best['profit_factor']:.2f}")
            
            # Strategy performance summary
            strategy_performance = {}
            for result in successful_results:
                strategy = result['strategy']
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(result['profit_factor'])
            
            print(f"\n📊 STRATEGY PERFORMANCE SUMMARY:")
            for strategy, pf_list in strategy_performance.items():
                avg_pf = np.mean(pf_list)
                count = len(pf_list)
                print(f"   {strategy}: {avg_pf:.2f} avg PF ({count} successful tests)")
        
        return all_results
    
    def run_optimized_parallel_penetration_tests(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run penetration tests in parallel with memory management
        """
        print(f"\n🔄 OPTIMIZED PARALLEL PENETRATION TESTING")
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
                    chunk_results = pool.map(run_penetration_test_worker, chunk_tests)
                    results.extend(chunk_results)
                
                # Progress tracking
                completed = chunk_end
                progress = (completed / len(test_combinations)) * 100
                print(f"✅ Progress: {progress:.1f}% ({completed}/{len(test_combinations)})")
                
            except Exception as e:
                print(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Add empty results for failed chunk
                for test in chunk_tests:
                    results.append(self.create_empty_result_with_strategy(
                        test['pair'], test['timeframe'], test['strategy'], 
                        f"Parallel processing error: {str(e)}"
                    ))
            
            # Memory cleanup after each chunk
            gc.collect()
        
        total_time = time.time() - start_time
        success_count = len([r for r in results if r.get('total_trades', 0) > 0])
        
        print(f"\n✅ PARALLEL PENETRATION TESTING COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🎯 Success rate: {success_count}/{len(test_combinations)} ({success_count/len(test_combinations)*100:.1f}%)")
        print(f"⚡ Speed: {len(test_combinations)/total_time:.1f} tests/second")
        
        return results
    
    def generate_penetration_excel_report(self, all_results: List[Dict], 
                                        analysis_period: str, period_config: Dict):
        """
        Generate professional 5-sheet Excel report for penetration analysis
        """
        print(f"\n📊 GENERATING PENETRATION EXCEL REPORT...")
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_results)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_period}_penetration_analysis_{timestamp}.xlsx"
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
                    
                    # SHEET 3: Strategy Analysis
                    strategy_analysis = self.create_strategy_analysis(successful_df)
                    strategy_analysis.to_excel(writer, sheet_name='Strategy_Analysis', index=False)
                    print("   ✅ Sheet 3: Strategy Analysis")
                    
                    # SHEET 4: Timeframe Analysis
                    timeframe_analysis = self.create_timeframe_analysis(successful_df)
                    timeframe_analysis.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    print("   ✅ Sheet 4: Timeframe Analysis")
                    
                    # SHEET 5: Pair Analysis
                    pair_analysis = self.create_pair_analysis(successful_df)
                    pair_analysis.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ✅ Sheet 5: Pair Analysis")
                    
                else:
                    # Create empty analysis sheets
                    empty_df = pd.DataFrame({'Note': ['No successful results to analyze']})
                    empty_df.to_excel(writer, sheet_name='Successful_Results', index=False)
                    empty_df.to_excel(writer, sheet_name='Strategy_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Timeframe_Analysis', index=False)
                    empty_df.to_excel(writer, sheet_name='Pair_Analysis', index=False)
                    print("   ⚠️  Empty analysis sheets (no successful results)")
            
            print(f"📁 PENETRATION EXCEL REPORT SAVED: {filename}")
            print(f"📊 5 comprehensive analysis sheets created")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")
            # Fallback: Save as CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df_all.to_csv(csv_filename, index=False)
            print(f"📁 Fallback CSV saved: {csv_filename}")
    
    def create_strategy_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create strategy performance analysis"""
        try:
            strategy_analysis = df.groupby('strategy').agg({
                'profit_factor': ['mean', 'count'],
                'win_rate': 'mean',
                'total_trades': 'sum',
                'total_return': 'mean',
                'zone_filter_rate': 'mean'
            }).round(2)
            
            # Flatten column names
            strategy_analysis.columns = ['Avg_Profit_Factor', 'Test_Count', 'Avg_Win_Rate', 
                                       'Total_Trades', 'Avg_Return', 'Avg_Filter_Rate']
            strategy_analysis = strategy_analysis.sort_values('Avg_Profit_Factor', ascending=False)
            
            return strategy_analysis.reset_index()
            
        except Exception as e:
            print(f"   ⚠️  Strategy analysis error: {str(e)}")
            return pd.DataFrame({'Strategy': ['Error'], 'Note': [str(e)]})
    
    def print_penetration_summary(self, results: List[Dict]):
        """Print console summary of penetration test results"""
        print(f"\n📊 PENETRATION STRATEGY COMPARISON:")
        print("=" * 70)
        
        # Create comparison table
        summary_data = []
        for result in results:
            if result['total_trades'] > 0:
                summary_data.append({
                    'Strategy': result['strategy'],
                    'Trades': result['total_trades'],
                    'Win_Rate': f"{result['win_rate']:.1f}%",
                    'Profit_Factor': f"{result['profit_factor']:.2f}",
                    'Return': f"{result['total_return']:.1f}%",
                    'Zones_Passed': f"{result['zones_passed']}/{result['zones_tested']}",
                    'Filter_Rate': f"{result['zone_filter_rate']:.1f}%"
                })
        
        if summary_data:
            # Print formatted table
            headers = ['Strategy', 'Trades', 'Win_Rate', 'Profit_Factor', 'Return', 'Zones_Passed', 'Filter_Rate']
            
            # Calculate column widths
            col_widths = {}
            for header in headers:
                col_widths[header] = max(len(header), max(len(str(row[header])) for row in summary_data))
            
            # Print header
            header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
            print(header_row)
            print("-" * len(header_row))
            
            # Print data rows
            for row in summary_data:
                data_row = " | ".join(str(row[header]).ljust(col_widths[header]) for header in headers)
                print(data_row)
            
            # Identify best performer
            best_pf = max(summary_data, key=lambda x: float(x['Profit_Factor'].replace('X', '999')))
            print(f"\n🏆 BEST PERFORMER: {best_pf['Strategy']} (PF: {best_pf['Profit_Factor']})")
            
        else:
            print("   No successful strategies with trades")
    
    def export_penetration_results(self, results: List[Dict], pair: str, timeframe: str) -> str:
        """Export single test penetration results to CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/penetration_{pair}_{timeframe}_{timestamp}.csv"
        
        os.makedirs('results', exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        
        print(f"💾 Penetration results exported: {filename}")
        return filename
    
    def create_empty_result_with_strategy(self, pair: str, timeframe: str, 
                                        strategy: str, reason: str) -> Dict:
        """Create empty result with strategy information"""
        result = self.create_empty_result(pair, timeframe, reason)
        result.update({
            'strategy': strategy,
            'strategy_description': self.penetration_strategies.get(strategy, {}).get('description', ''),
            'wick_threshold': self.penetration_strategies.get(strategy, {}).get('wick_threshold'),
            'zones_tested': 0,
            'zones_passed': 0,
            'zone_filter_rate': 0.0
        })
        return result

# ============================================================================
# PARALLEL PROCESSING WORKER FUNCTION
# ============================================================================

def run_penetration_test_worker(test_config: Dict) -> Dict:
    """
    Worker function for parallel penetration testing
    Each worker creates its own backtester instance
    """
    try:
        # Create fresh backtester instance for this worker
        backtester = ZonePenetrationBacktester()
        
        # Load data using parent class method
        data = backtester.load_data_with_validation(
            test_config['pair'], 
            test_config['timeframe'], 
            test_config['days_back']
        )
        
        if data is None:
            return backtester.create_empty_result_with_strategy(
                test_config['pair'], 
                test_config['timeframe'], 
                test_config['strategy'],
                "Insufficient data"
            )
        
        # Initialize components
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        trend_classifier = TrendClassifier(data)
        trend_data = trend_classifier.classify_trend_with_filter()
        
        # Execute backtest with specific penetration strategy
        result = backtester.execute_backtest_with_penetration_strategy(
            data, patterns, trend_data,
            test_config['pair'], 
            test_config['timeframe'], 
            test_config['strategy']
        )
        
        # Add analysis period info
        result['analysis_period'] = test_config['analysis_period']
        
        # Clean up
        del backtester
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return {
            'pair': test_config['pair'],
            'timeframe': test_config['timeframe'],
            'strategy': test_config['strategy'],
            'analysis_period': test_config['analysis_period'],
            'total_trades': 0,
            'zones_tested': 0,
            'zones_passed': 0,
            'zone_filter_rate': 0.0,
            'description': f"Worker error: {str(e)}"
        }

def main():
    """Main function with penetration testing menu"""
    print("🧪 ZONE PENETRATION BACKTESTING ENGINE")
    print("=" * 60)
    
    # Check system resources using parent class method
    engine = ZonePenetrationBacktester()
    if not engine.check_system_resources():
        print("❌ Insufficient system resources")
        return
    
    print("\n🎯 SELECT ANALYSIS MODE:")
    print("1. Quick Validation (Single test - EURUSD 3D)")
    print("2. Comprehensive Analysis - Priority 1 (2015-2025, All pairs/timeframes)")
    print("3. Comprehensive Analysis - Priority 2 (2020-2025, All pairs/timeframes)")
    print("4. Comprehensive Analysis - Priority 3 (2018-2025, All pairs/timeframes)")
    print("5. Custom Single Test")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        # Quick validation test
        print("\n🧪 QUICK PENETRATION VALIDATION:")
        print("Testing EURUSD 3D with all 7 penetration strategies...")
        
        results = engine.run_single_penetration_test('EURUSD', '3D', 730)
        
        if any(r['total_trades'] > 0 for r in results):
            print(f"\n✅ VALIDATION SUCCESSFUL - Penetration testing working!")
            engine.export_penetration_results(results, 'EURUSD', '3D')
        else:
            print(f"\n⚠️  No trades generated across all strategies")
    
    elif choice == '2':
        # Comprehensive analysis - Priority 1
        print("\n🚀 COMPREHENSIVE PENETRATION ANALYSIS - PRIORITY 1")
        print("This will test ALL pairs, timeframes, and 7 penetration strategies")
        print("Expected tests: ~2,000-5,000 depending on available data")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_comprehensive_penetration_analysis('priority_1')
        else:
            print("Analysis cancelled")
    
    elif choice == '3':
        # Comprehensive analysis - Priority 2
        print("\n🚀 COMPREHENSIVE PENETRATION ANALYSIS - PRIORITY 2")
        print("This will test ALL pairs, timeframes, and 7 penetration strategies")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_comprehensive_penetration_analysis('priority_2')
        else:
            print("Analysis cancelled")
    
    elif choice == '4':
        # Comprehensive analysis - Priority 3
        print("\n🚀 COMPREHENSIVE PENETRATION ANALYSIS - PRIORITY 3")
        print("This will test ALL pairs, timeframes, and 7 penetration strategies")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_comprehensive_penetration_analysis('priority_3')
        else:
            print("Analysis cancelled")
    
    elif choice == '5':
        # Custom single test
        print("\n🎯 CUSTOM PENETRATION TEST:")
        pairs = engine.discover_all_pairs()
        print(f"Available pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
        
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = input("Enter days back (e.g., 730): ").strip()
        
        try:
            days_back = int(days_back)
            results = engine.run_single_penetration_test(pair, timeframe, days_back)
            # Export results
            engine.export_penetration_results(results, pair, timeframe)
            
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
   main()