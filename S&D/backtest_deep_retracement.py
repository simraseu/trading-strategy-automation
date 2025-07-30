"""
Deep Retracement Backtesting Engine - Entry Price Variation Analysis
INHERITS from CoreBacktestEngine - Tests IDENTICAL trades with different entry depths
Built on 100% identical zone detection, validation, and trade management
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from core_backtest_engine import CoreBacktestEngine, run_single_test_worker, ANALYSIS_PERIODS

class DeepRetracementBacktestEngine(CoreBacktestEngine):
    """
    DEEP RETRACEMENT ENTRY ANALYSIS ENGINE
    Inherits ALL functionality from CoreBacktestEngine
    ONLY modifies entry price calculation - everything else identical
    """
    
    def __init__(self):
        """Initialize with parent functionality"""
        super().__init__()
        
        # Entry method configurations
        self.entry_methods = {
            'Current_Zone_Boundary': {
                'name': 'Zone Boundary + 5% Front-run (Baseline)',
                'description': 'Standard entry at zone edge with front-running'
            },
            'Deep_Retracement_Base_Close': {
                'name': 'Base Close + 5% Front-run (Deep)',
                'description': 'Entry at deepest base close with front-running'
            },
            'Ultra_Deep_Base_Close_No_Frontrun': {
                'name': 'Base Close Exact (Ultra Deep)',
                'description': 'Entry at exact deepest base close level'
            }
        }
    
    def run_entry_method_comparison(self, pair: str, timeframe: str, days_back: int = 730) -> Dict:
        """
        Run IDENTICAL trades with different entry methods
        CRITICAL: Same zones, same validation, same timing - only entry price varies
        """
        print(f"\n🧪 DEEP RETRACEMENT ENTRY ANALYSIS: {pair} {timeframe}")
        print("=" * 60)
        
        # Load data using parent method (identical)
        data = self.load_data_with_validation(pair, timeframe, days_back)
        if data is None:
            return self.create_empty_comparison_result(pair, timeframe, "Insufficient data")
        
        # Initialize components using parent methods (identical)
        from modules.candle_classifier import CandleClassifier
        from modules.zone_detector import ZoneDetector
        from modules.trend_classifier import TrendClassifier
        from modules.risk_manager import RiskManager
        
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        trend_classifier = TrendClassifier(data)
        trend_data = trend_classifier.classify_trend_with_filter()
        
        risk_manager = RiskManager(account_balance=10000)
        
        # Run comparison across all entry methods
        results = {}
        for method_key, method_config in self.entry_methods.items():
            print(f"\n📊 Testing Entry Method: {method_config['name']}")
            
            # Set current entry method
            self.current_entry_method = method_key
            
            result = self.execute_backtest_with_updated_logic(
                data, patterns, trend_data, risk_manager, pair, timeframe
            )
            
            result['entry_method'] = method_key
            result['entry_method_name'] = method_config['name']
            result['entry_method_description'] = method_config['description']
            
            results[method_key] = result
            
            print(f"   Trades: {result['total_trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Profit Factor: {result['profit_factor']:.2f}")
        
        return self.compile_comparison_results(results, pair, timeframe)
    
    def execute_single_realistic_trade(self, zone: Dict, data: pd.DataFrame, current_idx: int) -> Optional[Dict]:
        """
        OVERRIDE: Modified entry price calculation only
        Everything else identical to parent class
        """
        zone_high = zone['zone_high']
        zone_low = zone['zone_low']
        zone_range = zone_high - zone_low
        
        # MODIFIED: Entry calculation based on current method
        if zone['type'] in ['R-B-R', 'D-B-R']:  # Demand zones (buy)
            direction = 'BUY'
            
            if self.current_entry_method == 'Current_Zone_Boundary':
                # BASELINE: Zone boundary + 5% front-run
                entry_price = zone_high + (zone_range * 0.05)
                
            elif self.current_entry_method == 'Deep_Retracement_Base_Close':
                # DEEP: Highest base close + 5% front-run
                entry_price = self.calculate_deep_retracement_entry(zone, data, 'demand', True)
                
            elif self.current_entry_method == 'Ultra_Deep_Base_Close_No_Frontrun':
                # ULTRA DEEP: Highest base close exact
                entry_price = self.calculate_deep_retracement_entry(zone, data, 'demand', False)
            
            # Stop logic identical to parent
            initial_stop = zone_low - (zone_range * 0.33)
            
        elif zone['type'] in ['D-B-D', 'R-B-D']:  # Supply zones (sell)
            direction = 'SELL'
            
            if self.current_entry_method == 'Current_Zone_Boundary':
                # BASELINE: Zone boundary - 5% front-run
                entry_price = zone_low - (zone_range * 0.05)
                
            elif self.current_entry_method == 'Deep_Retracement_Base_Close':
                # DEEP: Lowest base close - 5% front-run
                entry_price = self.calculate_deep_retracement_entry(zone, data, 'supply', True)
                
            elif self.current_entry_method == 'Ultra_Deep_Base_Close_No_Frontrun':
                # ULTRA DEEP: Lowest base close exact
                entry_price = self.calculate_deep_retracement_entry(zone, data, 'supply', False)
            
            # Stop logic identical to parent
            initial_stop = zone_high + (zone_range * 0.33)
        else:
            return None
        
        # Check if current candle can trigger entry (limit order logic - identical to parent)
        current_candle = data.iloc[current_idx]
        
        can_enter = False
        if direction == 'BUY':
            if current_candle['high'] >= entry_price:
                can_enter = True
        elif direction == 'SELL':
            if current_candle['low'] <= entry_price:
                can_enter = True
        
        if not can_enter:
            return None
        
        # ALL remaining logic identical to parent class
        from config.settings import RISK_CONFIG
        
        risk_amount = 10000 * (RISK_CONFIG['risk_limits']['max_risk_per_trade'] / 100)
        pip_value = self.get_pip_value_for_pair(zone.get('pair', 'EURUSD'))
        stop_distance_pips = abs(entry_price - initial_stop) / pip_value
        
        if stop_distance_pips <= 0:
            return None
        
        position_size = risk_amount / stop_distance_pips
        
        # Set targets (identical to parent)
        risk_distance = abs(entry_price - initial_stop)
        target_rr = RISK_CONFIG['take_profit_rules']['risk_reward_ratio']
        
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * target_rr)
        else:
            target_price = entry_price - (risk_distance * target_rr)
        
        # Simulate outcome using parent method (identical)
        return self.simulate_realistic_outcome(
            entry_price, initial_stop, target_price, direction, 
            position_size, data, current_idx, zone['type'], stop_distance_pips, zone.get('pair', 'EURUSD'),
            zone_high, zone_low, zone
        )
    
    def calculate_deep_retracement_entry(self, zone: Dict, data: pd.DataFrame, 
                                       zone_direction: str, use_frontrun: bool) -> float:
        """
        Calculate deep retracement entry based on base candle closes
        """
        try:
            # Extract base candle indices
            base_info = zone.get('base', {})
            base_start_idx = base_info.get('start_idx')
            base_end_idx = base_info.get('end_idx')
            
            if base_start_idx is None or base_end_idx is None:
                # Fallback to zone boundary if base info missing
                zone_high = zone['zone_high']
                zone_low = zone['zone_low']
                zone_range = zone_high - zone_low
                
                if zone_direction == 'demand':
                    return zone_high + (zone_range * 0.05 if use_frontrun else 0)
                else:
                    return zone_low - (zone_range * 0.05 if use_frontrun else 0)
            
            # Get base candles data
            base_data = data.iloc[base_start_idx:base_end_idx+1]
            zone_range = zone['zone_high'] - zone['zone_low']
            
            if zone_direction == 'demand':
                # Demand: Find highest close in base
                highest_base_close = base_data['close'].max()
                if use_frontrun:
                    return highest_base_close + (zone_range * 0.05)
                else:
                    return highest_base_close
            else:
                # Supply: Find lowest close in base
                lowest_base_close = base_data['close'].min()
                if use_frontrun:
                    return lowest_base_close - (zone_range * 0.05)
                else:
                    return lowest_base_close
                    
        except Exception as e:
            print(f"   ⚠️  Deep retracement calculation error: {str(e)}")
            # Fallback to zone boundary
            zone_high = zone['zone_high']
            zone_low = zone['zone_low']
            zone_range = zone_high - zone_low
            
            if zone_direction == 'demand':
                return zone_high + (zone_range * 0.05 if use_frontrun else 0)
            else:
                return zone_low - (zone_range * 0.05 if use_frontrun else 0)
    
    def compile_comparison_results(self, results: Dict, pair: str, timeframe: str) -> Dict:
        """
        Compile comparison results across all entry methods
        """
        comparison_data = []
        
        for method_key, result in results.items():
            comparison_data.append({
                'pair': pair,
                'timeframe': timeframe,
                'entry_method': method_key,
                'entry_method_name': result['entry_method_name'],
                'total_trades': result['total_trades'],
                'winning_trades': result['winning_trades'],
                'losing_trades': result['losing_trades'],
                'breakeven_trades': result['breakeven_trades'],
                'win_rate': result['win_rate'],
                'loss_rate': result['loss_rate'],
                'be_rate': result['be_rate'],
                'profit_factor': result['profit_factor'],
                'total_pnl': result['total_pnl'],
                'gross_profit': result['gross_profit'],
                'gross_loss': result['gross_loss'],
                'total_return': result['total_return'],
                'avg_trade_duration': result['avg_trade_duration']
            })
        
        return {
            'pair': pair,
            'timeframe': timeframe,
            'comparison_data': comparison_data,
            'individual_results': results,
            'analysis_type': 'entry_method_comparison'
        }
    
    def create_empty_comparison_result(self, pair: str, timeframe: str, reason: str) -> Dict:
        """Create empty comparison result structure"""
        return {
            'pair': pair,
            'timeframe': timeframe,
            'comparison_data': [],
            'individual_results': {},
            'analysis_type': 'entry_method_comparison',
            'description': reason
        }
    
    def run_comprehensive_entry_analysis(self, analysis_period: str = 'priority_1') -> List[Dict]:
        """
        Run comprehensive entry method analysis across all pairs/timeframes
        """
        print(f"\n🚀 COMPREHENSIVE DEEP RETRACEMENT ANALYSIS - {analysis_period.upper()}")
        period_config = ANALYSIS_PERIODS[analysis_period]
        days_back = period_config['days_back']
        
        print(f"📊 Period: {period_config['name']}")
        print("=" * 70)
        
        # Discover valid combinations using parent method
        valid_combinations = self.discover_valid_data_combinations()
        if not valid_combinations:
            print("❌ No valid data combinations found")
            return []
        
        # Create test combinations
        test_combinations = []
        for pair, timeframe in valid_combinations:
            test_combinations.append({
                'pair': pair,
                'timeframe': timeframe,
                'days_back': days_back,
                'analysis_period': analysis_period,
                'analysis_type': 'entry_comparison'
            })
        
        print(f"📊 Testing {len(test_combinations)} combinations with {len(self.entry_methods)} entry methods each")
        print(f"📊 Total tests: {len(test_combinations) * len(self.entry_methods):,}")
        
        # Run parallel analysis
        all_results = self.run_optimized_entry_analysis(test_combinations)
        
        # Generate comprehensive Excel report
        if all_results:
            self.generate_entry_comparison_excel_report(all_results, analysis_period, period_config)
        
        return all_results
    
    def run_optimized_entry_analysis(self, test_combinations: List[Dict]) -> List[Dict]:
        """
        Run entry method comparison tests with optimization
        """
        print(f"\n🔄 OPTIMIZED ENTRY ANALYSIS EXECUTION")
        print(f"⚡ Workers: {self.max_workers}")
        
        results = []
        
        # Process combinations individually (each runs all 3 entry methods)
        for i, test_config in enumerate(test_combinations):
            print(f"\n📦 Processing {i+1}/{len(test_combinations)}: {test_config['pair']} {test_config['timeframe']}")
            
            try:
                comparison_result = self.run_entry_method_comparison(
                    test_config['pair'],
                    test_config['timeframe'],
                    test_config['days_back']
                )
                
                comparison_result['analysis_period'] = test_config['analysis_period']
                results.append(comparison_result)
                
                # Progress tracking
                progress = ((i + 1) / len(test_combinations)) * 100
                print(f"✅ Progress: {progress:.1f}% ({i+1}/{len(test_combinations)})")
                
            except Exception as e:
                print(f"❌ Error processing {test_config['pair']} {test_config['timeframe']}: {str(e)}")
                results.append(self.create_empty_comparison_result(
                    test_config['pair'], 
                    test_config['timeframe'], 
                    f"Processing error: {str(e)}"
                ))
        
        print(f"\n✅ ENTRY ANALYSIS COMPLETE!")
        successful = len([r for r in results if r.get('comparison_data')])
        print(f"🎯 Success rate: {successful}/{len(test_combinations)} ({successful/len(test_combinations)*100:.1f}%)")
        
        return results
    
    def generate_entry_comparison_excel_report(self, all_results: List[Dict], 
                                             analysis_period: str, period_config: Dict):
        """
        Generate comprehensive Excel report for entry method comparison
        """
        print(f"\n📊 GENERATING ENTRY COMPARISON EXCEL REPORT...")
        
        # Flatten results for Excel export
        flattened_data = []
        for result in all_results:
            if result.get('comparison_data'):
                for entry_data in result['comparison_data']:
                    entry_data['analysis_period'] = result.get('analysis_period', analysis_period)
                    flattened_data.append(entry_data)
        
        if not flattened_data:
            print("❌ No data to export")
            return
        
        df_all = pd.DataFrame(flattened_data)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{analysis_period}_deep_retracement_analysis_{timestamp}.xlsx"
        os.makedirs('results', exist_ok=True)
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # SHEET 1: All Results
                df_all.to_excel(writer, sheet_name='All_Entry_Methods', index=False)
                print("   ✅ Sheet 1: All Entry Methods")
                
                # SHEET 2: Entry Method Comparison
                successful_df = df_all[df_all['total_trades'] > 0]
                if len(successful_df) > 0:
                    
                    # Entry method performance analysis
                    entry_analysis = successful_df.groupby('entry_method').agg({
                        'profit_factor': ['mean', 'count'],
                        'win_rate': 'mean',
                        'total_trades': 'sum',
                        'total_return': 'mean',
                        'avg_trade_duration': 'mean'
                    }).round(2)
                    
                    entry_analysis.columns = ['Avg_Profit_Factor', 'Strategy_Count', 'Avg_Win_Rate', 
                                            'Total_Trades', 'Avg_Return', 'Avg_Duration']
                    entry_analysis = entry_analysis.sort_values('Avg_Profit_Factor', ascending=False)
                    entry_analysis.to_excel(writer, sheet_name='Entry_Method_Analysis', index=True)
                    print("   ✅ Sheet 2: Entry Method Analysis")
                    
                    # SHEET 3: Pair Analysis by Entry Method
                    pair_entry_analysis = successful_df.groupby(['pair', 'entry_method']).agg({
                        'profit_factor': 'mean',
                        'win_rate': 'mean',
                        'total_trades': 'sum',
                        'total_return': 'mean'
                    }).round(2)
                    pair_entry_analysis.to_excel(writer, sheet_name='Pair_Entry_Analysis', index=True)
                    print("   ✅ Sheet 3: Pair-Entry Analysis")
                    
                    # SHEET 4: Timeframe Analysis by Entry Method
                    tf_entry_analysis = successful_df.groupby(['timeframe', 'entry_method']).agg({
                        'profit_factor': 'mean',
                        'win_rate': 'mean',
                        'total_trades': 'sum',
                        'total_return': 'mean'
                    }).round(2)
                    tf_entry_analysis.to_excel(writer, sheet_name='Timeframe_Entry_Analysis', index=True)
                    print("   ✅ Sheet 4: Timeframe-Entry Analysis")
                
            print(f"📁 ENTRY COMPARISON REPORT SAVED: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating Excel report: {str(e)}")


def main():
    """Main function for deep retracement analysis"""
    print("🎯 DEEP RETRACEMENT ENTRY ANALYSIS - INHERITS FROM CORE ENGINE")
    print("=" * 70)
    
    engine = DeepRetracementBacktestEngine()
    
    print("\n🎯 SELECT ANALYSIS MODE:")
    print("1. Quick Entry Comparison (Single test - EURUSD 3D)")
    print("2. Comprehensive Entry Analysis - Priority 1 (2015-2025)")
    print("3. Custom Single Entry Comparison")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Quick validation
        print("\n🧪 QUICK ENTRY METHOD COMPARISON:")
        result = engine.run_entry_method_comparison('EURUSD', '3D', 730)
        
        if result.get('comparison_data'):
            print(f"\n📊 ENTRY METHOD COMPARISON RESULTS:")
            for entry_data in result['comparison_data']:
                print(f"\n   {entry_data['entry_method_name']}:")
                print(f"     Trades: {entry_data['total_trades']}")
                print(f"     Win Rate: {entry_data['win_rate']:.1f}%")
                print(f"     Profit Factor: {entry_data['profit_factor']:.2f}")
                print(f"     Total Return: {entry_data['total_return']:.2f}%")
                print(f"     Avg Duration: {entry_data['avg_trade_duration']:.1f} days")
        else:
            print(f"❌ No comparison data: {result.get('description', 'Unknown error')}")
    
    elif choice == '2':
        # Comprehensive analysis
        print("\n🚀 COMPREHENSIVE ENTRY METHOD ANALYSIS")
        print("This will compare all 3 entry methods across all pairs/timeframes")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            engine.run_comprehensive_entry_analysis('priority_1')
        else:
            print("Analysis cancelled")
    
    elif choice == '3':
        # Custom test
        pairs = engine.discover_all_pairs()
        print(f"Available pairs: {', '.join(pairs[:10])}..." if len(pairs) > 10 else f"Available pairs: {', '.join(pairs)}")
        
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        days_back = input("Enter days back (e.g., 730): ").strip()
        
        try:
            days_back = int(days_back)
            result = engine.run_entry_method_comparison(pair, timeframe, days_back)
            
            if result.get('comparison_data'):
                print(f"\n📊 CUSTOM ENTRY METHOD COMPARISON:")
                for entry_data in result['comparison_data']:
                    print(f"\n   {entry_data['entry_method_name']}:")
                    print(f"     Trades: {entry_data['total_trades']}")
                    print(f"     Win Rate: {entry_data['win_rate']:.1f}%")
                    print(f"     Profit Factor: {entry_data['profit_factor']:.2f}")
                    print(f"     Total Return: {entry_data['total_return']:.2f}%")
            else:
                print(f"❌ {result.get('description', 'No data available')}")
                
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()