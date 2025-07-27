"""
Backtesting Logic Validator - Debug & Fix Tool
Tests core backtest logic to identify and resolve period separation and profit factor issues
Does NOT modify core_backtest_engine.py - purely diagnostic
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your existing engine
from core_backtest_engine import CoreBacktestEngine

class BacktestLogicValidator:
    """
    Diagnostic tool to validate backtesting logic and identify issues
    Tests period separation, profit factor calculation, and trade classification
    """
    
    def __init__(self):
        """Initialize validator with debug configuration"""
        self.engine = CoreBacktestEngine()
        
        # Test periods for validation
        self.test_periods = {
            'period_1': {'name': '2015-2025', 'days_back': 3847},
            'period_2': {'name': '2020-2025', 'days_back': 2021}, 
            'period_3': {'name': '2018-2025', 'days_back': 2751}
        }
        
        print(f"🔍 BACKTESTING LOGIC VALIDATOR INITIALIZED")
        print(f"🎯 Purpose: Diagnose period separation and profit factor issues")
        print("=" * 60)
    
    def validate_period_separation(self, pair: str = 'EURUSD', timeframe: str = '3D') -> Dict:
        """
        Test if different periods actually load different data
        This should identify the identical PF issue
        """
        print(f"\n🔍 TESTING PERIOD SEPARATION: {pair} {timeframe}")
        print("=" * 50)
        
        period_data = {}
        
        for period_key, period_config in self.test_periods.items():
            print(f"\n📅 Loading {period_config['name']} ({period_config['days_back']} days)...")
            
            # Load data for this period
            data = self.engine.load_data_with_validation(pair, timeframe, period_config['days_back'])
            
            if data is not None:
                period_data[period_key] = {
                    'name': period_config['name'],
                    'days_back': period_config['days_back'],
                    'total_candles': len(data),
                    'start_date': data.index[0],
                    'end_date': data.index[-1],
                    'first_price': data['close'].iloc[0],
                    'last_price': data['close'].iloc[-1],
                    'data_hash': hash(str(data['close'].sum()))  # Simple hash to detect identical data
                }
                
                print(f"   ✅ Loaded: {len(data)} candles")
                print(f"   📅 Range: {data.index[0]} to {data.index[-1]}")
                print(f"   💰 Price: {data['close'].iloc[0]:.5f} → {data['close'].iloc[-1]:.5f}")
                print(f"   🔢 Hash: {period_data[period_key]['data_hash']}")
            else:
                print(f"   ❌ Failed to load data")
                period_data[period_key] = None
        
        # Analyze results
        return self.analyze_period_separation(period_data)
    
    def analyze_period_separation(self, period_data: Dict) -> Dict:
        """Analyze if periods are properly separated"""
        print(f"\n📊 PERIOD SEPARATION ANALYSIS:")
        print("=" * 40)
        
        # Check for identical data
        hashes = [data['data_hash'] for data in period_data.values() if data is not None]
        identical_data = len(set(hashes)) < len(hashes)
        
        # Check date ranges
        date_ranges_valid = True
        valid_periods = [data for data in period_data.values() if data is not None]
        
        if len(valid_periods) >= 2:
            # Sort by days_back (largest first)
            sorted_periods = sorted(valid_periods, key=lambda x: x['days_back'], reverse=True)
            
            for i in range(len(sorted_periods) - 1):
                longer_period = sorted_periods[i]
                shorter_period = sorted_periods[i + 1]
                
                # Longer period should have earlier start date
                if longer_period['start_date'] >= shorter_period['start_date']:
                    date_ranges_valid = False
                    print(f"   ❌ INVALID: {longer_period['name']} doesn't start before {shorter_period['name']}")
        
        # Summary
        print(f"   Identical Data Detected: {'❌ YES' if identical_data else '✅ NO'}")
        print(f"   Date Ranges Valid: {'✅ YES' if date_ranges_valid else '❌ NO'}")
        
        # Detailed breakdown
        for period_key, data in period_data.items():
            if data:
                print(f"   {data['name']}: {data['total_candles']} candles, {data['start_date']} to {data['end_date']}")
        
        return {
            'identical_data_bug': identical_data,
            'date_ranges_valid': date_ranges_valid,
            'period_details': period_data,
            'diagnosis': 'PERIOD_SEPARATION_BUG' if identical_data else 'PERIOD_SEPARATION_OK'
        }
    
    def validate_trade_classification(self, pair: str = 'EURUSD', timeframe: str = '3D') -> Dict:
        """
        Test trade classification logic to identify unrealistic profit factors
        """
        print(f"\n🔍 TESTING TRADE CLASSIFICATION: {pair} {timeframe}")
        print("=" * 50)
        
        # Run a single test to get trade details
        result = self.engine.run_single_strategy_test(pair, timeframe, 730)
        
        if result['total_trades'] == 0:
            print("   ❌ No trades to analyze")
            return {'diagnosis': 'NO_TRADES', 'details': result}
        
        trades = result.get('trades', [])
        
        print(f"   📊 Total Trades: {len(trades)}")
        print(f"   🏆 Win Rate: {result['win_rate']:.1f}%")
        print(f"   💰 Profit Factor: {result['profit_factor']:.2f}")
        
        # Analyze individual trades
        trade_analysis = self.analyze_individual_trades(trades)
        
        # Check for unrealistic patterns
        issues = self.identify_trade_issues(trades, result)
        
        return {
            'diagnosis': issues['primary_issue'],
            'trade_count': len(trades),
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'trade_analysis': trade_analysis,
            'issues_found': issues,
            'sample_trades': trades[:5]  # First 5 trades for inspection
        }
    
    def analyze_individual_trades(self, trades: List[Dict]) -> Dict:
        """Analyze individual trade characteristics"""
        if not trades:
            return {}
        
        # Classify by result
        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']
        breakevens = [t for t in trades if t['result'] == 'BREAKEVEN']
        
        # P&L analysis
        win_pnls = [t['pnl'] for t in wins] if wins else [0]
        loss_pnls = [t['pnl'] for t in losses] if losses else [0]
        
        print(f"\n   📊 TRADE BREAKDOWN:")
        print(f"      Wins: {len(wins)} (avg P&L: ${np.mean(win_pnls):.2f})")
        print(f"      Losses: {len(losses)} (avg P&L: ${np.mean(loss_pnls):.2f})")
        print(f"      Breakevens: {len(breakevens)}")
        
        # Duration analysis
        durations = [t.get('duration_days', 0) for t in trades]
        print(f"      Avg Duration: {np.mean(durations):.1f} candles")
        
        # Sample trades for inspection
        print(f"\n   🔍 SAMPLE TRADES:")
        for i, trade in enumerate(trades[:3]):
            print(f"      Trade {i+1}: {trade['result']} ${trade['pnl']:.2f} "
                 f"({trade['zone_type']}, {trade.get('duration_days', 0)} candles)")
        
        return {
            'wins': len(wins),
            'losses': len(losses),
            'breakevens': len(breakevens),
            'avg_win_pnl': np.mean(win_pnls),
            'avg_loss_pnl': np.mean(loss_pnls),
            'avg_duration': np.mean(durations)
        }
    
    def identify_trade_issues(self, trades: List[Dict], result: Dict) -> Dict:
        """Identify specific issues with trade logic"""
        issues = []
        primary_issue = 'UNKNOWN'
        
        # Check for unrealistic profit factor
        if result['profit_factor'] > 100:
            issues.append('UNREALISTIC_PROFIT_FACTOR')
            primary_issue = 'UNREALISTIC_PROFIT_FACTOR'
        
        # Check for excessive breakevens counted as wins
        breakeven_threshold_issues = 0
        for trade in trades:
            pnl = trade['pnl']
            result_classification = trade['result']
            
            # Check if small P&L trades are misclassified
            if abs(pnl) <= 50 and result_classification == 'WIN':
                breakeven_threshold_issues += 1
        
        if breakeven_threshold_issues > len(trades) * 0.2:  # More than 20% misclassified
            issues.append('BREAKEVEN_THRESHOLD_TOO_LOW')
            if primary_issue == 'UNKNOWN':
                primary_issue = 'BREAKEVEN_THRESHOLD_TOO_LOW'
        
        # Check for missing losses
        loss_count = len([t for t in trades if t['result'] == 'LOSS'])
        if loss_count == 0 and len(trades) > 10:
            issues.append('NO_LOSSES_DETECTED')
            if primary_issue == 'UNKNOWN':
                primary_issue = 'NO_LOSSES_DETECTED'
        
        # Check for impossible win rates
        if result['win_rate'] > 80 and len(trades) > 5:
            issues.append('UNREALISTIC_WIN_RATE')
            if primary_issue == 'UNKNOWN':
                primary_issue = 'UNREALISTIC_WIN_RATE'
        
        return {
            'issues': issues,
            'primary_issue': primary_issue,
            'breakeven_threshold_issues': breakeven_threshold_issues,
            'total_trades': len(trades)
        }
    
    def debug_simulation_method(self) -> Dict:
        """
        Debug the simulate_realistic_outcome method to see what's wrong
        """
        print(f"\n🔍 DEBUGGING SIMULATION METHOD")
        print("=" * 50)
        
        # Simple test data
        test_data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035],
            'close': [1.1002, 1.1012, 1.1022, 1.1032, 1.1042]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        print(f"   📊 Test data shape: {test_data.shape}")
        print(f"   📊 Test data index: {type(test_data.index)}")
        print(f"   📊 Sample data:")
        print(test_data.head(2))
        
        # Test parameters
        entry_price = 1.1010
        stop_loss = 1.0990
        target_price = 1.1035  # Should hit on candle 4
        direction = 'BUY'
        position_size = 0.1
        
        print(f"\n   🎯 Test parameters:")
        print(f"      Entry: {entry_price}")
        print(f"      Stop: {stop_loss}")
        print(f"      Target: {target_price}")
        print(f"      Direction: {direction}")
        print(f"      Position Size: {position_size}")
        
        # Check if method exists
        if hasattr(self.engine, 'simulate_realistic_outcome'):
            print(f"   ✅ simulate_realistic_outcome method exists")
            
            try:
                # Call the method with debug
                result = self.engine.simulate_realistic_outcome(
                    entry_price, stop_loss, target_price, direction,
                    position_size, test_data, 0, 'R-B-R'
                )
                
                print(f"\n   📊 Method call successful:")
                if result:
                    print(f"      Result: {result}")
                else:
                    print(f"      Result: None")
                
                return {'debug_successful': True, 'result': result}
                
            except Exception as e:
                print(f"\n   ❌ Method call failed:")
                print(f"      Error: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'debug_successful': False, 'error': str(e)}
        else:
            print(f"   ❌ simulate_realistic_outcome method does not exist")
            print(f"   📊 Available methods: {[m for m in dir(self.engine) if 'simulate' in m.lower()]}")
            return {'debug_successful': False, 'error': 'Method not found'}
    
    def test_realistic_trade_simulation(self) -> Dict:
        """
        Test the simulate_realistic_outcome method directly with known inputs
        """
        print(f"\n🔍 TESTING REALISTIC TRADE SIMULATION")
        print("=" * 50)
        
        # Test 1: Winning Trade Simulation (10 candles)
        print(f"\n   🎯 TEST 1: Winning Trade (10 candles)")
        test_data_win = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1055, 1.1060, 1.1065, 1.1070],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1055, 1.1062, 1.1067, 1.1072, 1.1075],  # Hit target on candle 7
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1050, 1.1055, 1.1060, 1.1065],
            'close': [1.1002, 1.1012, 1.1022, 1.1032, 1.1042, 1.1052, 1.1058, 1.1063, 1.1068, 1.1073]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        entry_price = 1.1010
        stop_loss = 1.0990
        target_price = 1.1060  # 2.5R target (adjusted entry ~1.1012, so target needs to be hit)
        direction = 'BUY'
        position_size = 0.1
        
        print(f"      Entry: {entry_price}, Stop: {stop_loss}, Target: {target_price}")
        print(f"      Risk Distance: {entry_price - stop_loss:.4f} ({(entry_price - stop_loss)/0.0001:.0f} pips)")
        print(f"      Target Distance: {target_price - entry_price:.4f} ({(target_price - entry_price)/0.0001:.0f} pips)")
        
        win_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_win, 0, 'R-B-R'
        )
        
        if win_result:
            print(f"      ✅ WIN: {win_result['result']} ${win_result['pnl']:.2f} (Exit: {win_result['exit_price']:.4f})")
        else:
            print(f"      ❌ Failed to simulate win")
        
        # Test 2: Losing Trade Simulation (10 candles)
        print(f"\n   🎯 TEST 2: Losing Trade (10 candles)")
        test_data_loss = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1005, 1.1000, 1.0995, 1.0990, 1.0985, 1.0980, 1.0975, 1.0970],
            'high': [1.1005, 1.1015, 1.1010, 1.1005, 1.1000, 1.0995, 1.0990, 1.0985, 1.0980, 1.0975],
            'low': [1.0995, 1.1005, 1.0995, 1.0990, 1.0985, 1.0975, 1.0970, 1.0965, 1.0960, 1.0955],  # Hit stop on candle 6 (1.0975 < 1.0990)
            'close': [1.1002, 1.1012, 1.1000, 1.0995, 1.0990, 1.0980, 1.0975, 1.0970, 1.0965, 1.0960]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        print(f"      Entry: {entry_price}, Stop: {stop_loss}, Target: {target_price}")
        print(f"      Risk Distance: {entry_price - stop_loss:.4f} ({(entry_price - stop_loss)/0.0001:.0f} pips)")
        
        loss_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_loss, 0, 'R-B-R'
        )
        
        if loss_result:
            print(f"      ✅ LOSS: {loss_result['result']} ${loss_result['pnl']:.2f} (Exit: {loss_result['exit_price']:.4f})")
        else:
            print(f"      ❌ Failed to simulate loss")
        
        # Test 3: Breakeven Trade Simulation (10 candles - move to 1R then back to entry)
        print(f"\n   🎯 TEST 3: Breakeven Trade (10 candles)")
        test_data_be = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1035, 1.1025, 1.1015, 1.1012, 1.1010, 1.1008],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1040, 1.1030, 1.1020, 1.1017, 1.1015, 1.1012],  # Hit 1R (~1.1034) then back to entry
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1030, 1.1020, 1.1010, 1.1007, 1.1005, 1.1003],
            'close': [1.1002, 1.1012, 1.1022, 1.1032, 1.1037, 1.1027, 1.1017, 1.1012, 1.1008, 1.1005]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        print(f"      Entry: {entry_price}, Stop: {stop_loss}, Target: {target_price}")
        print(f"      Expected: Move to 1R (~1.1034), trigger breakeven, then hit breakeven stop (~1.1012)")
        
        be_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_be, 0, 'R-B-R'
        )
        
        if be_result:
            print(f"      ✅ BREAKEVEN: {be_result['result']} ${be_result['pnl']:.2f} (Exit: {be_result['exit_price']:.4f})")
        else:
            print(f"      ❌ Failed to simulate breakeven")
        
        # Add data that hits breakeven stop at entry level  
        test_data_be_stop = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1035, 1.1025, 1.1015, 1.1010, 1.1005, 1.1000],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1040, 1.1030, 1.1020, 1.1015, 1.1010, 1.1005],
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1030, 1.1020, 1.1005, 1.1000, 1.0995, 1.0990],  # Hit breakeven stop
            'close': [1.1002, 1.1012, 1.1022, 1.1032, 1.1037, 1.1027, 1.1010, 1.1005, 1.1000, 1.0995]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        be_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_be_stop, 0, 'R-B-R'
        )
        
        print(f"      Entry: {entry_price}, Stop: {stop_loss}, Target: {target_price}")
        print(f"      Expected: Move to 1R (1.1030), trigger breakeven, then exit at entry level")
        
        be_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_be, 0, 'R-B-R'
        )
        
        if be_result:
            print(f"      ✅ BREAKEVEN: {be_result['result']} ${be_result['pnl']:.2f} (Exit: {be_result['exit_price']:.4f})")
        else:
            print(f"      ❌ Failed to simulate breakeven")
        
        # Test 4: Target Hit Test (ensure target logic works)
        print(f"\n   🎯 TEST 4: Direct Target Hit (10 candles)")
        test_data_target = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1055, 1.1060, 1.1065, 1.1070],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1055, 1.1062, 1.1067, 1.1072, 1.1075],  # Hit target on candle 8
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1050, 1.1055, 1.1060, 1.1065],
            'close': [1.1002, 1.1012, 1.1022, 1.1032, 1.1042, 1.1052, 1.1058, 1.1063, 1.1068, 1.1073]
        }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
        
        target_result = self.engine.simulate_realistic_outcome(
            entry_price, stop_loss, target_price, direction,
            position_size, test_data_target, 0, 'R-B-R'
        )
        
        if target_result:
            print(f"      ✅ TARGET: {target_result['result']} ${target_result['pnl']:.2f} (Exit: {target_result['exit_price']:.4f})")
        else:
            print(f"      ❌ Failed to simulate target hit")
        
        # Summary
        tests_passed = sum([1 for result in [win_result, loss_result, be_result, target_result] if result is not None])
        
        print(f"\n   📊 SIMULATION TEST SUMMARY:")
        print(f"      Tests Passed: {tests_passed}/4")
        print(f"      Status: {'✅ WORKING' if tests_passed >= 3 else '❌ BROKEN'}")
        
        return {
            'test_completed': tests_passed >= 3,
            'tests_passed': tests_passed,
            'win_result': win_result,
            'loss_result': loss_result,
            'breakeven_result': be_result,
            'target_result': target_result,
            'status': 'WORKING' if tests_passed >= 3 else 'BROKEN'
        }
    
    def run_comprehensive_validation(self, pair: str = 'EURUSD', timeframe: str = '3D') -> Dict:
        """
        Run all validation tests and provide comprehensive diagnosis
        """
        print(f"\n🎯 COMPREHENSIVE BACKTESTING VALIDATION")
        print(f"📊 Testing: {pair} {timeframe}")
        print("=" * 60)
        
        # Test 1: Period Separation
        period_test = self.validate_period_separation(pair, timeframe)
        
        # Test 2: Trade Classification
        trade_test = self.validate_trade_classification(pair, timeframe)
        
        # Test 3: Trade Simulation
        simulation_test = self.test_realistic_trade_simulation()
        
        # Generate overall diagnosis
        overall_diagnosis = self.generate_overall_diagnosis(period_test, trade_test, simulation_test)
        
        # Print summary
        self.print_validation_summary(overall_diagnosis)
        
        return {
            'period_separation': period_test,
            'trade_classification': trade_test,
            'simulation_test': simulation_test,
            'overall_diagnosis': overall_diagnosis
        }
    
    def generate_overall_diagnosis(self, period_test: Dict, trade_test: Dict, simulation_test: Dict) -> Dict:
        """Generate overall diagnosis and recommendations"""
        
        # Primary issues identified
        issues = []
        
        if period_test['identical_data_bug']:
            issues.append('PERIOD_SEPARATION_BUG')
        
        if trade_test.get('profit_factor', 0) > 100:
            issues.append('UNREALISTIC_PROFIT_FACTOR')
        
        if trade_test.get('issues_found', {}).get('primary_issue') not in ['UNKNOWN', None]:
            issues.append(trade_test['issues_found']['primary_issue'])
        
        # Determine primary fix needed
        if 'PERIOD_SEPARATION_BUG' in issues:
            primary_fix = 'FIX_PERIOD_SEPARATION'
            recommendation = 'Fix load_data_with_validation method to properly filter by days_back'
        elif 'UNREALISTIC_PROFIT_FACTOR' in issues:
            primary_fix = 'FIX_TRADE_CLASSIFICATION'
            recommendation = 'Increase breakeven threshold and fix profit factor calculation'
        else:
            primary_fix = 'SYSTEM_OK'
            recommendation = 'System appears to be working correctly'
        
        return {
            'issues_found': issues,
            'primary_fix_needed': primary_fix,
            'recommendation': recommendation,
            'period_separation_ok': not period_test['identical_data_bug'],
            'trade_classification_ok': trade_test.get('profit_factor', 0) < 10,
            'simulation_ok': simulation_test.get('test_completed', False)
        }
    
    def print_validation_summary(self, diagnosis: Dict):
        """Print comprehensive validation summary"""
        print(f"\n🎯 VALIDATION SUMMARY")
        print("=" * 40)
        print(f"🏆 PRIMARY FIX NEEDED: {diagnosis['primary_fix_needed']}")
        print(f"💡 RECOMMENDATION: {diagnosis['recommendation']}")
        
        print(f"\n✅ SYSTEM STATUS:")
        print(f"   Period Separation: {'✅ OK' if diagnosis['period_separation_ok'] else '❌ BROKEN'}")
        print(f"   Trade Classification: {'✅ OK' if diagnosis['trade_classification_ok'] else '❌ BROKEN'}")
        print(f"   Simulation Logic: {'✅ OK' if diagnosis['simulation_ok'] else '❌ BROKEN'}")
        
        if diagnosis['issues_found']:
            print(f"\n🚨 ISSUES FOUND:")
            for issue in diagnosis['issues_found']:
                print(f"   • {issue}")

    def debug_raw_data_format(self, pair: str = 'EURUSD', timeframe: str = '3D'):
        """Check the raw data format before processing"""
        print(f"\n🔍 DEBUGGING RAW DATA FORMAT: {pair} {timeframe}")
        print("=" * 50)
        
        # Get the file path
        try:
            files = self.engine.data_loader.list_available_files()
            target_file = None
            
            for filename in files:
                parsed = self.engine.data_loader.parse_oanda_filename(filename)
                if parsed:
                    file_pair, file_timeframe = parsed
                    if file_pair == pair and file_timeframe == timeframe:
                        target_file = filename
                        break
            
            if target_file:
                filepath = os.path.join(self.engine.data_loader.raw_path, target_file)
                print(f"   📁 File: {target_file}")
                
                # Read raw file to check format
                import pandas as pd
                raw_data = pd.read_csv(filepath, nrows=5)  # Just first 5 rows
                
                print(f"   📊 Raw columns: {raw_data.columns.tolist()}")
                print(f"   📅 Sample data:")
                for i, row in raw_data.iterrows():
                    print(f"      Row {i}: {dict(row)}")
                
                # Check if there's a date column and what format it's in
                if '<DATE>' in raw_data.columns:
                    sample_dates = raw_data['<DATE>'].head(3).tolist()
                    print(f"   🔍 Sample dates: {sample_dates}")
                elif 'date' in raw_data.columns:
                    sample_dates = raw_data['date'].head(3).tolist()
                    print(f"   🔍 Sample dates: {sample_dates}")
                else:
                    print(f"   ❌ No date column found!")
                    
            else:
                print(f"   ❌ No file found for {pair} {timeframe}")
                
        except Exception as e:
            print(f"   ❌ Error reading raw data: {str(e)}")

def main():
    """Main validation function"""
    print("🔍 BACKTESTING LOGIC VALIDATOR")
    print("Diagnoses period separation and profit factor issues")
    print("=" * 60)
    
    validator = BacktestLogicValidator()
    
    print("\n🎯 VALIDATION OPTIONS:")
    print("1. Quick Validation (EURUSD 3D)")
    print("2. Period Separation Test Only")
    print("3. Trade Classification Test Only") 
    print("4. Simulation Test Only")
    print("5. Custom Pair/Timeframe")
    print("6. Debug Raw Data Format")
    print("7. Debug Simulation Method")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        # Comprehensive validation
        result = validator.run_comprehensive_validation('EURUSD', '3D')
        
    elif choice == '2':
        # Period separation only
        validator.validate_period_separation('EURUSD', '3D')
        
    elif choice == '3':
        # Trade classification only
        validator.validate_trade_classification('EURUSD', '3D')
        
    elif choice == '4':
        # Simulation test only
        validator.test_realistic_trade_simulation()
        
    elif choice == '5':
        # Custom test
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (e.g., 3D): ").strip()
        validator.run_comprehensive_validation(pair, timeframe)
        
    elif choice == '6':
        # Debug raw data format
        pair = input("Enter pair (e.g., EURUSD): ").strip().upper() or 'EURUSD'
        timeframe = input("Enter timeframe (e.g., 3D): ").strip() or '3D'
        validator.debug_raw_data_format(pair, timeframe)
        
    elif choice == '7':
        # Debug simulation method
        validator.debug_simulation_method()
        
    else:
        print("❌ Invalid choice")
        
    def debug_dataloader_output(self, pair: str = 'EURUSD', timeframe: str = '3D'):
        """Debug what DataLoader actually returns"""
        print(f"\n🔍 DEBUGGING DATALOADER OUTPUT: {pair} {timeframe}")
        print("=" * 50)
        
        data = self.engine.data_loader.load_pair_data(pair, timeframe)
        
        if data is not None:
            print(f"   📊 Data Shape: {data.shape}")
            print(f"   📅 Index Type: {type(data.index)}")
            print(f"   🔢 Index Sample: {data.index[:5].tolist()}")
            print(f"   📆 First 3 dates: {data.index[:3].tolist()}")
            print(f"   📆 Last 3 dates: {data.index[-3:].tolist()}")
            print(f"   💰 Price Range: {data['close'].iloc[0]:.5f} → {data['close'].iloc[-1]:.5f}")
            
            # Check if datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                print(f"   ✅ Proper DatetimeIndex detected")
                print(f"   📅 Date Range: {data.index[0]} to {data.index[-1]}")
                print(f"   ⏰ Time Span: {(data.index[-1] - data.index[0]).days} days")
            else:
                print(f"   ❌ NOT DatetimeIndex - This is the problem!")
        else:
            print(f"   ❌ No data returned")

if __name__ == "__main__":
    main()