"""
Backtesting Engine Tests - Module 6
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.backtester import TradingBacktester
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

def test_backtester_initialization():
   """Test backtester initialization and configuration"""
   print("🧪 Testing Backtester Initialization")
   print("=" * 40)
   
   # Create mock signal generator for testing
   data_loader = DataLoader()
   data = data_loader.load_pair_data('EURUSD', 'Daily')
   
   candle_classifier = CandleClassifier(data)
   zone_detector = ZoneDetector(candle_classifier)
   trend_classifier = TrendClassifier(data)
   risk_manager = RiskManager(account_balance=10000)
   signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
   
   # Test initialization
   backtester = TradingBacktester(signal_generator, initial_balance=10000)
   
   assert backtester.initial_balance == 10000, "Initial balance not set correctly"
   assert backtester.current_balance == 10000, "Current balance not initialized"
   assert len(backtester.open_trades) == 0, "Open trades should be empty"
   assert len(backtester.closed_trades) == 0, "Closed trades should be empty"
   
   print("✅ Backtester initialization test passed")
   return True

def test_trade_execution():
   """Test basic trade execution functionality"""
   print("\n🧪 Testing Trade Execution")
   print("=" * 40)
   
   # Setup
   data_loader = DataLoader()
   data = data_loader.load_pair_data('EURUSD', 'Daily')
   
   candle_classifier = CandleClassifier(data)
   zone_detector = ZoneDetector(candle_classifier)
   trend_classifier = TrendClassifier(data)
   risk_manager = RiskManager(account_balance=10000)
   signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
   
   backtester = TradingBacktester(signal_generator, initial_balance=10000)
   
   # Create test signal
   test_signal = {
       'signal_id': 'TEST_001',
       'pair': 'EURUSD',
       'direction': 'BUY',
       'entry_price': 1.1000,
       'stop_loss': 1.0950,
       'take_profit_1': 1.1050,
       'take_profit_2': 1.1100,
       'position_size': 0.1,
       'risk_amount': 500,
       'zone_high': 1.1020,
       'zone_low': 1.0980,
       'signal_score': 75
   }
   
   # Test trade execution
   current_data = data.iloc[-1]
   current_date = data.index[-1]
   
   # Execute trade
   success = backtester.execute_trade(test_signal, current_data, current_date)
   
   assert success, "Trade execution should succeed"
   assert len(backtester.open_trades) == 1, "Should have one open trade"
   assert backtester.total_trades == 1, "Total trades counter should increment"
   
   # Check trade details
   trade = backtester.open_trades[0]
   assert trade['direction'] == 'BUY', "Trade direction should match signal"
   assert trade['status'] == 'open', "Trade status should be open"
   assert trade['break_even_moved'] == False, "Break-even should not be moved initially"
   
   print("✅ Trade execution test passed")
   return True

def test_stop_loss_management():
   """Test stop loss hit detection and management"""
   print("\n🧪 Testing Stop Loss Management")
   print("=" * 40)
   
   # Setup backtester with test trade
   data_loader = DataLoader()
   data = data_loader.load_pair_data('EURUSD', 'Daily')
   
   candle_classifier = CandleClassifier(data)
   zone_detector = ZoneDetector(candle_classifier)
   trend_classifier = TrendClassifier(data)
   risk_manager = RiskManager(account_balance=10000)
   signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
   
   backtester = TradingBacktester(signal_generator, initial_balance=10000)
   
   # Create test trade in open trades
   test_trade = {
       'trade_id': 'T001',
       'direction': 'BUY',
       'entry_price': 1.1000,
       'stop_loss': 1.0950,
       'take_profit_1': 1.1050,
       'take_profit_2': 1.1100,
       'position_size': 0.1,
       'break_even_moved': False,
       'status': 'open'
   }
   
   backtester.open_trades.append(test_trade)
   
   # Test stop loss hit scenario
   stop_loss_data = pd.Series({
       'open': 1.0960,
       'high': 1.0970,
       'low': 1.0940,  # Hits stop loss
       'close': 1.0945
   }, name=data.index[-1])
   
   # Check stop loss detection
   stop_hit = backtester.check_stop_loss_hit(test_trade, stop_loss_data)
   assert stop_hit, "Stop loss should be detected when low hits stop"
   
   print("✅ Stop loss management test passed")
   return True

def test_break_even_management():
   """Test break-even stop management at 1:1 risk/reward"""
   print("\n🧪 Testing Break-Even Management")
   print("=" * 40)
   
   # Setup
   data_loader = DataLoader()
   data = data_loader.load_pair_data('EURUSD', 'Daily')
   
   candle_classifier = CandleClassifier(data)
   zone_detector = ZoneDetector(candle_classifier)
   trend_classifier = TrendClassifier(data)
   risk_manager = RiskManager(account_balance=10000)
   signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
   
   backtester = TradingBacktester(signal_generator, initial_balance=10000)
   
   # Create test trade
   test_trade = {
       'trade_id': 'T001',
       'direction': 'BUY',
       'entry_price': 1.1000,
       'stop_loss': 1.0950,  # 50 pip risk
       'take_profit_1': 1.1050,  # 1:1 level
       'take_profit_2': 1.1100,  # 1:2 level
       'position_size': 0.1,
       'break_even_moved': False,
       'status': 'open'
   }
   
   backtester.open_trades.append(test_trade)
   
   # Test break-even trigger (price hits 1:1)
   breakeven_data = pd.Series({
       'open': 1.1040,
       'high': 1.1055,  # Hits 1:1 level
       'low': 1.1035,
       'close': 1.1050
   }, name=data.index[-1])
   
   # Check break-even trigger
   should_move = backtester.should_move_to_breakeven(test_trade, breakeven_data)
   assert should_move, "Break-even should trigger when price hits 1:1"
   
   # Move to break-even
   backtester.move_to_breakeven(test_trade)
   
   assert test_trade['break_even_moved'] == True, "Break-even flag should be set"
   assert test_trade['stop_loss'] == 1.1000, "Stop should be moved to entry price"
   
   print("✅ Break-even management test passed")
   return True

def test_small_backtest():
   """Test small backtest with limited data"""
   print("\n🧪 Testing Small Backtest")
   print("=" * 40)
   
   try:
       # Load data
       data_loader = DataLoader()
       data = data_loader.load_pair_data('EURUSD', 'Daily')
       
       # Setup components
       candle_classifier = CandleClassifier(data)
       classified_data = candle_classifier.classify_all_candles()
       
       zone_detector = ZoneDetector(candle_classifier)
       trend_classifier = TrendClassifier(data)
       risk_manager = RiskManager(account_balance=10000)
       signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
       
       # Run small backtest (3 months)
       backtester = TradingBacktester(signal_generator, initial_balance=10000)
       
       # Use recent 3 months for testing
       end_date = data.index[-1]
       start_date = end_date - pd.Timedelta(days=90)
       
       # Ensure start date exists in data
       start_date_str = start_date.strftime('%Y-%m-%d')
       end_date_str = end_date.strftime('%Y-%m-%d')
       
       # Find closest available dates
       available_start = data.index[data.index >= start_date][0]
       start_date_str = available_start.strftime('%Y-%m-%d')
       
       print(f"   Testing period: {start_date_str} to {end_date_str}")
       
       results = backtester.run_walk_forward_backtest(
           classified_data,
           start_date_str,
           end_date_str,
           lookback_days=365,
           pair='EURUSD'
       )
       
       # Validate results structure
       required_keys = [
           'total_trades', 'win_rate', 'profit_factor', 
           'final_balance', 'equity_curve', 'closed_trades'
       ]
       
       for key in required_keys:
           assert key in results, f"Missing required result key: {key}"
       
       print(f"   Results: {results['total_trades']} trades, {results['win_rate']}% win rate")
       print(f"   Final balance: ${results['final_balance']:,.2f}")
       print(f"   Profit factor: {results['profit_factor']}")
       
       print("✅ Small backtest test passed")
       return True
       
   except Exception as e:
       print(f"❌ Small backtest failed: {e}")
       import traceback
       traceback.print_exc()
       return False

def run_all_tests():
   """Run all backtesting tests"""
   print("🚀 BACKTESTING ENGINE - COMPREHENSIVE TESTING")
   print("=" * 60)
   
   test_results = []
   
   # Test 1: Initialization
   test_results.append(test_backtester_initialization())
   
   # Test 2: Trade execution
   test_results.append(test_trade_execution())
   
   # Test 3: Stop loss management
   test_results.append(test_stop_loss_management())
   
   # Test 4: Break-even management
   test_results.append(test_break_even_management())
   
   # Test 5: Small backtest
   test_results.append(test_small_backtest())
   
   # Final results
   passed_tests = sum(test_results)
   total_tests = len(test_results)
   
   print(f"\n🎯 FINAL RESULTS:")
   print(f"   Tests passed: {passed_tests}/{total_tests}")
   print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
   
   if passed_tests == total_tests:
       print("✅ ALL TESTS PASSED - BACKTESTING ENGINE MODULE 6 READY!")
       return True
   else:
       print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
       return False

if __name__ == "__main__":
   success = run_all_tests()
   
   if success:
       print("\n🎉 Module 6 Core Components Complete!")
       print("📋 Next steps:")
       print("   1. Run complete 8-year backtest")
       print("   2. Generate performance report")
       print("   3. Create equity curve visualization")
       print("   4. Export trade logs for analysis")
       print("   5. Validate against manual strategy baseline")
   else:
       print("\n⚠️  Module 6 requires fixes before proceeding")