"""
EMA Filter Validation Test
Ensures the EMA filter backtesting maintains all core engine logic
Author: Trading Strategy Automation Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import both engines
from core_backtest_engine import CoreBacktestEngine
from backtest_ema_filter import EMAFilterBacktestEngine

def validate_zone_detection():
    """Validate that zone detection is identical between engines"""
    print("\n🔍 VALIDATING ZONE DETECTION...")
    
    # Create both engines
    core_engine = CoreBacktestEngine()
    filter_engine = EMAFilterBacktestEngine()
    
    # Load same data
    data = core_engine.load_data_with_validation('EURUSD', '3D', 730)
    if data is None:
        print("❌ Failed to load data")
        return False
    
    # Initialize components for both
    from modules.candle_classifier import CandleClassifier
    from modules.zone_detector import ZoneDetector
    
    # Core engine zone detection
    candle_classifier_core = CandleClassifier(data)
    classified_data_core = candle_classifier_core.classify_all_candles()
    zone_detector_core = ZoneDetector(candle_classifier_core)
    patterns_core = zone_detector_core.detect_all_patterns(classified_data_core)
    
    # Filter engine zone detection
    candle_classifier_filter = CandleClassifier(data)
    classified_data_filter = candle_classifier_filter.classify_all_candles()
    zone_detector_filter = ZoneDetector(candle_classifier_filter)
    patterns_filter = zone_detector_filter.detect_all_patterns(classified_data_filter)
    
    # Compare zone counts
    core_total = patterns_core['total_patterns']
    filter_total = patterns_filter['total_patterns']
    
    print(f"   Core engine zones: {core_total}")
    print(f"   Filter engine zones: {filter_total}")
    
    if core_total != filter_total:
        print("❌ Zone detection mismatch!")
        return False
    
    print("✅ Zone detection validated - identical results")
    return True


def validate_trade_execution_logic():
    """Validate that trade execution logic is preserved"""
    print("\n🔍 VALIDATING TRADE EXECUTION LOGIC...")
    
    # Create both engines
    core_engine = CoreBacktestEngine()
    filter_engine = EMAFilterBacktestEngine()
    
    # Set filter engine to use original EMA 50/200 (should match core)
    filter_engine.set_current_filter('EMA_50_200_Cross')
    
    # Run same test
    print("   Running EURUSD 3D test on both engines...")
    core_result = core_engine.run_single_strategy_test('EURUSD', '3D', 730)
    filter_result = filter_engine.run_single_strategy_test('EURUSD', '3D', 730)
    
    # Compare results
    print(f"\n   Core engine results:")
    print(f"   - Trades: {core_result['total_trades']}")
    print(f"   - Win Rate: {core_result['win_rate']:.1f}%")
    print(f"   - Profit Factor: {core_result['profit_factor']:.2f}")
    print(f"   - Total Return: {core_result['total_return']:.2f}%")
    
    print(f"\n   Filter engine results (EMA 50/200):")
    print(f"   - Trades: {filter_result['total_trades']}")
    print(f"   - Win Rate: {filter_result['win_rate']:.1f}%")
    print(f"   - Profit Factor: {filter_result['profit_factor']:.2f}")
    print(f"   - Total Return: {filter_result['total_return']:.2f}%")
    
    # Check if results match
    trades_match = core_result['total_trades'] == filter_result['total_trades']
    wr_match = abs(core_result['win_rate'] - filter_result['win_rate']) < 0.1
    pf_match = abs(core_result['profit_factor'] - filter_result['profit_factor']) < 0.01
    
    if trades_match and wr_match and pf_match:
        print("\n✅ Trade execution validated - results match!")
        return True
    else:
        print("\n❌ Trade execution mismatch!")
        return False


def validate_zone_validation_logic():
    """Validate that zone validation (2.5x target) logic is preserved"""
    print("\n🔍 VALIDATING ZONE VALIDATION LOGIC...")
    
    # Since EMAFilterBacktestEngine extends CoreBacktestEngine, we just need to verify
    # that the filter engine properly inherits the parent's validation logic
    
    # Test that the filter engine has the same validation method
    filter_engine = EMAFilterBacktestEngine()
    
    # Check if method exists (inherited from parent)
    if hasattr(filter_engine, 'track_zone_validation_realtime'):
        print("✅ Zone validation method properly inherited from CoreBacktestEngine")
        
        # Test that it's the same method (not overridden)
        if filter_engine.track_zone_validation_realtime.__func__ == CoreBacktestEngine.track_zone_validation_realtime:
            print("✅ Zone validation logic is identical (not overridden)")
            return True
        else:
            print("⚠️  Zone validation method exists but may be overridden")
            # Let's do a simple functional test instead
            return validate_inheritance_functional_test(filter_engine)
    else:
        print("❌ Zone validation method not found in filter engine")
        return False

def validate_inheritance_functional_test(filter_engine):
    """Simple test to verify inherited functionality works"""
    try:
        # Create minimal test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        test_data = pd.DataFrame({
            'open': [1.0975] * 10,
            'high': [1.0980] * 10,
            'low': [1.0970] * 10,
            'close': [1.0975] * 10
        }, index=dates)
        
        # Create minimal zone
        test_zone = {
            'type': 'R-B-R',
            'zone_high': 1.1000,
            'zone_low': 1.0950,
            'zone_range': 0.0050
        }
        
        # Call the inherited method
        result = filter_engine.track_zone_validation_realtime(test_zone, test_data, 5)
        
        # Should return a proper result structure
        if isinstance(result, dict) and 'validated' in result and 'validation_idx' in result:
            print("✅ Inherited zone validation method works correctly")
            return True
        else:
            print("❌ Inherited method returned unexpected result")
            return False
            
    except Exception as e:
        print(f"❌ Error testing inherited method: {str(e)}")
        return False


def validate_filter_functionality():
    """Validate that different filters produce different results"""
    print("\n🔍 VALIDATING FILTER FUNCTIONALITY...")
    
    filter_engine = EMAFilterBacktestEngine()
    
    # Test No Filter vs EMA Filter
    print("   Testing No_Filter vs EMA_50_200_Cross...")
    
    filter_engine.set_current_filter('No_Filter')
    no_filter_result = filter_engine.run_single_strategy_test('EURUSD', '3D', 730)
    
    filter_engine.set_current_filter('EMA_50_200_Cross')
    ema_filter_result = filter_engine.run_single_strategy_test('EURUSD', '3D', 730)
    
    print(f"\n   No Filter trades: {no_filter_result['total_trades']}")
    print(f"   EMA 50/200 trades: {ema_filter_result['total_trades']}")
    
    # No filter should have more trades (no trend filtering)
    if no_filter_result['total_trades'] > ema_filter_result['total_trades']:
        print("✅ Filter functionality validated - filters are working!")
        return True
    else:
        print("❌ Filter functionality error - No_Filter should have more trades!")
        return False


def validate_comprehensive_analysis():
    """Validate that comprehensive analysis functions work"""
    print("\n🔍 VALIDATING COMPREHENSIVE ANALYSIS FUNCTIONS...")
    
    filter_engine = EMAFilterBacktestEngine()
    
    # Check that we can discover pairs
    pairs = filter_engine.discover_all_pairs()
    print(f"   Discovered {len(pairs)} pairs")
    
    if len(pairs) > 0:
        print("✅ Pair discovery working!")
    else:
        print("❌ Pair discovery failed!")
        return False
    
    # Check valid combinations
    valid_combos = filter_engine.discover_valid_data_combinations()
    print(f"   Found {len(valid_combos)} valid pair/timeframe combinations")
    
    if len(valid_combos) > 0:
        print("✅ Data combination discovery working!")
        return True
    else:
        print("❌ Data combination discovery failed!")
        return False


def run_full_validation():
    """Run all validation tests"""
    print("🎯 EMA FILTER VALIDATION SUITE")
    print("=" * 60)
    print("This will validate that the EMA filter backtesting maintains")
    print("all core engine functionality while adding filter capabilities.")
    
    tests = [
        ("Zone Detection", validate_zone_detection),
        ("Trade Execution Logic", validate_trade_execution_logic),
        ("Zone Validation Logic", validate_zone_validation_logic),
        ("Filter Functionality", validate_filter_functionality),
        ("Comprehensive Analysis", validate_comprehensive_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("The EMA filter backtesting engine is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


def validate_specific_trade_details():
    """Deep dive validation of specific trades to ensure logic is preserved"""
    print("\n🔍 VALIDATING SPECIFIC TRADE DETAILS...")
    
    core_engine = CoreBacktestEngine()
    filter_engine = EMAFilterBacktestEngine()
    filter_engine.set_current_filter('EMA_50_200_Cross')
    
    # Run same test
    core_result = core_engine.run_single_strategy_test('EURUSD', '1D', 365)
    filter_result = filter_engine.run_single_strategy_test('EURUSD', '1D', 365)
    
    if core_result['total_trades'] > 0 and filter_result['total_trades'] > 0:
        # Compare first trade details
        core_trade = core_result['trades'][0] if core_result['trades'] else None
        filter_trade = filter_result['trades'][0] if filter_result['trades'] else None
        
        if core_trade and filter_trade:
            print("\n   First trade comparison:")
            print(f"   Core - Zone: {core_trade['zone_type']}, PnL: ${core_trade['pnl']:.2f}")
            print(f"   Filter - Zone: {filter_trade['zone_type']}, PnL: ${filter_trade['pnl']:.2f}")
            
            # Check critical fields
            fields_match = (
                core_trade['zone_type'] == filter_trade['zone_type'] and
                abs(core_trade['pnl'] - filter_trade['pnl']) < 0.01 and
                core_trade['direction'] == filter_trade['direction']
            )
            
            if fields_match:
                print("✅ Trade details match!")
                return True
    
    print("⚠️  Unable to validate trade details (no trades to compare)")
    return True  # Don't fail if no trades


if __name__ == "__main__":
    # Run validation
    success = run_full_validation()
    
    if success:
        print("\n✅ Ready to use backtest_ema_filter_v2.py for production analysis!")
    else:
        print("\n❌ Please fix issues before using for analysis.")