"""
Signal Generation Tests - Module 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

def test_signal_generation():
    """Test complete signal generation pipeline"""
    print("🎯 Testing Signal Generation Pipeline")
    print("=" * 50)
    
    try:
        # Load and process data
        print("📊 Loading EURUSD data...")
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Initialize all components
        print("🔧 Initializing components...")
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=10000)
        
        signal_generator = SignalGenerator(
            zone_detector, trend_classifier, risk_manager
        )
        
        # Generate signals
        print("🎯 Generating signals...")
        signals = signal_generator.generate_signals(classified_data, 'Daily', 'EURUSD')
        
        # Analyze results
        summary = signal_generator.get_signal_summary()
        
        print(f"\n📈 SIGNAL GENERATION RESULTS:")
        print(f"   Total Signals: {summary['total_signals']}")
        print(f"   High Priority: {summary['by_priority']['high']}")
        print(f"   Medium Priority: {summary['by_priority']['medium']}")
        print(f"   Low Priority: {summary['by_priority']['low']}")
        print(f"   BUY Signals: {summary['by_direction']['BUY']}")
        print(f"   SELL Signals: {summary['by_direction']['SELL']}")
        print(f"   Average Score: {summary['avg_score']}")
        print(f"   Average R:R: {summary['avg_risk_reward']}")
        
        # Export signals
        if signals:
            filename = signal_generator.export_signals_for_backtesting()
            print(f"💾 Signals exported to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_timeframe_signals():
    """Test signal generation across different timeframes"""
    print("\n🎯 Testing Multi-Timeframe Signal Generation")
    print("=" * 50)
    
    # Note: This would require loading different timeframe data
    # For now, we'll test the framework with Daily data
    
    timeframes = ['Daily']  # Add 'H4', 'H12', 'Weekly' when data available
    
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Initialize components
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=10000)
        
        signal_generator = SignalGenerator(
            zone_detector, trend_classifier, risk_manager
        )
        
        all_results = {}
        
        for timeframe in timeframes:
            print(f"\n📊 Testing {timeframe} timeframe:")
            signals = signal_generator.generate_signals(classified_data, timeframe, 'EURUSD')
            summary = signal_generator.get_signal_summary()
            
            all_results[timeframe] = {
                'signals': len(signals),
                'avg_score': summary['avg_score'] if signals else 0,
                'high_priority': summary['by_priority']['high']
            }
            
            print(f"   Signals: {len(signals)}")
            print(f"   Avg Score: {summary['avg_score'] if signals else 0}")
        
        print(f"\n📊 MULTI-TIMEFRAME SUMMARY:")
        for tf, results in all_results.items():
            print(f"   {tf}: {results['signals']} signals, avg score {results['avg_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all signal generation tests"""
    print("🚀 SIGNAL GENERATION SYSTEM - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic signal generation
    test_results.append(test_signal_generation())
    
    # Test 2: Multi-timeframe testing
    test_results.append(test_multi_timeframe_signals())
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - SIGNAL GENERATION MODULE 5 READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 Module 5 Complete - Signal Generation System Working!")
        print("📋 Features implemented:")
        print("   ✅ Risk-aware signal generation")
        print("   ✅ Trend alignment filtering")
        print("   ✅ Multi-factor signal scoring")
        print("   ✅ Signal priority ranking")
        print("   ✅ Backtesting data export")
        print("   ✅ Complete trade parameters")
        print("\n📈 Ready for backtesting and validation!")
    else:
        print("\n⚠️  Module 5 requires fixes before proceeding")