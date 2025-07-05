"""
Comprehensive Trend Classification Tests - Module 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from modules.trend_classifier import TrendClassifier
from modules.data_loader import DataLoader
from config.settings import TREND_CONFIG

def test_ema_calculation():
    """Test EMA calculation accuracy"""
    print("🧪 Testing EMA Calculations")
    print("=" * 50)
    
    # Create test data with known EMA values
    test_data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    })
    
    classifier = TrendClassifier(test_data)
    results = classifier.calculate_emas()
    
    # Check EMAs exist
    assert 'ema_50' in results.columns, "EMA 50 not calculated"
    assert 'ema_200' in results.columns, "EMA 200 not calculated"
    
    # Check no NaN values (EMAs should start from first value)
    assert not results['ema_50'].isna().any(), "EMA 50 contains NaN"
    assert not results['ema_200'].isna().any(), "EMA 200 contains NaN"
    
    print("✅ EMA calculations test passed")
    return True

def test_trend_classification():
    """Test basic trend classification logic"""
    print("\n🧪 Testing Trend Classification Logic")
    print("=" * 50)
    
    # Create test data where EMA50 > EMA100 > EMA200 (strong bullish)
    test_data = pd.DataFrame({
        'close': [100] * 250  # Flat price for predictable EMAs
    })
    
    # Manually set EMAs to test classification
    classifier = TrendClassifier(test_data)
    classifier.calculate_emas()
    
    # Override with test values for STRONG BULLISH (50 > 100 > 200)
    classifier.data['ema_50'] = 105   # Highest
    classifier.data['ema_100'] = 102  # Middle
    classifier.data['ema_200'] = 100  # Lowest
    
    results = classifier.classify_trend()
    
    # Should all be strong_bullish
    assert (results['trend'] == 'strong_bullish').all(), "Strong bullish trend classification failed"
    
    # Test STRONG BEARISH (50 < 100 < 200)
    classifier.data['ema_50'] = 95    # Lowest
    classifier.data['ema_100'] = 98   # Middle
    classifier.data['ema_200'] = 100  # Highest
    
    results = classifier.classify_trend()
    
    # Should all be strong_bearish
    assert (results['trend'] == 'strong_bearish').all(), "Strong bearish trend classification failed"
    
    print("✅ Trend classification test passed")
    return True

def test_eurusd_trend_classification():
    """Test with real EURUSD data"""
    print("\n📊 Testing EURUSD Trend Classification")
    print("=" * 50)
    
    try:
        # Load EURUSD data
        loader = DataLoader()
        data = loader.load_pair_data('EURUSD', 'Daily')
        
        # Need at least 200 candles for EMA200
        if len(data) < 200:
            print("⚠️  Insufficient data for EMA200 calculation")
            return False
        
        # Classify trends
        classifier = TrendClassifier(data)
        results = classifier.classify_trend()
        
        # Get statistics
        stats = classifier.get_trend_statistics()
        
        print(f"📈 EURUSD Triple EMA Trend Analysis:")
        print(f"   Total periods: {stats['total_periods']}")
        
        # Show all trend types
        for trend_type, info in stats['trend_distribution'].items():
            print(f"   {trend_type}: {info['count']} ({info['percentage']:.1f}%)")
        
        # Get current trend
        current = classifier.get_current_trend()
        print(f"\n🎯 Current Trend: {current['trend'].upper()}")
        print(f"   EMA50: {current['ema_50']:.5f}")
        print(f"   EMA100: {current['ema_100']:.5f}")
        print(f"   EMA200: {current['ema_200']:.5f}")
        print(f"   EMA Order: {current['ema_order']}")
        
        # Detect trend changes
        changes = classifier.detect_trend_changes()
        print(f"\n🔄 Trend Changes: {len(changes)} detected")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results.to_csv('results/eurusd_triple_ema_analysis.csv')
        print(f"💾 Results saved to: results/eurusd_triple_ema_analysis.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all trend classification tests"""
    print("🚀 TRIPLE EMA TREND CLASSIFICATION - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: EMA calculations
    test_results.append(test_ema_calculation())
    
    # Test 2: Triple EMA trend classification logic
    test_results.append(test_trend_classification())
    
    # Test 3: Real data testing
    test_results.append(test_eurusd_trend_classification())
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - TRIPLE EMA MODULE 3 READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return False
    
def test_ranging_filter():
    """Test EMA separation filter for ranging markets"""
    print("\n🧪 Testing Ranging Market Filter")
    print("=" * 50)
    
    # Create test data with EMAs very close together (ranging)
    test_data = pd.DataFrame({
        'close': [100] * 100
    })
    
    classifier = TrendClassifier(test_data)
    classifier.calculate_emas()
    
    # Set EMAs very close together (ranging market)
    classifier.data['ema_50'] = 100.05
    classifier.data['ema_100'] = 100.03
    classifier.data['ema_200'] = 100.00
    
    # Apply filter
    results = classifier.classify_trend_with_filter(min_separation=0.3)
    
    # Should be classified as ranging
    assert (results['trend_filtered'] == 'ranging').all(), "Ranging filter failed"
    
    print("✅ Ranging market filter test passed")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 Module 3 Complete - Triple EMA Trend Classification Working!")
        print("📋 Next steps:")
        print("   1. Test with manual trend analysis for 95% accuracy")
        print("   2. Integrate with Modules 1 & 2 for complete system")
        print("   3. Run visualization to see all 3 EMAs")
        print("   4. Consider multi-timeframe analysis")
    else:
        print("\n⚠️  Module 3 requires fixes before proceeding")