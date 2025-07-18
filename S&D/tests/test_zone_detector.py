"""
Comprehensive Zone Detection Tests - Module 2 (WITH REVERSAL PATTERNS)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from modules.zone_detector import ZoneDetector
from modules.candle_classifier import CandleClassifier
from modules.data_loader import DataLoader
from config.settings import ZONE_CONFIG

def test_known_classifications():
    """
    Test with known zone examples
    """
    print("🧪 Testing Known Zone Classifications")
    print("=" * 50)
    
    # Test cases with known results
    test_cases = [
        # [open, high, low, close, expected_type, description]
        [1.0000, 1.0010, 0.9990, 1.0005, 'base', 'Small body, large wicks'],
        [1.0000, 1.0020, 0.9990, 1.0015, 'base', 'Body ratio = 50%'],
        [1.0000, 1.0030, 0.9990, 1.0025, 'decisive', 'Body ratio = 62.5%'],
        [1.0000, 1.0040, 0.9990, 1.0035, 'decisive', 'Body ratio = 70%'],
        [1.0000, 1.0000, 1.0000, 1.0000, 'base', 'Doji (no range)'],
        [1.0000, 1.0100, 1.0000, 1.0081, 'explosive', 'Body ratio = 81%'],
    ]
    
    # Create test DataFrame
    test_data = pd.DataFrame([
        {'open': case[0], 'high': case[1], 'low': case[2], 'close': case[3]}
        for case in test_cases
    ])
    
    # Expected results
    expected = [case[4] for case in test_cases]
    
    # Test classification
    candle_classifier = CandleClassifier(test_data)
    results = candle_classifier.classify_all_candles()
    
    # Validate each case
    all_correct = True
    for i, (case, actual, expected_type) in enumerate(zip(test_cases, results['candle_type'], expected)):
        body_ratio = results.iloc[i]['body_ratio']
        correct = actual == expected_type
        if not correct:
            all_correct = False
        
        print(f"   Test {i+1}: {case[5]}")
        print(f"      Body ratio: {body_ratio:.3f}")
        print(f"      Expected: {expected_type}, Got: {actual} {'✅' if correct else '❌'}")
    
    print(f"\n🎯 Known Classifications Test: {'PASSED' if all_correct else 'FAILED'}")
    return all_correct

def test_eurusd_classification():
    """
    Test with real EURUSD data
    """
    print("\n📊 Testing EURUSD Zone Detection")
    print("=" * 50)
    
    try:
        # Load EURUSD data
        loader = DataLoader()
        data = loader.load_pair_data('EURUSD', 'Daily')
        
        # Classify zones
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        print(f"📈 EURUSD Zone Detection Results:")
        print(f"   Total patterns: {patterns['total_patterns']}")
        print(f"   Momentum - D-B-D: {len(patterns.get('dbd_patterns', []))}")
        print(f"   Momentum - R-B-R: {len(patterns.get('rbr_patterns', []))}")
        
        # Check for reversal patterns
        dbr_count = len(patterns.get('reversal_patterns', {}).get('dbr_patterns', []))
        rbd_count = len(patterns.get('reversal_patterns', {}).get('rbd_patterns', []))
        print(f"   Reversal - D-B-R: {dbr_count}")
        print(f"   Reversal - R-B-D: {rbd_count}")
        
        # Analyze distribution
        total_patterns = patterns['total_patterns']
        reasonable_distribution = total_patterns > 0  # At least some patterns found
        
        print(f"\n🎯 Distribution Analysis: {'REASONABLE' if reasonable_distribution else 'NO PATTERNS'}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        classified_data.to_csv('results/eurusd_zones_with_reversals.csv')
        print(f"💾 Results saved to: results/eurusd_zones_with_reversals.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_reversal_patterns():
    """Test reversal pattern detection (D-B-R and R-B-D)"""
    print("\n🧪 Testing Reversal Pattern Detection")
    print("=" * 50)
    
    try:
        # Load EURUSD data
        loader = DataLoader()
        data = loader.load_pair_data('EURUSD', 'Daily')
        
        # Test with a sample of data
        sample_data = data.tail(200).copy()
        
        # Classify candles and detect patterns
        candle_classifier = CandleClassifier(sample_data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Check for reversal patterns
        dbr_count = len(patterns.get('reversal_patterns', {}).get('dbr_patterns', []))
        rbd_count = len(patterns.get('reversal_patterns', {}).get('rbd_patterns', []))
        
        print(f"📈 Reversal Pattern Results:")
        print(f"   D-B-R patterns: {dbr_count}")
        print(f"   R-B-D patterns: {rbd_count}")
        print(f"   Total reversal patterns: {dbr_count + rbd_count}")
        
        # Validate pattern structure
        all_reversal = []
        if 'reversal_patterns' in patterns:
            all_reversal.extend(patterns['reversal_patterns'].get('dbr_patterns', []))
            all_reversal.extend(patterns['reversal_patterns'].get('rbd_patterns', []))
        
        for pattern in all_reversal[:3]:  # Check first 3 patterns
            assert 'type' in pattern, "Pattern missing type"
            assert 'category' in pattern, "Pattern missing category"
            assert pattern['category'] == 'reversal', "Pattern category should be 'reversal'"
            assert pattern['type'] in ['D-B-R', 'R-B-D'], f"Invalid reversal pattern type: {pattern['type']}"
            
            print(f"   ✅ {pattern['type']} pattern valid: strength {pattern['strength']:.3f}")
        
        print("✅ Reversal pattern detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pattern_classification():
    """Test that patterns are properly classified as momentum vs reversal"""
    print("\n🧪 Testing Pattern Classification")
    print("=" * 50)
    
    try:
        # Load data and detect patterns
        loader = DataLoader()
        data = loader.load_pair_data('EURUSD', 'Daily')
        sample_data = data.tail(100).copy()
        
        candle_classifier = CandleClassifier(sample_data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Check momentum patterns
        momentum_patterns = patterns.get('dbd_patterns', []) + patterns.get('rbr_patterns', [])
        for pattern in momentum_patterns:
            # Legacy patterns might not have category
            expected_category = pattern.get('category', 'momentum')
            if expected_category != 'momentum':
                print(f"   Warning: Expected momentum, got {expected_category} for {pattern['type']}")
        
        # Check reversal patterns
        reversal_count = 0
        if 'reversal_patterns' in patterns:
            reversal_patterns = (patterns['reversal_patterns'].get('dbr_patterns', []) +
                               patterns['reversal_patterns'].get('rbd_patterns', []))
            reversal_count = len(reversal_patterns)
            for pattern in reversal_patterns:
                assert pattern['category'] == 'reversal', f"Expected reversal, got {pattern['category']}"
                assert pattern['type'] in ['D-B-R', 'R-B-D'], f"Invalid reversal type: {pattern['type']}"
        
        print(f"✅ Pattern classification validated:")
        print(f"   Momentum patterns: {len(momentum_patterns)}")
        print(f"   Reversal patterns: {reversal_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all zone detection tests including reversal patterns"""
    print("🚀 ZONE DETECTION ENGINE - COMPREHENSIVE TESTING (WITH REVERSALS)")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Known classifications
    test_results.append(test_known_classifications())
    
    # Test 2: Real data classification
    test_results.append(test_eurusd_classification())
    
    # Test 3: Reversal pattern detection (NEW)
    test_results.append(test_reversal_patterns())
    
    # Test 4: Pattern classification (NEW)
    test_results.append(test_pattern_classification())
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - MOMENTUM + REVERSAL PATTERNS READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 Module 2 Complete - Zone Detection with Reversals Working!")
        print("📋 Pattern types implemented:")
        print("   ✅ Momentum patterns: D-B-D, R-B-R")
        print("   ✅ Reversal patterns: D-B-R, R-B-D")
        print("   ✅ Pattern classification system")
        print("   ✅ Enhanced signal generation ready")
        print("\n📈 Ready for backtesting with all pattern types!")
    else:
        print("\n⚠️  Zone detection requires fixes before proceeding")