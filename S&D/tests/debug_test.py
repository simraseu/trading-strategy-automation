"""
Quick debug test for candle classification
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.candle_classifier import CandleClassifier
import pandas as pd

def debug_single_candle():
    """Test a single candle classification"""
    print("🔍 Debug Test - Single Candle Classification")
    
    # Create a test candle with exactly 50% body ratio
    test_data = pd.DataFrame([{
        'open': 1.0000,
        'high': 1.0020,
        'low': 0.9990,
        'close': 1.0015
    }])
    
    classifier = CandleClassifier(test_data)
    
    # Test the single candle method directly
    result = classifier.classify_single_candle(1.0000, 1.0020, 0.9990, 1.0015)
    
    # Calculate expected values
    body_size = abs(1.0015 - 1.0000)  # 0.0015
    total_range = 1.0020 - 0.9990     # 0.0030
    body_ratio = body_size / total_range  # 0.5 (50%)
    
    print(f"   Body size: {body_size}")
    print(f"   Total range: {total_range}")
    print(f"   Body ratio: {body_ratio} (50%)")
    print(f"   Classification: {result}")
    print(f"   Expected: base (because 50% <= 50%)")
    print(f"   Result: {'✅ CORRECT' if result == 'base' else '❌ INCORRECT'}")
    
    # Test exactly 80% (should be decisive, not explosive)
    result_80 = classifier.classify_single_candle(1.0000, 1.0100, 1.0000, 1.0080)
    body_ratio_80 = 0.0080 / 0.0100  # 80%
    print(f"\n   80% body ratio test:")
    print(f"   Body ratio: {body_ratio_80} (80%)")
    print(f"   Classification: {result_80}")
    print(f"   Expected: decisive (because 80% is NOT > 80%)")
    print(f"   Result: {'✅ CORRECT' if result_80 == 'decisive' else '❌ INCORRECT'}")
    
    # Test 81% (should be explosive)
    result_81 = classifier.classify_single_candle(1.0000, 1.0100, 1.0000, 1.0081)
    body_ratio_81 = 0.0081 / 0.0100  # 81%
    print(f"\n   81% body ratio test:")
    print(f"   Body ratio: {body_ratio_81} (81%)")
    print(f"   Classification: {result_81}")
    print(f"   Expected: explosive (because 81% > 80%)")
    print(f"   Result: {'✅ CORRECT' if result_81 == 'explosive' else '❌ INCORRECT'}")

if __name__ == "__main__":
    debug_single_candle()