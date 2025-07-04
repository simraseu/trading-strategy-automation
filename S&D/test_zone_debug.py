"""
Zone Detection Debug Tool - Module 2
Quick testing and debugging for zone detection development
"""

import sys
import os
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from config.settings import ZONE_CONFIG, CANDLE_THRESHOLDS, DATA_SETTINGS, PATHS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def debug_zone_detection():
    """Debug zone detection with sample data"""
    print("=" * 60)
    print("ZONE DETECTION DEBUG TOOL - MODULE 2")
    print("=" * 60)
    
    try:
        # Load data
        print("\n1. Loading data...")
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')  # Use correct method
        print(f"   ✅ Loaded {len(data)} candles")

        # Test with sample data (first 100 candles for speed)
        print("\n3. Testing with sample data...")
        sample_data = data.head(100)
        print(f"   📊 Sample size: {len(sample_data)} candles")

        # DEBUG: Print column names
        print(f"   🔍 Column names: {list(sample_data.columns)}")
        print(f"   🔍 Data sample:")
        print(sample_data.head(2))
        
        # Initialize components
        print("\n2. Initializing components...")
        candle_classifier = CandleClassifier(data)  # Pass data to classifier
        zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
        print("   ✅ Components initialized")
        
        # Test with sample data (first 100 candles for speed)
        print("\n3. Testing with sample data...")
        sample_data = data.head(100)
        print(f"   📊 Sample size: {len(sample_data)} candles")
        
        # Run pattern detection
        print("\n4. Running pattern detection...")
        patterns = zone_detector.detect_all_patterns(sample_data)
        
        # Display results
        print("\n5. PATTERN DETECTION RESULTS:")
        print(f"   📈 Total patterns found: {patterns['total_patterns']}")
        print(f"   📉 D-B-D patterns: {len(patterns['dbd_patterns'])}")
        print(f"   📊 R-B-R patterns: {len(patterns['rbr_patterns'])}")
        
        # Show pattern details
        if patterns['dbd_patterns']:
            print("\n   D-B-D Pattern Details:")
            for i, pattern in enumerate(patterns['dbd_patterns'][:3]):  # Show first 3
                print(f"     Pattern {i+1}:")
                print(f"       Strength: {pattern['strength']:.3f}")
                print(f"       Zone Range: {pattern['zone_range']:.5f}")
                print(f"       Base Candles: {pattern['base']['candle_count']}")
                print(f"       Leg-out Ratio: {pattern['leg_out']['ratio_to_base']:.2f}")
        
        if patterns['rbr_patterns']:
            print("\n   R-B-R Pattern Details:")
            for i, pattern in enumerate(patterns['rbr_patterns'][:3]):  # Show first 3
                print(f"     Pattern {i+1}:")
                print(f"       Strength: {pattern['strength']:.3f}")
                print(f"       Zone Range: {pattern['zone_range']:.5f}")
                print(f"       Base Candles: {pattern['base']['candle_count']}")
                print(f"       Leg-out Ratio: {pattern['leg_out']['ratio_to_base']:.2f}")
        
        # Generate summary
        print("\n6. PATTERN SUMMARY:")
        summary = zone_detector.get_pattern_summary(patterns)
        print(f"   📊 Average Strength: {summary['avg_strength']:.3f}")
        print(f"   🔥 High Strength (≥0.8): {summary['strength_distribution']['high']}")
        print(f"   🔸 Medium Strength (0.5-0.8): {summary['strength_distribution']['medium']}")
        print(f"   🔹 Low Strength (<0.5): {summary['strength_distribution']['low']}")
        
        # Test performance
        print("\n7. PERFORMANCE TEST:")
        import time

        # Test with full dataset
        print(f"   🔍 Testing full dataset: {len(data)} candles")
        start_time = time.time()
        full_patterns = zone_detector.detect_all_patterns(data)  # ← FULL DATASET
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"   ⏱️  Processing time (full dataset): {processing_time:.2f}s")
        print(f"   📈 Patterns found: {full_patterns['total_patterns']}")
        print(f"   🚀 Processing rate: {len(data)/processing_time:.0f} candles/second")

        # Show detailed breakdown
        print(f"   📉 D-B-D patterns: {len(full_patterns['dbd_patterns'])}")
        print(f"   📊 R-B-R patterns: {len(full_patterns['rbr_patterns'])}")
        print(f"   📈 Pattern density: {full_patterns['total_patterns']/len(data)*100:.1f}% of candles")
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ ZONE DETECTION DEBUG COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in zone detection debug: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_patterns():
    """Test specific pattern scenarios"""
    print("\n" + "=" * 40)
    print("SPECIFIC PATTERN TESTS")
    print("=" * 40)
    
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')  # Use correct method
        
        # Initialize components
        candle_classifier = CandleClassifier(data)  # Pass data to classifier
        zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
        
        # Test D-B-D detection specifically
        print("\n1. Testing D-B-D Detection:")
        sample_data = data.head(200)
        dbd_patterns = zone_detector.detect_dbd_patterns(sample_data)
        
        print(f"   Found {len(dbd_patterns)} D-B-D patterns")
        
        if dbd_patterns:
            best_pattern = max(dbd_patterns, key=lambda x: x['strength'])
            print(f"   Best D-B-D strength: {best_pattern['strength']:.3f}")
            print(f"   Best D-B-D base candles: {best_pattern['base']['candle_count']}")
        
        # Test R-B-R detection specifically
        print("\n2. Testing R-B-R Detection:")
        rbr_patterns = zone_detector.detect_rbr_patterns(sample_data)
        
        print(f"   Found {len(rbr_patterns)} R-B-R patterns")
        
        if rbr_patterns:
            best_pattern = max(rbr_patterns, key=lambda x: x['strength'])
            print(f"   Best R-B-R strength: {best_pattern['strength']:.3f}")
            print(f"   Best R-B-R base candles: {best_pattern['base']['candle_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in specific pattern tests: {str(e)}")
        return False


if __name__ == "__main__":
    # Run debug tests
    success = debug_zone_detection()
    
    if success:
        test_specific_patterns()
        print("\n🎉 All debug tests completed!")
    else:
        print("\n💥 Debug tests failed!")