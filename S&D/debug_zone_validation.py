"""
Debug zone validation issues
"""
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

def debug_zone_boundaries():
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    # Get recent data
    recent_data = data.tail(100)
    
    print(f"📊 Recent price range: {recent_data['low'].min():.5f} - {recent_data['high'].max():.5f}")
    
    # Detect zones
    candle_classifier = CandleClassifier(recent_data)
    classified_data = candle_classifier.classify_all_candles()
    
    zone_detector = ZoneDetector(candle_classifier)
    patterns = zone_detector.detect_all_patterns(classified_data)
    
    print(f"\n🎯 Found {patterns['total_patterns']} patterns")
    
    # Validate each zone
    all_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
    for i, zone in enumerate(all_zones[:5]):  # Check first 5
        print(f"\nZone {i+1}: {zone['type']}")
        print(f"   Zone: {zone['zone_low']:.5f} - {zone['zone_high']:.5f}")
        print(f"   Base indices: {zone['base']['start_idx']} - {zone['base']['end_idx']}")
        
        # Check if zone overlaps any recent candles
        overlaps = False
        for idx, candle in recent_data.iterrows():
            if candle['low'] <= zone['zone_high'] and candle['high'] >= zone['zone_low']:
                overlaps = True
                break
        
        print(f"   Overlaps recent candles: {overlaps}")
        if not overlaps:
            print(f"   ❌ INVALID ZONE - no candle overlap!")

if __name__ == "__main__":
    debug_zone_boundaries()