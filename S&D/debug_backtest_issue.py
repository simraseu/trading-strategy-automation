"""
Zone Detection Debug - Find Missing Trades
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
import pandas as pd

def debug_missing_zones():
    """Debug specific missing trades T008, T012, T014"""
    
    missing_trades = [
        {'id': 'T008', 'zone_low': 1.0665, 'zone_high': 1.0706, 'base_end': '30-05-2024', 'type': 'R-B-R'},
        {'id': 'T012', 'zone_low': 1.0595, 'zone_high': 1.0663, 'base_end': '18-06-2024', 'type': 'R-B-R'},
        {'id': 'T014', 'zone_low': 1.0913, 'zone_high': 1.1095, 'base_end': '11-07-2024', 'type': 'R-B-R'}
    ]
    
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    for trade in missing_trades:
        print(f"\n🔍 DEBUGGING {trade['id']}: {trade['zone_low']:.4f}-{trade['zone_high']:.4f}")
        print("=" * 60)
        
        # Parse date and create window
        base_end_date = pd.to_datetime(trade['base_end'], format='%d-%m-%Y')
        base_end_idx = data.index.get_loc(base_end_date)
        
        window_start = max(0, base_end_idx - 50)
        window_end = min(len(data), base_end_idx + 20)
        window_data = data.iloc[window_start:window_end]
        
        print(f"📅 Base end date: {base_end_date}")
        print(f"📊 Analysis window: {window_data.index[0]} to {window_data.index[-1]}")
        print(f"💰 Price range in window: {window_data['low'].min():.4f} - {window_data['high'].max():.4f}")
        
        # Check if zone is within price range
        zone_center = (trade['zone_high'] + trade['zone_low']) / 2
        data_center = (window_data['high'].max() + window_data['low'].min()) / 2
        distance_from_data = abs(zone_center - data_center)
        
        print(f"🎯 Zone center: {zone_center:.4f}")
        print(f"📈 Data center: {data_center:.4f}")
        print(f"📏 Distance: {distance_from_data:.4f} ({distance_from_data*10000:.0f} pips)")
        
        # Run zone detection
        candle_classifier = CandleClassifier(window_data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        print(f"🔍 Zones found: {patterns['total_patterns']}")
        print(f"   R-B-R: {len(patterns['rbr_patterns'])}")
        print(f"   D-B-D: {len(patterns['dbd_patterns'])}")
        
        # Check for any zones near target
        target_zones = patterns['rbr_patterns'] if trade['type'] == 'R-B-R' else patterns['dbd_patterns']
        
        if target_zones:
            print(f"\n📋 {trade['type']} zones in window:")
            for i, zone in enumerate(target_zones):
                detected_center = (zone['zone_high'] + zone['zone_low']) / 2
                center_diff = abs(detected_center - zone_center)
                
                print(f"   Zone {i+1}: {zone['zone_low']:.4f}-{zone['zone_high']:.4f}")
                print(f"      Center: {detected_center:.4f} (diff: {center_diff:.4f})")
                print(f"      Base candles: {zone['base']['candle_count']}")
                print(f"      Leg-out ratio: {zone['leg_out']['ratio_to_base']:.2f}")
                
                if center_diff < 0.0200:  # Within 200 pips
                    print(f"      ✅ POTENTIAL MATCH!")
                else:
                    print(f"      ❌ Too far from target")
        else:
            print(f"   ❌ NO {trade['type']} zones detected")
        
        # Manual candle analysis around base end date
        print(f"\n🕯️ CANDLE ANALYSIS around {base_end_date}:")
        search_start = max(0, base_end_idx - 5)
        search_end = min(len(data), base_end_idx + 5)
        
        for i in range(search_start, search_end):
            candle = data.iloc[i]
            date = data.index[i]
            
            # Check if candle overlaps with target zone
            overlaps = (candle['low'] <= trade['zone_high'] and candle['high'] >= trade['zone_low'])
            is_base_end = (date == base_end_date)
            
            # Classify candle
            classification = candle_classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            print(f"   {date.strftime('%Y-%m-%d')}: {candle['low']:.4f}-{candle['high']:.4f} "
                  f"[{classification}] {'🎯' if overlaps else '  '} {'📍' if is_base_end else ''}")

if __name__ == "__main__":
    debug_missing_zones()