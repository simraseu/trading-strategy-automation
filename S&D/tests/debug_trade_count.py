"""
DEBUG: Trade Count Discrepancy Analysis
Compare trade generation between distance_edge and complete backtester
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier

def debug_trade_count_analysis():
    """Debug exactly where trades are being lost"""
    
    print("🔍 TRADE COUNT DEBUG ANALYSIS")
    print("=" * 60)
    
    # Target: AUDCAD 1D (the pair with discrepancy)
    pair = 'AUDCAD'
    timeframe = '1D'
    
    print(f"🎯 Testing: {pair} {timeframe}")
    print(f"🎯 Expected: 88 trades (45 momentum + 43 reversal)")
    print(f"🎯 Actual: 34 trades")
    print(f"🎯 Missing: 54 trades")
    
    # Load data using same method as complete backtester
    print(f"\n📊 Loading data...")
    data_loader = DataLoader()
    
    try:
        if timeframe == '1D':
            data = data_loader.load_pair_data(pair, 'Daily')
        else:
            data = data_loader.load_pair_data(pair, timeframe)
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    print(f"✅ Data loaded: {len(data)} candles")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    
    # Apply same data limitation as FIXED complete backtester
    days_back = 99999  # Test with default first
    original_data_length = len(data)

    # Use same logic as your fixed backtester
    if days_back < 9999:
        max_candles = min(days_back + 1000, len(data))
        data = data.iloc[-max_candles:]
        print(f"   📊 Using last {days_back} days + 1000 lookback ({len(data)} candles)")
    elif days_back == 99999:
        print(f"   📊 Using ALL available data ({len(data)} candles)")

    print(f"   Original data length: {original_data_length}")
    print(f"   After fixed filter: {len(data)} candles")
    print(f"   Filtered date range: {data.index[0]} to {data.index[-1]}")
    
    if len(data) < 100:
        print(f"❌ Insufficient data after filtering")
        return
    
    # Step 1: Initialize components (same as complete backtester)
    print(f"\n🔧 Initializing components...")
    candle_classifier = CandleClassifier(data)
    classified_data = candle_classifier.classify_all_candles()
    print(f"✅ Candle classification complete")
    
    zone_detector = ZoneDetector(candle_classifier)
    patterns = zone_detector.detect_all_patterns(classified_data)
    print(f"✅ Zone detection complete")
    
    trend_classifier = TrendClassifier(data)
    trend_data = trend_classifier.classify_trend_simplified()
    print(f"✅ Trend classification complete")
    
    # Step 2: DETAILED PATTERN ANALYSIS
    print(f"\n🔍 DETAILED PATTERN ANALYSIS:")
    print(f"   D-B-D patterns: {len(patterns['dbd_patterns'])}")
    print(f"   R-B-R patterns: {len(patterns['rbr_patterns'])}")
    print(f"   D-B-R patterns: {len(patterns.get('dbr_patterns', []))}")
    print(f"   R-B-D patterns: {len(patterns.get('rbd_patterns', []))}")
    
    # Combine patterns (same logic as complete backtester)
    momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
    reversal_patterns = patterns.get('dbr_patterns', []) + patterns.get('rbd_patterns', [])
    all_patterns = momentum_patterns + reversal_patterns
    
    print(f"\n📊 PATTERN COMBINATION:")
    print(f"   Momentum patterns: {len(momentum_patterns)}")
    print(f"   Reversal patterns: {len(reversal_patterns)}")
    print(f"   Total patterns: {len(all_patterns)}")
    
    # Step 3: DISTANCE FILTERING ANALYSIS
    print(f"\n🔍 DISTANCE FILTERING ANALYSIS:")
    
    # Test different distance thresholds
    distance_thresholds = [2.0, 2.5, 3.0]
    
    for threshold in distance_thresholds:
        valid_patterns = [
            pattern for pattern in all_patterns
            if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
            and pattern['leg_out']['ratio_to_base'] >= threshold
        ]
        print(f"   Distance ≥{threshold}: {len(valid_patterns)} patterns")
    
    # Use 2.5 threshold (same as distance_edge)
    valid_patterns_2_5 = [
        pattern for pattern in all_patterns
        if 'leg_out' in pattern and 'ratio_to_base' in pattern['leg_out']
        and pattern['leg_out']['ratio_to_base'] >= 2.5
    ]
    
    print(f"\n✅ Using 2.5 distance threshold: {len(valid_patterns_2_5)} patterns")
    
    if not valid_patterns_2_5:
        print(f"❌ No patterns meet distance requirement!")
        return
    
    # Step 4: BUILD ACTIVATION SCHEDULE (same as complete backtester)
    print(f"\n🔍 ACTIVATION SCHEDULE ANALYSIS:")
    
    zone_activation_schedule = []
    for i, pattern in enumerate(valid_patterns_2_5):
        zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
        if zone_end_idx is not None and zone_end_idx < len(data):
            zone_activation_schedule.append({
                'date': data.index[zone_end_idx],
                'pattern': pattern,
                'zone_id': f"{pattern['type']}_{zone_end_idx}_{pattern['zone_low']:.5f}",
                'pattern_idx': i
            })
        else:
            print(f"   ⚠️ Pattern {i} has invalid end_idx: {zone_end_idx}")
    
    zone_activation_schedule.sort(key=lambda x: x['date'])
    
    print(f"   Valid activation schedule: {len(zone_activation_schedule)} zones")
    print(f"   Date range: {zone_activation_schedule[0]['date'] if zone_activation_schedule else 'N/A'} to {zone_activation_schedule[-1]['date'] if zone_activation_schedule else 'N/A'}")
    
    # Step 5: TREND ALIGNMENT ANALYSIS
    print(f"\n🔍 TREND ALIGNMENT ANALYSIS:")
    
    aligned_count = 0
    trend_breakdown = {'bullish_aligned': 0, 'bearish_aligned': 0, 'not_aligned': 0}
    
    for zone_info in zone_activation_schedule:
        pattern = zone_info['pattern']
        zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
        
        if zone_end_idx is None or zone_end_idx >= len(trend_data):
            trend_breakdown['not_aligned'] += 1
            continue
        
        current_trend = trend_data['trend'].iloc[zone_end_idx]
        is_aligned = (
            (pattern['type'] in ['R-B-R', 'D-B-R'] and current_trend == 'bullish') or
            (pattern['type'] in ['D-B-D', 'R-B-D'] and current_trend == 'bearish')
        )
        
        if is_aligned:
            aligned_count += 1
            if current_trend == 'bullish':
                trend_breakdown['bullish_aligned'] += 1
            else:
                trend_breakdown['bearish_aligned'] += 1
        else:
            trend_breakdown['not_aligned'] += 1
    
    print(f"   Trend-aligned patterns: {aligned_count}")
    print(f"   Bullish aligned: {trend_breakdown['bullish_aligned']}")
    print(f"   Bearish aligned: {trend_breakdown['bearish_aligned']}")
    print(f"   Not aligned: {trend_breakdown['not_aligned']}")
    
    # Step 6: ENTRY EXECUTION SIMULATION
    print(f"\n🔍 ENTRY EXECUTION ANALYSIS:")
    
    successful_entries = 0
    entry_failures = {'no_entry': 0, 'invalid_stop': 0, 'other': 0}
    
    for zone_info in zone_activation_schedule:
        pattern = zone_info['pattern']
        zone_end_idx = pattern.get('end_idx', pattern.get('base', {}).get('end_idx'))
        
        if zone_end_idx is None or zone_end_idx >= len(trend_data):
            continue
            
        current_trend = trend_data['trend'].iloc[zone_end_idx]
        is_aligned = (
            (pattern['type'] in ['R-B-R', 'D-B-R'] and current_trend == 'bullish') or
            (pattern['type'] in ['D-B-D', 'R-B-D'] and current_trend == 'bearish')
        )
        
        if not is_aligned:
            continue
        
        # Simulate entry logic
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        zone_range = zone_high - zone_low
        
        if pattern['type'] in ['R-B-R', 'D-B-R']:
            entry_price = zone_low + (zone_range * 0.05)
            initial_stop = zone_low - (zone_range * 0.33)
        else:
            entry_price = zone_high - (zone_range * 0.05)
            initial_stop = zone_high + (zone_range * 0.33)
        
        stop_distance = abs(entry_price - initial_stop)
        if stop_distance <= 0:
            entry_failures['invalid_stop'] += 1
            continue
        
        # Check for entry execution
        entry_found = False
        search_limit = min(100, len(data) - zone_end_idx - 1)
        
        for i in range(zone_end_idx + 1, zone_end_idx + 1 + search_limit):
            if i >= len(data):
                break
                
            candle = data.iloc[i]
            
            if pattern['type'] in ['R-B-R', 'D-B-R'] and candle['low'] <= entry_price:
                entry_found = True
                break
            elif pattern['type'] in ['D-B-D', 'R-B-D'] and candle['high'] >= entry_price:
                entry_found = True
                break
        
        if entry_found:
            successful_entries += 1
        else:
            entry_failures['no_entry'] += 1
    
    print(f"   Successful entries: {successful_entries}")
    print(f"   Failed entries - no trigger: {entry_failures['no_entry']}")
    print(f"   Failed entries - invalid stop: {entry_failures['invalid_stop']}")
    print(f"   Failed entries - other: {entry_failures['other']}")
    
    # Step 7: FINAL SUMMARY
    print(f"\n📊 FINAL TRADE COUNT BREAKDOWN:")
    print(f"   Original data length: {original_data_length} candles")
    print(f"   After data filter: {len(data)} candles")
    print(f"   Total patterns detected: {len(all_patterns)}")
    print(f"   After distance filter (≥2.5): {len(valid_patterns_2_5)}")
    print(f"   After activation schedule: {len(zone_activation_schedule)}")
    print(f"   After trend alignment: {aligned_count}")
    print(f"   After entry execution: {successful_entries}")
    print(f"\n🎯 EXPECTED: 88 trades | ACTUAL: {successful_entries} trades")
    print(f"🎯 DISCREPANCY: {88 - successful_entries} missing trades")
    
    if successful_entries != 88:
        print(f"\n⚠️ DISCREPANCY IDENTIFIED!")
        print(f"   Most likely cause: Data filtering or distance threshold difference")
        print(f"   Check: Are both systems using the same date range?")
        print(f"   Check: Are both systems using 2.5 distance threshold?")
    else:
        print(f"\n✅ TRADE COUNTS MATCH!")
        print(f"   The logic is identical - check strategy-specific filtering")

if __name__ == "__main__":
    debug_trade_count_analysis()