"""
Debug Signal Generation Flow - Find Where It Goes Wrong
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

def debug_signal_flow():
    """Debug each step of signal generation"""
    print("🔍 DEBUGGING SIGNAL GENERATION FLOW")
    print("=" * 60)
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    current_price = data['close'].iloc[-1]
    
    print(f"📊 Data: {len(data)} candles")
    print(f"💰 Current price: {current_price:.5f}")
    print(f"📅 Recent dates: {data.index[-5:].tolist()}")
    
    # Step 1: Zone Detection
    candle_classifier = CandleClassifier(data)
    classified_data = candle_classifier.classify_all_candles()
    
    zone_detector = ZoneDetector(candle_classifier)
    print(f"\n🎯 STEP 1: ZONE DETECTION")
    patterns = zone_detector.detect_all_patterns(classified_data)
    
    all_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
    print(f"   Total zones detected: {len(all_zones)}")
    
    # Analyze zone recency
    recent_zones = []
    old_zones = []
    
    for zone in all_zones:
        zone_end_date = data.index[zone['end_idx']]
        days_ago = (data.index[-1] - zone_end_date).days
        
        if days_ago <= 365:  # Last year
            recent_zones.append((zone, days_ago))
        else:
            old_zones.append((zone, days_ago))
    
    print(f"   Recent zones (last year): {len(recent_zones)}")
    print(f"   Old zones (>1 year): {len(old_zones)}")
    
    # Show recent zones
    if recent_zones:
        print(f"\n📋 RECENT ZONES (last year):")
        recent_zones.sort(key=lambda x: x[1])  # Sort by days ago
        for i, (zone, days_ago) in enumerate(recent_zones[:10]):
            zone_date = data.index[zone['end_idx']]
            print(f"   {i+1:2d}. {zone['type']} | {zone['zone_low']:.5f}-{zone['zone_high']:.5f} | {days_ago:3d} days ago ({zone_date.strftime('%Y-%m-%d')})")
    
    # Step 2: Trend Classification
    print(f"\n📊 STEP 2: TREND CLASSIFICATION")
    trend_classifier = TrendClassifier(data)
    trend_data = trend_classifier.classify_trend_with_filter()
    current_trend = trend_data['trend_filtered'].iloc[-1]
    print(f"   Current trend: {current_trend}")
    
    # Step 3: Trend Alignment
    print(f"\n🎯 STEP 3: TREND ALIGNMENT FILTER")
    bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
    bearish_trends = ['strong_bearish', 'medium_bearish', 'weak_bearish']
    
    aligned_zones = []
    for zone in all_zones:
        zone_type = zone['type']
        if current_trend in bullish_trends and zone_type == 'R-B-R':
            aligned_zones.append(zone)
        elif current_trend in bearish_trends and zone_type == 'D-B-D':
            aligned_zones.append(zone)
    
    print(f"   Zones before alignment: {len(all_zones)}")
    print(f"   Zones after alignment: {len(aligned_zones)}")
    
    # Show aligned zones
    if aligned_zones:
        print(f"\n📋 TREND-ALIGNED ZONES:")
        for i, zone in enumerate(aligned_zones[:10]):
            zone_date = data.index[zone['end_idx']]
            days_ago = (data.index[-1] - zone_date).days
            print(f"   {i+1:2d}. {zone['type']} | {zone['zone_low']:.5f}-{zone['zone_high']:.5f} | {days_ago:3d} days ago")
    
    # Step 4: Risk Validation
    print(f"\n🛡️  STEP 4: RISK VALIDATION")
    risk_manager = RiskManager(account_balance=10000)
    
    tradeable_count = 0
    untradeable_reasons = {}
    
    for zone in aligned_zones:
        risk_validation = risk_manager.validate_zone_for_trading(zone, current_price, 'EURUSD')
        
        if risk_validation['is_tradeable']:
            tradeable_count += 1
        else:
            reason = risk_validation['reason']
            untradeable_reasons[reason] = untradeable_reasons.get(reason, 0) + 1
    
    print(f"   Zones after risk validation: {tradeable_count}")
    
    if untradeable_reasons:
        print(f"   Rejection reasons:")
        for reason, count in untradeable_reasons.items():
            print(f"      {reason}: {count} zones")
    
    # Step 5: Show the actual 4 signals that are generated
    print(f"\n🎯 STEP 5: FINAL SIGNAL ANALYSIS")
    signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
    signals = signal_generator.generate_signals(classified_data, 'Daily', 'EURUSD')
    
    if signals:
        print(f"   Final signals generated: {len(signals)}")
        for i, signal in enumerate(signals):
            print(f"\n   Signal {i+1}:")
            print(f"      Zone: {signal['zone_low']:.5f} - {signal['zone_high']:.5f}")
            print(f"      Entry: {signal['entry_price']:.5f}")
            print(f"      Stop: {signal['stop_loss']:.5f}")
            print(f"      Score: {signal['signal_score']:.1f}")
            
            # Try to find when this zone was formed
            zone_center = (signal['zone_high'] + signal['zone_low']) / 2
            print(f"      Zone center: {zone_center:.5f}")
            print(f"      Distance from current: {abs(zone_center - current_price):.5f}")

if __name__ == "__main__":
    debug_signal_flow()