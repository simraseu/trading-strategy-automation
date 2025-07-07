"""
Debug Risk Validation Failure
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

def debug_risk_validation_detailed():
    """Debug exactly why risk validation is failing"""
    print("🔍 DEBUGGING RISK VALIDATION FAILURE")
    print("=" * 50)
    
    # Load data - use same window as your debug test
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    # Use 2025-06-20 data (when price was 1.14987)
    test_date = '2025-06-20'
    date_idx = data.index.get_loc(test_date)
    history_start = max(0, date_idx - 365)
    historical_data = data.iloc[history_start:date_idx]
    current_price = historical_data['close'].iloc[-1]
    
    print(f"📅 Test date: {test_date}")
    print(f"💰 Current price: {current_price:.5f}")
    
    # Initialize components
    candle_classifier = CandleClassifier(historical_data)
    classified_data = candle_classifier.classify_all_candles()
    
    zone_detector = ZoneDetector(candle_classifier)
    patterns = zone_detector.detect_all_patterns(classified_data)
    
    trend_classifier = TrendClassifier(historical_data)
    trend_data = trend_classifier.classify_trend_with_filter()
    current_trend = trend_data['trend_filtered'].iloc[-1]
    
    risk_manager = RiskManager(account_balance=10000)
    
    print(f"📊 Current trend: {current_trend}")
    print(f"🎯 Total zones: {patterns['total_patterns']}")
    
    # Get trend-aligned zones manually
    all_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
    bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
    
    aligned_zones = []
    for zone in all_zones:
        if current_trend in bullish_trends and zone['type'] == 'R-B-R':
            aligned_zones.append(zone)
    
    print(f"🎯 Trend-aligned zones: {len(aligned_zones)}")
    
    # Test each zone individually for risk validation
    for i, zone in enumerate(aligned_zones):
        print(f"\n🧪 Testing Zone {i+1}: {zone['type']}")
        print(f"   Zone range: {zone['zone_low']:.5f} - {zone['zone_high']:.5f}")
        print(f"   Zone center: {(zone['zone_low'] + zone['zone_high'])/2:.5f}")
        print(f"   Distance from market: {abs((zone['zone_low'] + zone['zone_high'])/2 - current_price):.5f}")
        
        # Test risk validation step by step
        risk_validation = risk_manager.validate_zone_for_trading(
            zone, current_price, 'EURUSD', historical_data
        )
        
        print(f"   Risk validation result: {risk_validation['is_tradeable']}")
        if not risk_validation['is_tradeable']:
            print(f"   Rejection reason: {risk_validation['reason']}")
        else:
            print(f"   ✅ TRADEABLE!")
            print(f"      Entry: {risk_validation['entry_price']:.5f}")
            print(f"      Stop: {risk_validation['stop_loss_price']:.5f}")
            print(f"      Position size: {risk_validation['position_size']}")

def debug_zone_testing_logic():
    """Debug the zone testing (33% penetration) logic specifically"""
    print("\n🔍 DEBUGGING ZONE TESTING LOGIC")
    print("=" * 40)
    
    # Load full data
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    # Get a zone from historical data
    candle_classifier = CandleClassifier(data)
    classified_data = candle_classifier.classify_all_candles()
    
    zone_detector = ZoneDetector(candle_classifier)
    patterns = zone_detector.detect_all_patterns(classified_data)
    
    risk_manager = RiskManager()
    
    # Test first R-B-R zone
    rbr_zones = patterns['rbr_patterns']
    if rbr_zones:
        test_zone = rbr_zones[0]
        print(f"🎯 Testing zone from index {test_zone['start_idx']} to {test_zone['end_idx']}")
        print(f"   Zone: {test_zone['zone_low']:.5f} - {test_zone['zone_high']:.5f}")
        print(f"   Formation date: {data.index[test_zone['end_idx']]}")
        
        # Check if zone has been tested
        is_valid, reason = risk_manager.check_zone_testing(test_zone, data)
        print(f"   Zone testing result: {is_valid}")
        print(f"   Reason: {reason}")
        
        # If zone was tested, show when
        if not is_valid:
            zone_end_idx = test_zone['end_idx']
            zone_high = test_zone['zone_high']
            zone_low = test_zone['zone_low']
            zone_size = zone_high - zone_low
            test_level = zone_high - (zone_size * 0.33)
            
            print(f"   33% test level: {test_level:.5f}")
            
            # Find when it was tested
            for i in range(zone_end_idx + 1, min(zone_end_idx + 100, len(data))):
                candle = data.iloc[i]
                if candle['close'] < test_level:
                    print(f"   Tested on: {data.index[i]} at close {candle['close']:.5f}")
                    break

if __name__ == "__main__":
    debug_risk_validation_detailed()
    debug_zone_testing_logic()