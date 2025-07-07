"""
Test the risk validation fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

def test_risk_validation_fixes():
    """Test if risk validation now works"""
    print("🧪 TESTING RISK VALIDATION FIXES")
    print("=" * 40)
    
    # Load data - same as before
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    test_date = '2025-06-20'
    date_idx = data.index.get_loc(test_date)
    history_start = max(0, date_idx - 365)
    historical_data = data.iloc[history_start:date_idx]
    current_price = historical_data['close'].iloc[-1]
    
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
    
    # Get trend-aligned zones
    all_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
    bullish_trends = ['strong_bullish', 'medium_bullish', 'weak_bullish']
    
    aligned_zones = []
    for zone in all_zones:
        if current_trend in bullish_trends and zone['type'] == 'R-B-R':
            aligned_zones.append(zone)
    
    print(f"🎯 Trend-aligned zones: {len(aligned_zones)}")
    
    # Test each zone with FIXES
    tradeable_count = 0
    for i, zone in enumerate(aligned_zones):
        print(f"\n🧪 Testing Zone {i+1} (FIXED)")
        
        risk_validation = risk_manager.validate_zone_for_trading(
            zone, current_price, 'EURUSD', historical_data
        )
        
        if risk_validation['is_tradeable']:
            tradeable_count += 1
            print(f"   ✅ TRADEABLE!")
            print(f"      Entry: {risk_validation['entry_price']:.5f}")
            print(f"      Stop: {risk_validation['stop_loss_price']:.5f}")
            print(f"      Position size: {risk_validation['position_size']}")
        else:
            print(f"   ❌ Rejected: {risk_validation['reason']}")
    
    print(f"\n📊 RESULTS AFTER FIXES:")
    print(f"   Tradeable zones: {tradeable_count}/{len(aligned_zones)}")
    
    if tradeable_count > 0:
        print("✅ FIXES SUCCESSFUL - Signals should now generate!")
    else:
        print("❌ Still no tradeable signals - need further adjustment")

if __name__ == "__main__":
    test_risk_validation_fixes()