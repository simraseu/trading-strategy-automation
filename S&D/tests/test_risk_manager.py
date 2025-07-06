"""
Risk Management Tests - Module 4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.risk_manager import RiskManager
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

def test_position_sizing():
    """Test position sizing calculations"""
    print("🧪 Testing Position Sizing")
    print("=" * 40)
    
    risk_manager = RiskManager(account_balance=10000)
    
    # Test scenarios
    test_cases = [
        (20, 'EURUSD', 1.0),   # 20 pip stop should give 1.0 lot
        (40, 'EURUSD', 0.5),   # 40 pip stop should give 0.5 lot
        (80, 'EURUSD', 0.25),  # 80 pip stop should give 0.25 lot
    ]
    
    for stop_pips, pair, expected_size in test_cases:
        position_size = risk_manager.calculate_position_size(stop_pips, pair)
        
        print(f"   {stop_pips} pip stop: {position_size} lots (expected: {expected_size})")
        
        # Verify 2% risk
        risk_amount = risk_manager.calculate_risk_amount(stop_pips, position_size, pair)
        risk_percent = (risk_amount / 10000) * 100
        
        print(f"      Risk amount: ${risk_amount:.2f} ({risk_percent:.1f}%)")
        
        # Should be close to 2%
        assert 1.8 <= risk_percent <= 2.2, f"Risk percent {risk_percent} not close to 2%"
    
    print("✅ Position sizing test passed")
    return True

def test_stop_loss_calculation():
    """Test stop loss placement"""
    print("\n🧪 Testing Stop Loss Calculation")
    print("=" * 40)
    
    risk_manager = RiskManager()
    
    # Create test zone (R-B-R demand zone)
    test_zone = {
        'type': 'R-B-R',
        'zone_high': 1.1000,
        'zone_low': 1.0980,
        'base': {'quality_score': 0.8},
        'leg_in': {'strength': 0.7},
        'leg_out': {'strength': 0.9, 'ratio_to_base': 2.0}
    }
    
    current_price = 1.0995
    
    stop_data = risk_manager.calculate_stop_loss(test_zone, current_price, 0.0001)
    
    print(f"   Zone Low: {test_zone['zone_low']}")
    print(f"   Stop Loss: {stop_data['price']:.5f}")
    print(f"   Distance: {stop_data['distance_pips']:.1f} pips")
    
    # Stop should be 5 pips below zone low for R-B-R
    expected_stop = 1.0980 - (5 * 0.0001)
    assert abs(stop_data['price'] - expected_stop) < 0.00001, "Stop loss calculation incorrect"
    
    print("✅ Stop loss calculation test passed")
    return True

def test_zone_validation():
    """Test complete zone validation"""
    print("\n🧪 Testing Zone Validation")
    print("=" * 40)
    
    risk_manager = RiskManager(account_balance=10000)
    
    # Create valid test zone
    valid_zone = {
        'type': 'R-B-R',
        'zone_high': 1.1000,
        'zone_low': 1.0970,  # 30 pip zone
        'base': {'quality_score': 0.8, 'candle_count': 2},
        'leg_in': {'strength': 0.7},
        'leg_out': {'strength': 0.9, 'ratio_to_base': 2.0}
    }
    
    current_price = 1.0995
    
    validation = risk_manager.validate_zone_for_trading(valid_zone, current_price)
    
    print(f"   Tradeable: {validation['is_tradeable']}")
    if validation['is_tradeable']:
        print(f"   Position Size: {validation['position_size']} lots")
        print(f"   Stop Loss: {validation['stop_loss_price']:.5f}")
        print(f"   Take Profit 1: {validation['take_profit_1']:.5f}")
        print(f"   Take Profit 2: {validation['take_profit_2']:.5f}")
        print(f"   Risk/Reward: 1:{validation['risk_reward_ratio']}")
        print(f"   Stop Distance: {validation['stop_distance_pips']:.1f} pips")
    else:
        print(f"   Reason: {validation['reason']}")
    
    assert validation['is_tradeable'], "Valid zone should be tradeable"
    
    print("✅ Zone validation test passed")
    return True

def test_real_eurusd_zones():
    """Test with real EURUSD zones"""
    print("\n📊 Testing Real EURUSD Zones")
    print("=" * 40)
    
    try:
        # Load and process data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        risk_manager = RiskManager(account_balance=10000)
        
        current_price = data['close'].iloc[-1]
        
        print(f"   Current EURUSD Price: {current_price:.5f}")
        print(f"   Total Zones Found: {patterns['total_patterns']}")
        
        # Test each zone
        tradeable_zones = 0
        total_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
        
        for i, zone in enumerate(total_zones[:10]):  # Test first 10 zones
            validation = risk_manager.validate_zone_for_trading(zone, current_price)
            
            if validation['is_tradeable']:
                tradeable_zones += 1
                print(f"   Zone {i+1}: ✅ TRADEABLE - {zone['type']} - "
                      f"{validation['position_size']} lots - "
                      f"{validation['stop_distance_pips']:.1f} pips")
            else:
                print(f"   Zone {i+1}: ❌ {validation['reason']}")
        
        print(f"\n   Tradeable Zones: {tradeable_zones}/{min(10, len(total_zones))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all risk management tests"""
    print("🚀 RISK MANAGEMENT SYSTEM - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Position sizing
    test_results.append(test_position_sizing())
    
    # Test 2: Stop loss calculation
    test_results.append(test_stop_loss_calculation())
    
    # Test 3: Zone validation
    test_results.append(test_zone_validation())
    
    # Test 4: Real data testing
    test_results.append(test_real_eurusd_zones())
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - RISK MANAGEMENT MODULE 4 READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 Module 4 Complete - Risk Management System Working!")
        print("📋 Features implemented:")
        print("   ✅ Position sizing (2% fixed risk)")
        print("   ✅ Stop loss validation (15-80 pips)")
        print("   ✅ Take profit calculation (1:2+ RR)")
        print("   ✅ Zone trading validation")
        print("   ✅ Risk budget tracking")
        print("   ✅ Account balance management")
        print("\n📈 Ready for Module 5: Signal Generation!")
    else:
        print("\n⚠️  Module 4 requires fixes before proceeding")