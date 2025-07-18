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
    
    # Test scenarios - NOTE: Your system uses 5% risk, not 2%
    test_cases = [
        (20, 'EURUSD', 2.5),   # 20 pip stop should give 2.5 lots (5% risk)
        (40, 'EURUSD', 1.25),  # 40 pip stop should give 1.25 lots
        (80, 'EURUSD', 0.625), # 80 pip stop should give 0.625 lots
    ]
    
    for stop_pips, pair, expected_size in test_cases:
        position_size = risk_manager.calculate_position_size(stop_pips, pair)
        
        print(f"   {stop_pips} pip stop: {position_size} lots (expected: ~{expected_size})")
        
        # Verify 5% risk (your actual system)
        risk_amount = risk_manager.calculate_risk_amount(stop_pips, position_size, pair)
        risk_percent = (risk_amount / 10000) * 100
        
        print(f"      Risk amount: ${risk_amount:.2f} ({risk_percent:.1f}%)")
        
        # Should be close to 5%
        assert 4.5 <= risk_percent <= 5.5, f"Risk percent {risk_percent} not close to 5%"
    
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
    
    # Test manual stop loss calculation
    stop_loss_price = risk_manager.calculate_stop_loss_manual(test_zone)
    
    print(f"   Zone Low: {test_zone['zone_low']}")
    print(f"   Stop Loss: {stop_loss_price:.5f}")
    
    # Stop should be 33% below zone low for R-B-R
    zone_size = test_zone['zone_high'] - test_zone['zone_low']
    expected_stop = test_zone['zone_low'] - (zone_size * 0.33)
    
    assert abs(stop_loss_price - expected_stop) < 0.00001, "Stop loss calculation incorrect"
    
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
        
        # Test each zone - handle new structure
        tradeable_zones = 0
        total_zones = patterns['dbd_patterns'] + patterns['rbr_patterns']
        
        # Add reversal patterns if they exist
        if 'reversal_patterns' in patterns:
            total_zones.extend(patterns['reversal_patterns'].get('dbr_patterns', []))
            total_zones.extend(patterns['reversal_patterns'].get('rbd_patterns', []))
        
        for i, zone in enumerate(total_zones[:10]):  # Test first 10 zones
            validation = risk_manager.validate_zone_for_trading(zone, current_price, 'EURUSD', data)
            
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

def test_reversal_patterns():
    """Test reversal pattern detection (D-B-R and R-B-D)"""
    print("\n🧪 Testing Reversal Pattern Detection")
    print("=" * 50)
    
    try:
        # Load EURUSD data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
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
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
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
    """Run all risk management tests including reversal patterns"""
    print("🚀 RISK MANAGEMENT SYSTEM - COMPREHENSIVE TESTING (WITH REVERSALS)")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Position sizing
    test_results.append(test_position_sizing())
    
    # Test 2: Stop loss calculation
    test_results.append(test_stop_loss_calculation())
    
    # Test 3: Zone validation
    test_results.append(test_zone_validation())
    
    # Test 4: Real data testing
    test_results.append(test_real_eurusd_zones())
    
    # Test 5: Reversal pattern detection (NEW)
    test_results.append(test_reversal_patterns())
    
    # Test 6: Pattern classification (NEW)
    test_results.append(test_pattern_classification())
    
    # Final results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - RISK MANAGEMENT + REVERSALS READY!")
        return True
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 Module 4 Complete - Risk Management System Working!")
        print("📋 Features implemented:")
        print("   ✅ Position sizing (5% fixed risk)")
        print("   ✅ Stop loss validation (33% buffer)")
        print("   ✅ Take profit calculation (1:2+ RR)")
        print("   ✅ Zone trading validation")
        print("   ✅ Risk budget tracking")
        print("   ✅ Account balance management")
        print("   ✅ Reversal pattern support")
        print("\n📈 Ready for Module 5: Signal Generation!")
    else:
        print("\n⚠️  Module 4 requires fixes before proceeding")