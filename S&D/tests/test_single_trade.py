"""
Test core trade simulation logic - isolate the infinite loop
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import time
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager

def test_single_trade_simulation():
    """Test trade simulation logic without parallel processing"""
    
    print("🧪 Testing single trade simulation...")
    start_time = time.time()
    
    # Load data (we know this works)
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', '3D')
    
    # Initialize components (we know this works)
    candle_classifier = CandleClassifier(data)
    classified_data = candle_classifier.classify_all_candles()
    
    zone_detector = ZoneDetector(candle_classifier)
    patterns = zone_detector.detect_all_patterns(classified_data)
    
    trend_classifier = TrendClassifier(data)
    trend_data = trend_classifier.classify_trend_simplified()
    
    risk_manager = RiskManager(account_balance=10000)
    
    # Get one pattern to test
    momentum_patterns = patterns['dbd_patterns'] + patterns['rbr_patterns']
    if not momentum_patterns:
        print("❌ No patterns found for testing")
        return
    
    test_pattern = momentum_patterns[0]
    print(f"✅ Testing pattern: {test_pattern['type']}")
    
    # Test trade execution logic
    zone_high = test_pattern['zone_high']
    zone_low = test_pattern['zone_low']
    zone_range = zone_high - zone_low
    
    # Entry and stop logic
    if test_pattern['type'] in ['R-B-R', 'D-B-R']:
        entry_price = zone_low + (zone_range * 0.05)
        direction = 'BUY'
        initial_stop = zone_low - (zone_range * 0.33)
    else:
        entry_price = zone_high - (zone_range * 0.05)
        direction = 'SELL'
        initial_stop = zone_high + (zone_range * 0.33)
    
    stop_distance = abs(entry_price - initial_stop)
    risk_amount = 500
    position_size = risk_amount / stop_distance
    
    print(f"✅ Trade setup: {direction} at {entry_price:.5f}, Stop: {initial_stop:.5f}")
    
    # Find entry point
    zone_end_idx = test_pattern.get('end_idx', 50)
    entry_idx = None
    
    # CRITICAL TEST: This loop might be the infinite loop source
    search_limit = min(50, len(data) - zone_end_idx - 1)  # Limit search
    print(f"🔍 Searching for entry over {search_limit} candles...")
    
    for i in range(zone_end_idx + 1, zone_end_idx + 1 + search_limit):
        if i >= len(data):
            break
            
        candle = data.iloc[i]
        
        if direction == 'BUY' and candle['low'] <= entry_price:
            entry_idx = i
            print(f"✅ Entry found at index {entry_idx}")
            break
        elif direction == 'SELL' and candle['high'] >= entry_price:
            entry_idx = i
            print(f"✅ Entry found at index {entry_idx}")
            break
    
    if entry_idx is None:
        print("⚠️  No entry found in limited search")
        return
    
    # Test trade simulation (this is likely where infinite loop occurs)
    print("🔍 Testing trade simulation logic...")
    
    current_stop = initial_stop
    risk_distance = abs(entry_price - initial_stop)
    
    # CRITICAL TEST: Limit simulation length to avoid infinite loop
    max_simulation_length = min(100, len(data) - entry_idx - 1)
    print(f"🔍 Simulating trade over {max_simulation_length} candles...")
    
    simulation_start = time.time()
    
    for i in range(entry_idx + 1, entry_idx + 1 + max_simulation_length):
        if i >= len(data):
            break
            
        candle = data.iloc[i]
        days_held = i - entry_idx
        
        # Check stop loss (simplified)
        if direction == 'BUY' and candle['low'] <= current_stop:
            pnl = (current_stop - entry_price) * position_size
            simulation_time = time.time() - simulation_start
            print(f"✅ Trade stopped out after {days_held} days")
            print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
            break
        elif direction == 'SELL' and candle['high'] >= current_stop:
            pnl = (entry_price - current_stop) * position_size
            simulation_time = time.time() - simulation_start
            print(f"✅ Trade stopped out after {days_held} days")
            print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
            break
        
        # Check for 2R target
        if direction == 'BUY':
            target_price = entry_price + (risk_distance * 2)
            if candle['high'] >= target_price:
                pnl = (target_price - entry_price) * position_size
                simulation_time = time.time() - simulation_start
                print(f"✅ Trade hit target after {days_held} days")
                print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
                break
        else:
            target_price = entry_price - (risk_distance * 2)
            if candle['low'] <= target_price:
                pnl = (entry_price - target_price) * position_size
                simulation_time = time.time() - simulation_start
                print(f"✅ Trade hit target after {days_held} days")
                print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
                break
    else:
        # Loop completed without break
        simulation_time = time.time() - simulation_start
        print(f"✅ Simulation completed normally in {simulation_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n✅ Total test completed in {total_time:.2f} seconds")
    
    if total_time > 10:
        print("⚠️  Test took longer than expected - potential performance issue")
    else:
        print("✅ Test performance normal - issue is likely in parallel processing")

if __name__ == "__main__":
    test_single_trade_simulation()