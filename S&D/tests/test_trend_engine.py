"""
Quick Test Suite for TrendEngine
Verify all functionality works as expected
"""

import pandas as pd
import numpy as np
import sys
import os

# Add path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.trend_classifier import TrendEngine

def create_test_data(n_candles=100):
    """Create simple test OHLC data"""
    dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='D')
    
    # Create trending price data
    base_price = 1.1000
    trend = np.cumsum(np.random.randn(n_candles) * 0.001)
    noise = np.random.randn(n_candles) * 0.0005
    
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(n_candles) * 0.0002,
        'high': close_prices + abs(np.random.randn(n_candles)) * 0.0003,
        'low': close_prices - abs(np.random.randn(n_candles)) * 0.0003,
        'close': close_prices
    }, index=dates)
    
    # Ensure OHLC logic is valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_default_mode():
    """Test 1: Default Mode (EMA50/200)"""
    print("🧪 TEST 1: Default Mode (EMA50/200)")
    print("=" * 50)
    
    data = create_test_data(200)  # Need enough data for EMA200
    
    # Test default initialization
    trend_engine = TrendEngine(data)
    
    # Check default config
    assert trend_engine.config['fast_ema'] == 50, "Default fast EMA should be 50"
    assert trend_engine.config['slow_ema'] == 200, "Default slow EMA should be 200"
    assert trend_engine.mode == 'default', "Should be in default mode"
    
    # Test main method (required by signal generator)
    trend_data = trend_engine.classify_trend_with_filter()
    
    # Check required columns exist
    required_columns = ['ema_50', 'ema_200', 'trend_filtered', 'ema_separation']
    for col in required_columns:
        assert col in trend_data.columns, f"Missing required column: {col}"
    
    # Check trends are valid
    unique_trends = trend_data['trend_filtered'].unique()
    assert set(unique_trends).issubset({'bullish', 'bearish'}), "Invalid trend values"
    
    # Check EMA separation calculation
    manual_separation = trend_data['ema_50'] - trend_data['ema_200']
    np.testing.assert_array_almost_equal(
        trend_data['ema_separation'].values, 
        manual_separation.values, 
        decimal=10
    )
    
    print("✅ Default mode test PASSED")
    return trend_engine, trend_data

def test_research_mode():
    """Test 2: Research Mode (Custom EMAs)"""
    print("\n🧪 TEST 2: Research Mode (Custom EMAs)")
    print("=" * 50)
    
    data = create_test_data(100)
    
    # Test custom configuration
    config = {'fast_ema': 21, 'slow_ema': 55, 'method': 'dual_ema'}
    trend_engine = TrendEngine(data, config)
    
    # Check custom config
    assert trend_engine.config['fast_ema'] == 21, "Custom fast EMA should be 21"
    assert trend_engine.config['slow_ema'] == 55, "Custom slow EMA should be 55"
    assert trend_engine.mode == 'research', "Should be in research mode"
    
    # Test trend classification
    trend_data = trend_engine.classify_trend_with_filter()
    
    # Check columns exist
    required_columns = ['ema_50', 'ema_200', 'trend_filtered', 'ema_separation']
    for col in required_columns:
        assert col in trend_data.columns, f"Missing required column: {col}"
    
    # Check dynamic columns exist
    assert 'ema_21' in trend_data.columns, "Missing dynamic EMA21 column"
    assert 'ema_55' in trend_data.columns, "Missing dynamic EMA55 column"
    
    # Verify ema_50 actually contains EMA21 data (compatibility mapping)
    np.testing.assert_array_almost_equal(
        trend_data['ema_50'].values, 
        trend_data['ema_21'].values, 
        decimal=10
    )
    
    print("✅ Research mode test PASSED")
    return trend_engine, trend_data

def test_signal_generator_compatibility():
    """Test 3: Signal Generator Compatibility"""
    print("\n🧪 TEST 3: Signal Generator Compatibility")
    print("=" * 50)
    
    data = create_test_data(200)
    trend_engine = TrendEngine(data)
    
    # Test method that signal generator calls
    trend_data = trend_engine.classify_trend_with_filter()
    
    # Simulate signal generator usage
    current_trend = trend_data['trend_filtered'].iloc[-1]
    trend_strength = trend_data['ema_separation'].iloc[-1]
    current_ema50 = trend_data['ema_50'].iloc[-1]
    current_ema200 = trend_data['ema_200'].iloc[-1]
    
    # Validate signal generator expectations
    assert current_trend in ['bullish', 'bearish'], "Invalid current trend"
    assert isinstance(trend_strength, (int, float)), "EMA separation should be numeric"
    assert isinstance(current_ema50, (int, float)), "EMA50 should be numeric"
    assert isinstance(current_ema200, (int, float)), "EMA200 should be numeric"
    
    # Test get_current_trend method
    current_status = trend_engine.get_current_trend()
    
    required_keys = ['trend', 'ema_fast', 'ema_slow', 'ema_separation', 'config']
    for key in required_keys:
        assert key in current_status, f"Missing key in current trend: {key}"
    
    print("✅ Signal generator compatibility test PASSED")
    return current_status

def test_research_methods():
    """Test 4: Research Methods for Future EMA Testing"""
    print("\n🧪 TEST 4: Research Methods")
    print("=" * 50)
    
    data = create_test_data(100)
    trend_engine = TrendEngine(data)
    
    # Test single EMA combination
    test_result = trend_engine.test_ema_combination(12, 26)
    
    assert 'ema_12' in test_result.columns, "Missing EMA12 column"
    assert 'ema_26' in test_result.columns, "Missing EMA26 column" 
    assert 'trend' in test_result.columns, "Missing trend column"
    assert 'ema_separation' in test_result.columns, "Missing separation column"
    
    # Test batch testing
    ema_combinations = [(12, 26), (21, 55), (50, 200)]
    batch_results = trend_engine.batch_test_emas(ema_combinations)
    
    assert len(batch_results) == 3, "Should have 3 batch results"
    for combo in ema_combinations:
        assert combo in batch_results, f"Missing batch result for {combo}"
    
    # Test optimization
    optimal_result = trend_engine.get_optimal_ema_periods(ema_combinations)
    
    assert 'optimal_combination' in optimal_result, "Missing optimal combination"
    assert 'metric_value' in optimal_result, "Missing metric value"
    
    print("✅ Research methods test PASSED")
    return batch_results

def test_performance():
    """Test 5: Performance Check"""
    print("\n🧪 TEST 5: Performance Check")
    print("=" * 50)
    
    import time
    
    # Test with larger dataset
    data = create_test_data(1000)
    
    # Time default mode
    start_time = time.time()
    trend_engine = TrendEngine(data)
    trend_data = trend_engine.classify_trend_with_filter()
    default_time = time.time() - start_time
    
    # Time research mode  
    start_time = time.time()
    config = {'fast_ema': 21, 'slow_ema': 89, 'method': 'dual_ema'}
    trend_engine_research = TrendEngine(data, config)
    trend_data_research = trend_engine_research.classify_trend_with_filter()
    research_time = time.time() - start_time
    
    # Time batch testing
    start_time = time.time()
    ema_combinations = [(10, 20), (20, 50), (50, 100), (21, 89)]
    batch_results = trend_engine.batch_test_emas(ema_combinations)
    batch_time = time.time() - start_time
    
    print(f"   Default mode: {default_time:.3f}s")
    print(f"   Research mode: {research_time:.3f}s") 
    print(f"   Batch testing (4 combos): {batch_time:.3f}s")
    
    # Performance checks
    assert default_time < 1.0, "Default mode too slow"
    assert research_time < 1.0, "Research mode too slow"
    assert batch_time < 2.0, "Batch testing too slow"
    
    print("✅ Performance test PASSED")

def run_all_tests():
    """Run complete test suite"""
    print("🚀 TREND ENGINE TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all tests
        trend_engine_default, trend_data_default = test_default_mode()
        trend_engine_research, trend_data_research = test_research_mode()
        current_status = test_signal_generator_compatibility()
        batch_results = test_research_methods()
        test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! TrendEngine is working perfectly!")
        print("=" * 60)
        
        # Show sample results
        print("\n📊 SAMPLE RESULTS:")
        print(f"Default Mode - Latest Trend: {current_status['trend']}")
        print(f"Default Mode - EMA Separation: {current_status['ema_separation']:.6f}")
        print(f"Default Mode - Config: {current_status['config']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ TrendEngine ready for backtesting!")
    else:
        print("\n❌ Please fix issues before proceeding")