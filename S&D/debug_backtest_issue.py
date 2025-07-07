"""
Debug Backtest Signal Repetition Issue
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
from modules.backtester import TradingBacktester

def debug_signal_repetition():
    """Debug why same signals are being generated repeatedly"""
    print("🔍 DEBUGGING SIGNAL REPETITION ISSUE")
    print("=" * 50)
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    # Test 3 consecutive weeks
    test_dates = ['2025-06-13', '2025-06-20', '2025-06-27']
    
    for test_date in test_dates:
        print(f"\n📅 Testing date: {test_date}")
        
        # Get historical window (as backtester would)
        date_idx = data.index.get_loc(test_date)
        history_start = max(0, date_idx - 365)
        historical_data = data.iloc[history_start:date_idx]
        
        print(f"   Historical data: {len(historical_data)} candles")
        print(f"   Last historical date: {historical_data.index[-1]}")
        print(f"   Last historical close: {historical_data['close'].iloc[-1]:.5f}")
        
        # Initialize fresh components (as fix would do)
        candle_classifier = CandleClassifier(historical_data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(historical_data)
        risk_manager = RiskManager(account_balance=10000)
        signal_generator = SignalGenerator(zone_detector, trend_classifier, risk_manager)
        
        # Generate signals
        signals = signal_generator.generate_signals(classified_data, 'Daily', 'EURUSD')
        
        print(f"   Signals generated: {len(signals)}")
        for i, signal in enumerate(signals):
            print(f"      Signal {i+1}: {signal['direction']} at {signal['entry_price']:.5f}")
            print(f"         Zone: {signal['zone_low']:.5f} - {signal['zone_high']:.5f}")

if __name__ == "__main__":
    debug_signal_repetition()