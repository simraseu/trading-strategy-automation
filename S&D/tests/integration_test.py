# Quick integration test
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier

# Load data
data = DataLoader().load_pair_data('EURUSD', 'Daily')

# Full pipeline test
candle_classifier = CandleClassifier(data)
classified_data = candle_classifier.classify_all_candles()

zone_detector = ZoneDetector(candle_classifier)
patterns = zone_detector.detect_all_patterns(classified_data)

trend_classifier = TrendClassifier(data)
trend_data = trend_classifier.classify_trend_with_filter()

print(f"✅ Full pipeline working:")
print(f"   Candles: {len(classified_data)}")
print(f"   Patterns: {patterns['total_patterns']}")
print(f"   Current trend: {trend_data['trend_filtered'].iloc[-1]}")