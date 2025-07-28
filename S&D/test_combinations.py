from core_backtest_engine import CoreBacktestEngine

engine = CoreBacktestEngine()
combinations = engine.discover_valid_data_combinations()
print(f"Total combinations to test: {len(combinations)}")