"""
Clean Zone Visualization Tool - Module 2 Debug (WORKING VERSION)
Clear visualization with non-overlapping labels and visible candles
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from config.settings import ZONE_CONFIG

def plot_candlesticks(ax, data, start_idx=0):
    """Plot clean candlestick chart"""
    
    for i, (idx, candle) in enumerate(data.iterrows()):
        x = start_idx + i
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        # Determine candle color
        if close_price >= open_price:
            color = 'green'
            alpha = 0.7
            body_bottom = open_price
            body_top = close_price
        else:
            color = 'red'
            alpha = 0.7
            body_bottom = close_price
            body_top = open_price
        
        # Draw the wick (high-low line)
        ax.plot([x, x], [low_price, high_price], color='black', linewidth=1.5, alpha=0.8)
        
        # Draw the body (rectangle)
        body_height = abs(close_price - open_price)
        if body_height > 0:
            rect = Rectangle((x - 0.35, body_bottom), 0.7, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.8, alpha=alpha)
            ax.add_patch(rect)
        else:
            # Doji candle (open = close)
            ax.plot([x - 0.35, x + 0.35], [open_price, open_price], color='black', linewidth=2)

def visualize_zones_simple():
    """Simple zone visualization to test if basic functionality works"""
    
    print("🕯️  SIMPLE ZONE VISUALIZATION - TESTING")
    print("=" * 50)
    
    try:
        # Load data
        print("📊 Loading EURUSD data...")
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Weekly')
        
        # Initialize components
        candle_classifier = CandleClassifier(data)
        zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
        
        print("✅ Components initialized successfully")
        
        # Use smaller sample for testing
        sample_size = 100
        sample_data = data.tail(sample_size).copy()
        sample_data.reset_index(drop=True, inplace=True)
        
        print(f"🔍 Analyzing {len(sample_data)} candles...")
        
        # Try to detect patterns
        patterns = zone_detector.detect_all_patterns(sample_data)
        
        total_patterns = patterns['total_patterns']
        print(f"📈 Found {total_patterns} patterns")
        
        # Create basic plot
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Plot candlesticks
        plot_candlesticks(ax, sample_data)
        
        # Plot zones with minimal labels
        pattern_count = 0
        
        # Plot D-B-D zones
        for i, pattern in enumerate(patterns['dbd_patterns']):
            pattern_count += 1
            
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            zone_high = pattern['zone_high']
            zone_low = pattern['zone_low']
            strength = pattern['strength']
            
            # Simple zone rectangle
            width = end_idx - start_idx + 1
            rect = Rectangle((start_idx - 0.5, zone_low), width, zone_high - zone_low, 
                            facecolor='red', alpha=0.15, edgecolor='darkred', linewidth=2)
            ax.add_patch(rect)
            
            # Simple label
            label_x = start_idx + width/2
            label_y = zone_high + (sample_data['high'].max() - sample_data['low'].min()) * 0.02
            
            ax.text(label_x, label_y, f'D-B-D #{i+1}\nS:{strength:.2f}', 
                   ha='center', va='bottom', fontsize=8, color='darkred', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot R-B-R zones
        for i, pattern in enumerate(patterns['rbr_patterns']):
            pattern_count += 1
            
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            zone_high = pattern['zone_high']
            zone_low = pattern['zone_low']
            strength = pattern['strength']
            
            # Simple zone rectangle
            width = end_idx - start_idx + 1
            rect = Rectangle((start_idx - 0.5, zone_low), width, zone_high - zone_low, 
                            facecolor='blue', alpha=0.15, edgecolor='darkblue', linewidth=2)
            ax.add_patch(rect)
            
            # Simple label
            label_x = start_idx + width/2
            label_y = zone_low - (sample_data['high'].max() - sample_data['low'].min()) * 0.02
            
            ax.text(label_x, label_y, f'R-B-R #{i+1}\nS:{strength:.2f}', 
                   ha='center', va='top', fontsize=8, color='darkblue', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Basic formatting
        ax.set_title(f'EURUSD Zones - {total_patterns} patterns detected', fontsize=14, weight='bold')
        ax.set_xlabel('Candle Index', fontsize=12)
        ax.set_ylabel('Price Level', fontsize=12)
        
        # Simple X-axis
        major_ticks = list(range(0, len(sample_data), 10))
        ax.set_xticks(major_ticks)
        ax.set_xticklabels([f'#{x}' for x in major_ticks])
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/zone_test.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Successfully plotted {total_patterns} patterns")
        
        # Print basic pattern info
        print(f"\n📋 PATTERN SUMMARY:")
        print(f"   D-B-D patterns: {len(patterns['dbd_patterns'])}")
        print(f"   R-B-R patterns: {len(patterns['rbr_patterns'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_zone_detector():
    """Debug zone detector step by step"""
    
    print("\n🔍 DEBUGGING ZONE DETECTOR...")
    print("=" * 50)
    
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        sample_data = data.tail(50).copy()  # Very small sample
        sample_data.reset_index(drop=True, inplace=True)
        
        print(f"✅ Loaded {len(sample_data)} candles")
        
        # Test candle classifier
        candle_classifier = CandleClassifier(sample_data)
        classified_data = candle_classifier.classify_all_candles()
        
        print(f"✅ Candle classification complete")
        print(f"   Base: {(classified_data['candle_type'] == 'base').sum()}")
        print(f"   Decisive: {(classified_data['candle_type'] == 'decisive').sum()}")
        print(f"   Explosive: {(classified_data['candle_type'] == 'explosive').sum()}")
        
        # Test zone detector
        zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
        print(f"✅ Zone detector initialized")
        
        # Try pattern detection
        patterns = zone_detector.detect_all_patterns(sample_data)
        print(f"✅ Pattern detection complete: {patterns['total_patterns']} patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in zone detector: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_specific_candles(data, candle_indices):
    """Debug specific candles mentioned in validation"""
    
    classifier = CandleClassifier(data)
    
    print(f"\n🔍 DEBUGGING SPECIFIC CANDLES: {candle_indices}")
    print("=" * 60)
    
    for idx in candle_indices:
        if idx < len(data):
            candle = data.iloc[idx]
            
            # Manual calculation
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            ratio = body_size / total_range if total_range > 0 else 0
            
            # Classifier result
            classification = classifier.classify_single_candle(
                candle['open'], candle['high'], candle['low'], candle['close']
            )
            
            # Determine expected classification
            if ratio <= 0.50:
                expected = 'base'
            elif ratio > 0.80:
                expected = 'explosive'
            else:
                expected = 'decisive'
            
            direction = 'Bullish' if candle['close'] > candle['open'] else 'Bearish'
            
            print(f"Candle #{idx}:")
            print(f"  OHLC: {candle['open']:.5f}, {candle['high']:.5f}, {candle['low']:.5f}, {candle['close']:.5f}")
            print(f"  Body: {body_size:.5f} | Range: {total_range:.5f}")
            print(f"  Body/Range Ratio: {ratio:.3f} ({ratio*100:.1f}%)")
            print(f"  Expected: {expected} | Got: {classification} {'✅' if expected == classification else '❌'}")
            print(f"  Direction: {direction}")
            print()

if __name__ == "__main__":
    print("🎯 ZONE VISUALIZATION - TESTING MODE")
    print("=" * 60)
    
    # Step 1: Test zone detector
    if debug_zone_detector():
        print("\n✅ Zone detector working, trying visualization...")
        
        # Step 2: Try simple visualization
        if visualize_zones_simple():
            print("\n✅ Basic visualization successful!")
            
            # Step 3: Debug specific candles
            print("\n🔍 DEBUGGING PROBLEMATIC CANDLES...")
            data_loader = DataLoader()
            data = data_loader.load_pair_data('EURUSD', 'Daily')
            sample_data = data.tail(150).copy()
            sample_data.reset_index(drop=True, inplace=True)
            
            problematic_candles = [73, 77, 78, 99, 110]
            debug_specific_candles(sample_data, problematic_candles)
            
        else:
            print("\n❌ Visualization failed")
    else:
        print("\n❌ Zone detector failed")
    
    print("\n📊 Check results/zone_test.png if successful")