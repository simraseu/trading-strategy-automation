"""
Zone Visualization Tool with Candlesticks
Visual validation of detected D-B-D and R-B-R patterns
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from config.settings import ZONE_CONFIG, CANDLE_THRESHOLDS

def plot_candlesticks(ax, data, start_idx=0):
    """Plot candlestick chart"""
    
    for i, (idx, candle) in enumerate(data.iterrows()):
        x = start_idx + i
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        # Determine candle color
        if close_price >= open_price:
            color = 'green'  # Bullish candle
            body_bottom = open_price
            body_top = close_price
        else:
            color = 'red'    # Bearish candle
            body_bottom = close_price
            body_top = open_price
        
        # Draw the wick (high-low line)
        ax.plot([x, x], [low_price, high_price], color='black', linewidth=1)
        
        # Draw the body (rectangle)
        body_height = abs(close_price - open_price)
        if body_height > 0:  # Avoid zero-height rectangles
            rect = Rectangle((x - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        else:
            # Doji candle (open = close)
            ax.plot([x - 0.3, x + 0.3], [open_price, open_price], color='black', linewidth=2)

def visualize_zones_candlesticks():
    """Visualize detected zones on candlestick chart"""
    
    # Load data
    print("📊 Loading EURUSD data...")
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    
    # Initialize components
    candle_classifier = CandleClassifier(data)
    zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
    
    # Use a smaller sample for visualization (last 100 candles for clarity)
    sample_data = data.tail(100).copy()
    sample_data.reset_index(drop=True, inplace=True)
    
    print(f"🔍 Analyzing {len(sample_data)} candles for visualization...")
    
    # Detect patterns
    patterns = zone_detector.detect_all_patterns(sample_data)
    
    print(f"📈 Found {patterns['total_patterns']} patterns to visualize")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot candlesticks
    plot_candlesticks(ax, sample_data)
    
    # Plot D-B-D zones (red/bearish)
    for i, pattern in enumerate(patterns['dbd_patterns']):
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        strength = pattern['strength']
        
        # Draw zone rectangle
        width = end_idx - start_idx + 1
        rect = Rectangle((start_idx - 0.5, zone_low), width, zone_high - zone_low, 
                        facecolor='red', alpha=0.2, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        
        # Add strength label
        ax.text(start_idx + width/2, zone_high + (sample_data['high'].max() - sample_data['low'].min()) * 0.01, 
               f'D-B-D\n{strength:.3f}', 
               ha='center', va='bottom', fontsize=9, color='darkred', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw zone boundary lines
        ax.plot([start_idx - 0.5, end_idx + 0.5], [zone_high, zone_high], 
               color='red', linewidth=2, linestyle='-', alpha=0.8)
        ax.plot([start_idx - 0.5, end_idx + 0.5], [zone_low, zone_low], 
               color='red', linewidth=2, linestyle='-', alpha=0.8)
    
    # Plot R-B-R zones (blue/bullish)
    for i, pattern in enumerate(patterns['rbr_patterns']):
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        strength = pattern['strength']
        
        # Draw zone rectangle
        width = end_idx - start_idx + 1
        rect = Rectangle((start_idx - 0.5, zone_low), width, zone_high - zone_low, 
                        facecolor='blue', alpha=0.2, edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        
        # Add strength label
        ax.text(start_idx + width/2, zone_low - (sample_data['high'].max() - sample_data['low'].min()) * 0.015, 
               f'R-B-R\n{strength:.3f}', 
               ha='center', va='top', fontsize=9, color='darkblue', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw zone boundary lines
        ax.plot([start_idx - 0.5, end_idx + 0.5], [zone_high, zone_high], 
               color='blue', linewidth=2, linestyle='-', alpha=0.8)
        ax.plot([start_idx - 0.5, end_idx + 0.5], [zone_low, zone_low], 
               color='blue', linewidth=2, linestyle='-', alpha=0.8)
    
    # Formatting with enhanced X-axis labeling
    ax.set_title(f'EURUSD Supply & Demand Zones - Candlestick Analysis\n{patterns["total_patterns"]} patterns detected', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Candle Index (Most Recent 100 Daily Candles)', fontsize=12)
    ax.set_ylabel('Price Level', fontsize=12)

    # Set major X-axis ticks every 5 candles with clear labels
    major_ticks = list(range(0, len(sample_data), 5))
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f'#{x}' for x in major_ticks], fontsize=10)

    # Add minor ticks every candle
    minor_ticks = list(range(0, len(sample_data), 1))
    ax.set_xticks(minor_ticks, minor=True)

    # Enhanced grid system
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)  # Major grid every 5
    ax.grid(True, alpha=0.15, linestyle='-', which='minor', linewidth=0.3)  # Minor grid every candle

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='Bullish Candle'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black', label='Bearish Candle'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.3, edgecolor='red', linestyle='--', label='D-B-D Zone'),
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.3, edgecolor='blue', linestyle='--', label='R-B-R Zone')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add statistics box
    summary = zone_detector.get_pattern_summary(patterns)
    stats_text = f"""Zone Detection Statistics:
D-B-D Zones: {len(patterns['dbd_patterns'])}
R-B-R Zones: {len(patterns['rbr_patterns'])}
Average Strength: {summary['avg_strength']:.3f}
High Quality (≥0.8): {summary['strength_distribution']['high']}
Medium Quality (0.5-0.8): {summary['strength_distribution']['medium']}
Pattern Density: {patterns['total_patterns']/len(sample_data)*100:.1f}%"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the chart
    plt.savefig('results/zone_detection_candlesticks.png', dpi=300, bbox_inches='tight')
    print("💾 Chart saved to results/zone_detection_candlesticks.png")
    
    plt.show()
    
    # Print detailed pattern analysis
    print_pattern_details(patterns, sample_data)

def print_pattern_details(patterns, sample_data):
    """Print detailed pattern information"""
    
    print("\n" + "="*70)
    print("DETAILED CANDLESTICK PATTERN ANALYSIS")
    print("="*70)
    
    if patterns['dbd_patterns']:
        print("\n📉 DROP-BASE-DROP (SUPPLY) ZONES:")
        print("-" * 50)
        for i, pattern in enumerate(patterns['dbd_patterns'], 1):
            range_pips = pattern['zone_range'] * 10000
            print(f"  🔴 D-B-D Zone #{i}:")
            print(f"     📍 Location: Candles {pattern['start_idx']} to {pattern['end_idx']}")
            print(f"     📊 Price Range: {pattern['zone_low']:.5f} - {pattern['zone_high']:.5f}")
            print(f"     📏 Zone Size: {range_pips:.1f} pips")
            print(f"     ⭐ Strength Score: {pattern['strength']:.3f}")
            print(f"     🎯 Base Candles: {pattern['base']['candle_count']}")
            print(f"     🚀 Leg-out Ratio: {pattern['leg_out']['ratio_to_base']:.2f}x base size")
            print()
    
    if patterns['rbr_patterns']:
        print("\n📈 RALLY-BASE-RALLY (DEMAND) ZONES:")
        print("-" * 50)
        for i, pattern in enumerate(patterns['rbr_patterns'], 1):
            range_pips = pattern['zone_range'] * 10000
            print(f"  🔵 R-B-R Zone #{i}:")
            print(f"     📍 Location: Candles {pattern['start_idx']} to {pattern['end_idx']}")
            print(f"     📊 Price Range: {pattern['zone_low']:.5f} - {pattern['zone_high']:.5f}")
            print(f"     📏 Zone Size: {range_pips:.1f} pips")
            print(f"     ⭐ Strength Score: {pattern['strength']:.3f}")
            print(f"     🎯 Base Candles: {pattern['base']['candle_count']}")
            print(f"     🚀 Leg-out Ratio: {pattern['leg_out']['ratio_to_base']:.2f}x base size")
            print()

def debug_specific_obvious_pattern():
    """Debug the obvious pattern at candles 38-41"""
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_pair_data('EURUSD', 'Daily')
    candle_classifier = CandleClassifier(data)
    zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
    
    sample_data = data.tail(100).copy()
    sample_data.reset_index(drop=True, inplace=True)
    
    print("🔍 DEBUGGING OBVIOUS PATTERN: Candles 38-41")
    print("="*60)
    
    # Show the specific candles
    pattern_candles = sample_data.iloc[38:42]
    print("\n📊 CANDLE DATA:")
    for i, (idx, candle) in enumerate(pattern_candles.iterrows()):
        candle_idx = 38 + i
        direction = "🟢 UP" if candle['close'] > candle['open'] else "🔴 DOWN"
        classification = candle_classifier.classify_single_candle(
            candle['open'], candle['high'], candle['low'], candle['close']
        )
        print(f"   Candle {candle_idx}: {direction} | {classification} | O={candle['open']:.5f} C={candle['close']:.5f}")
    
    # Test each component manually
    print(f"\n🔵 TESTING LEG-IN (Candle 38):")
    leg_in_data = sample_data.iloc[38:39]  # Just candle 38
    leg_in_result = zone_detector.is_valid_leg(leg_in_data, 'bullish')
    print(f"   Single candle leg-in valid: {leg_in_result}")
    
    # Test leg-in detection function
    leg_in = zone_detector.identify_leg_in(sample_data, 38, 'bullish')
    print(f"   identify_leg_in result: {leg_in}")

    print(f"\n🧪 TESTING FIXED LEG-IN:")
    # Test just candle 38 as leg-in
    single_leg = sample_data.iloc[38:39]
    print(f"   Candle 38 alone - Range: {single_leg['high'].max() - single_leg['low'].min():.5f}")
    print(f"   Candle 38 alone - Net movement: {single_leg['close'].iloc[0] - single_leg['open'].iloc[0]:.5f}")

    # Check if candle 39 looks like base relative to 38
    candle_38_high = sample_data.iloc[38]['high']
    candle_38_low = sample_data.iloc[38]['low']
    candle_39 = sample_data.iloc[39]
    print(f"   Candle 39 within 38's range: {candle_39['low'] >= candle_38_low * 0.98 and candle_39['high'] <= candle_38_high * 1.02}")
    

if __name__ == "__main__":
    print("🕯️  CANDLESTICK ZONE VISUALIZATION TOOL")
    print("="*50)
    visualize_zones_candlesticks()
    
    # Add debug for missed pattern
    print("\n" + "🚨"*20)
    debug_specific_obvious_pattern()
    
    print("\n✅ Visualization complete!")