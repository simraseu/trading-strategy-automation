"""
Signal Visualization Tool - CORRECTED VERSION
Shows generated signals with proper zone mapping
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from modules.trend_classifier import TrendClassifier
from modules.risk_manager import RiskManager
from modules.signal_generator import SignalGenerator

plt.style.use('default')

class SignalVisualizer:
    def __init__(self):
        self.colors = {
            'buy_signal': '#2ECC71',
            'sell_signal': '#E74C3C',
            'entry_level': '#3498DB',
            'stop_loss': '#E74C3C',
            'take_profit': '#27AE60',
            'demand_zone': '#4ECDC4',
            'supply_zone': '#FF6B6B',
            'bullish_candle': '#2ECC71',
            'bearish_candle': '#E74C3C',
        }
    
    def plot_candlesticks(self, ax, data):
        """Plot candlesticks for recent data"""
        for i, (idx, candle) in enumerate(data.iterrows()):
            x = i  # Use simple index
            open_price = candle['open']
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']
            
            # Candle color
            if close_price > open_price:
                color = self.colors['bullish_candle']
                edge_color = '#27AE60'
                body_bottom = open_price
                body_top = close_price
            else:
                color = self.colors['bearish_candle']
                edge_color = '#C0392B'
                body_bottom = close_price
                body_top = open_price
            
            # Draw wick
            ax.plot([x, x], [low_price, high_price], 
                   color=edge_color, linewidth=1, alpha=0.8)
            
            # Draw body
            body_height = abs(close_price - open_price)
            if body_height > 0:
                rect = patches.Rectangle((x - 0.3, body_bottom), 0.6, body_height, 
                                       facecolor=color, edgecolor=edge_color, 
                                       linewidth=0.5, alpha=0.7)
                ax.add_patch(rect)
            else:
                # Doji
                ax.plot([x - 0.3, x + 0.3], [open_price, open_price], 
                       color=edge_color, linewidth=2)

    def create_simple_signal_chart(self, signals, data):
        """Create a simple chart showing all signals"""
        
        if not signals:
            print("❌ No signals to visualize")
            return
        
        # Use last X candles for context
        recent_data = data.tail(5000).copy()
        recent_data = recent_data.reset_index(drop=True)
        
        # Create single chart with all signals
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        
        # Plot candlesticks
        self.plot_candlesticks(ax, recent_data)
        
        # Plot each signal
        for i, signal in enumerate(signals):
            self.plot_signal_on_chart(ax, signal, len(recent_data), i)
        
        # Formatting
        ax.set_title(f'EURUSD Daily - {len(signals)} Generated Signals', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save chart
        import os
        os.makedirs('results/charts', exist_ok=True)
        chart_filename = f"results/charts/signals_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Signal chart saved to: {chart_filename}")
        
        plt.show()

    def plot_signal_on_chart(self, ax, signal, data_length, signal_num):
        """Plot a single signal on the chart"""
        
        # Signal details
        direction = signal['direction']
        entry = signal['entry_price']
        stop = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal['take_profit_2']
        zone_high = signal['zone_high']
        zone_low = signal['zone_low']
        
        # Line positions (show on recent data)
        line_start = 0
        line_end = data_length
        
        # Signal color
        signal_color = self.colors['buy_signal'] if direction == 'BUY' else self.colors['sell_signal']
        
        # Plot zone rectangle (across entire visible area)
        zone_color = self.colors['demand_zone'] if direction == 'BUY' else self.colors['supply_zone']
        rect = patches.Rectangle((line_start, zone_low), 
                               line_end - line_start, zone_high - zone_low,
                               facecolor=zone_color, alpha=0.15, 
                               edgecolor=zone_color, linewidth=1,
                               label=f'Signal {signal_num+1} Zone')
        ax.add_patch(rect)
        
        # Plot trading levels
        ax.hlines(entry, line_start, line_end, 
                 colors=signal_color, linewidth=2, linestyle='-', alpha=0.8,
                 label=f'S{signal_num+1} Entry: {entry:.5f}')
        
        ax.hlines(stop, line_start, line_end, 
                 colors='red', linewidth=1.5, linestyle='--', alpha=0.7,
                 label=f'S{signal_num+1} Stop: {stop:.5f}')
        
        ax.hlines(tp1, line_start, line_end, 
                 colors='green', linewidth=1, linestyle=':', alpha=0.6,
                 label=f'S{signal_num+1} TP1: {tp1:.5f}')
        
        ax.hlines(tp2, line_start, line_end, 
                 colors='darkgreen', linewidth=1.5, linestyle=':', alpha=0.8,
                 label=f'S{signal_num+1} TP2: {tp2:.5f}')
        
        # Add signal annotation
        arrow_x = data_length - 10
        if direction == 'BUY':
            ax.annotate(f'BUY {signal_num+1}', xy=(arrow_x, entry), 
                       xytext=(arrow_x, entry - (entry * 0.001)),
                       arrowprops=dict(arrowstyle='->', color=signal_color, lw=2),
                       fontsize=10, fontweight='bold', color=signal_color)
        else:
            ax.annotate(f'SELL {signal_num+1}', xy=(arrow_x, entry), 
                       xytext=(arrow_x, entry + (entry * 0.001)),
                       arrowprops=dict(arrowstyle='->', color=signal_color, lw=2),
                       fontsize=10, fontweight='bold', color=signal_color)

def debug_signals_simple():
    """Simple signal visualization for debugging"""
    print("🔧 DEBUG SIGNAL VISUALIZATION")
    print("=" * 40)
    
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        print(f"📊 Data loaded: {len(data)} candles")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        # Initialize components
        candle_classifier = CandleClassifier(data)
        classified_data = candle_classifier.classify_all_candles()
        
        zone_detector = ZoneDetector(candle_classifier)
        trend_classifier = TrendClassifier(data)
        risk_manager = RiskManager(account_balance=10000)
        
        signal_generator = SignalGenerator(
            zone_detector, trend_classifier, risk_manager
        )
        
        # Generate signals
        signals = signal_generator.generate_signals(classified_data, 'Daily', 'EURUSD')
        
        if signals:
            print(f"✅ Generated {len(signals)} signals")
            
            # Print signal details
            for i, signal in enumerate(signals):
                print(f"\nSignal {i+1}:")
                print(f"  Direction: {signal['direction']}")
                print(f"  Entry: {signal['entry_price']:.5f}")
                print(f"  Stop: {signal['stop_loss']:.5f}")
                print(f"  TP1: {signal['take_profit_1']:.5f}")
                print(f"  TP2: {signal['take_profit_2']:.5f}")
                print(f"  Zone: {signal['zone_low']:.5f} - {signal['zone_high']:.5f}")
                print(f"  Risk: ${signal['risk_amount']:.0f} ({signal['position_size']} lots)")
                print(f"  Score: {signal['signal_score']:.1f}")
            
            # Create visualization
            visualizer = SignalVisualizer()
            visualizer.create_simple_signal_chart(signals, data)
            
            # Export with basic info
            signal_generator.export_signals_for_backtesting()
            
        else:
            print("❌ No signals generated")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_signals_simple()