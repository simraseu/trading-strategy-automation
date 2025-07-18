"""
Recent Pattern Formation Visualizer - Last 2 months
Shows all momentum and reversal patterns for validation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime, timedelta
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class RecentPatternVisualizer:
    def __init__(self):
        self.colors = {
            'bullish_candle': '#2ECC71',
            'bearish_candle': '#E74C3C',
            'doji_candle': '#95A5A6',
            
            # Momentum patterns
            'rbr_zone': '#4ECDC4',      # Teal for R-B-R
            'dbd_zone': '#FF6B6B',      # Red for D-B-D
            
            # Reversal patterns  
            'dbr_zone': '#9B59B6',      # Purple for D-B-R
            'rbd_zone': '#F39C12',      # Orange for R-B-D
            
            'base_marker': '#34495E'
        }
    
    def visualize_last_2_months(self):
        """Show all patterns from last 2 months"""
        print("📊 RECENT PATTERN FORMATION VISUALIZER")
        print("=" * 300)
        
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Get last 2 months (300 trading days)
        recent_data = data.tail(300).copy()
        recent_data = recent_data.reset_index()
        
        print(f"📅 Period: {recent_data['date'].iloc[0]} to {recent_data['date'].iloc[-1]}")
        print(f"📈 Candles: {len(recent_data)}")
        
        # Detect all patterns
        candle_classifier = CandleClassifier(recent_data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Display pattern counts
        print(f"\n🎯 PATTERNS FOUND:")
        print(f"   D-B-D (momentum): {len(patterns.get('dbd_patterns', []))}")
        print(f"   R-B-R (momentum): {len(patterns.get('rbr_patterns', []))}")
        print(f"   D-B-R (reversal): {len(patterns.get('dbr_patterns', []))}")
        print(f"   R-B-D (reversal): {len(patterns.get('rbd_patterns', []))}")
        print(f"   TOTAL: {patterns['total_patterns']}")
        
        # Create visualization
        if patterns['total_patterns'] > 0:
            self.create_pattern_chart(recent_data, patterns)
        else:
            print("❌ No patterns found in recent period")
    
    def create_pattern_chart(self, data, patterns):
        """Create comprehensive pattern chart"""
        
        fig, ax = plt.subplots(figsize=(24, 14))
        
        # Plot candlesticks
        self.plot_candlesticks(ax, data)
        
        # Plot all pattern types
        pattern_counter = 0
        
        # Momentum patterns
        for pattern in patterns.get('dbd_patterns', []):
            pattern_counter += 1
            self.plot_single_pattern(ax, pattern, data, pattern_counter, 'momentum')
        
        for pattern in patterns.get('rbr_patterns', []):
            pattern_counter += 1
            self.plot_single_pattern(ax, pattern, data, pattern_counter, 'momentum')
        
        # Reversal patterns
        for pattern in patterns.get('dbr_patterns', []):
            pattern_counter += 1
            self.plot_single_pattern(ax, pattern, data, pattern_counter, 'reversal')
        
        for pattern in patterns.get('rbd_patterns', []):
            pattern_counter += 1
            self.plot_single_pattern(ax, pattern, data, pattern_counter, 'reversal')
        
        # Format chart
        self.format_chart(ax, data, patterns)
        
        # Save and show
        self.save_chart(fig)
        plt.show()
    
    def plot_candlesticks(self, ax, data):
        """Plot candlesticks"""
        
        for i, row in data.iterrows():
            x = i
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Candle color
            if close_price > open_price:
                color = self.colors['bullish_candle']
                edge_color = '#27AE60'
                body_bottom = open_price
                body_top = close_price
            elif close_price < open_price:
                color = self.colors['bearish_candle']
                edge_color = '#C0392B'
                body_bottom = close_price
                body_top = open_price
            else:
                color = self.colors['doji_candle']
                edge_color = '#7F8C8D'
                body_bottom = open_price
                body_top = close_price
            
            # Draw wick
            ax.plot([x, x], [low_price, high_price], 
                   color=edge_color, linewidth=1.5, alpha=0.8)
            
            # Draw body
            body_height = abs(close_price - open_price)
            if body_height > 0:
                rect = patches.Rectangle((x - 0.4, body_bottom), 0.8, body_height,
                                       facecolor=color, edgecolor=edge_color,
                                       linewidth=0.8, alpha=0.8)
                ax.add_patch(rect)
            else:
                # Doji
                ax.plot([x - 0.4, x + 0.4], [open_price, open_price],
                       color=edge_color, linewidth=2.5)
    
    def plot_single_pattern(self, ax, pattern, data, pattern_num, category):
        """Plot individual pattern with detailed labeling"""
        
        pattern_type = pattern['type']
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        
        # Get pattern-specific color
        zone_color = self.colors.get(f"{pattern_type.lower().replace('-', '')}_zone", '#95A5A6')
        
        # Pattern structure indices
        leg_in_start = pattern['leg_in']['start_idx']
        leg_in_end = pattern['leg_in']['end_idx']
        base_start = pattern['base']['start_idx']
        base_end = pattern['base']['end_idx']
        leg_out_start = pattern['leg_out']['start_idx']
        leg_out_end = pattern['leg_out']['end_idx']
        
        # Plot zone rectangle
        zone_width = leg_out_end - leg_in_start + 2
        rect = patches.Rectangle((leg_in_start - 0.5, zone_low), zone_width, 
                               zone_high - zone_low,
                               facecolor=zone_color, alpha=0.2,
                               edgecolor=zone_color, linewidth=2,
                               label=f'{pattern_type} Zone' if pattern_num == 1 else '')
        ax.add_patch(rect)
        
        # Zone boundaries
        ax.hlines(zone_high, leg_in_start - 0.5, leg_out_end + 0.5, 
                 colors=zone_color, linewidth=2, alpha=0.8)
        ax.hlines(zone_low, leg_in_start - 0.5, leg_out_end + 0.5, 
                 colors=zone_color, linewidth=2, alpha=0.8)
        
        # Pattern structure markers
        # Leg-in
        ax.annotate('LEG-IN', xy=(leg_in_start + (leg_in_end - leg_in_start)/2, zone_high + 0.002),
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7),
                   color='white')
        
        # Base
        ax.annotate('BASE', xy=(base_start + (base_end - base_start)/2, zone_high + 0.002),
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.7),
                   color='white')
        
        # Leg-out
        ax.annotate('LEG-OUT', xy=(leg_out_start + (leg_out_end - leg_out_start)/2, zone_high + 0.002),
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7),
                   color='white')
        
        # Pattern info label
        leg_in_dir = "🔴" if pattern['leg_in']['direction'] == 'bearish' else "🟢"
        leg_out_dir = "🔴" if pattern['leg_out']['direction'] == 'bearish' else "🟢"
        base_candles = pattern['base']['candle_count']
        strength = pattern['strength']
        
        info_text = f"{pattern_type} #{pattern_num}\n"
        info_text += f"{leg_in_dir}→BASE→{leg_out_dir}\n"
        info_text += f"Base: {base_candles}C\n"
        info_text += f"Str: {strength:.2f}\n"
        info_text += f"({category})"
        
        # Position label to avoid overlap
        label_y = zone_low - 0.008 - (pattern_num % 3) * 0.006
        
        ax.text(leg_in_start + zone_width/2, label_y, info_text,
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor=zone_color, alpha=0.9),
               color='white')
    
    def format_chart(self, ax, data, patterns):
        """Format the chart"""
        
        total_patterns = patterns['total_patterns']
        momentum_count = len(patterns.get('dbd_patterns', [])) + len(patterns.get('rbr_patterns', []))
        reversal_count = len(patterns.get('dbr_patterns', [])) + len(patterns.get('rbd_patterns', []))
        
        title = f'Recent Pattern Analysis - Last 2 Months\n'
        title += f'Total Patterns: {total_patterns} | '
        title += f'Momentum: {momentum_count} | Reversal: {reversal_count}'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price', fontsize=14)
        ax.set_xlabel('Trading Days', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set date labels
        date_ticks = range(0, len(data), 5)  # Every 5 days
        date_labels = [data.iloc[i]['date'].strftime('%m-%d') for i in date_ticks]
        ax.set_xticks(date_ticks)
        ax.set_xticklabels(date_labels, rotation=45)
        
        # Legend for pattern types
        legend_text = (
            "🎯 PATTERN TYPES:\n"
            "🔵 D-B-D: Bearish→Base→Bearish (momentum)\n"
            "🟢 R-B-R: Bullish→Base→Bullish (momentum)\n"
            "🟣 D-B-R: Bearish→Base→Bullish (reversal)\n"
            "🟠 R-B-D: Bullish→Base→Bearish (reversal)"
        )
        
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95),
               fontsize=10, verticalalignment='top')
    
    def save_chart(self, fig):
        """Save the chart"""
        import os
        
        os.makedirs('results/recent_patterns', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/recent_patterns/recent_patterns_{timestamp}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {filename}")

def main():
    """Run the recent pattern visualizer"""
    visualizer = RecentPatternVisualizer()
    visualizer.visualize_last_2_months()

if __name__ == "__main__":
    main()