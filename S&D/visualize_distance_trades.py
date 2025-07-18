"""
Distance Edge Trade Visualizer
Shows actual patterns with distance ratios and trade outcomes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class DistanceTradeVisualizer:
    def __init__(self):
        self.colors = {
            'bullish_candle': '#2ECC71',
            'bearish_candle': '#E74C3C',
            'zone_valid': '#4ECDC4',
            'zone_high_quality': '#27AE60',
            'entry_point': '#3498DB',
            'stop_loss': '#E74C3C',
            'take_profit': '#2ECC71',
            'win_trade': '#27AE60',
            'loss_trade': '#E74C3C',
            'timeout_trade': '#F39C12'
        }
    
    def visualize_distance_samples(self, distance_threshold=2.5, max_samples=6):
        """
        Visualize sample patterns at specific distance threshold
        
        Args:
            distance_threshold: Minimum distance multiplier to show
            max_samples: Maximum number of charts to create
        """
        print(f"📊 DISTANCE TRADE VISUALIZER - {distance_threshold}x Threshold")
        print("=" * 60)
        
        # Load data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        # Use recent data for cleaner visualization
        viz_data = data.tail(500).copy()
        viz_data = viz_data.reset_index()
        
        print(f"📅 Analysis period: {viz_data['date'].iloc[0]} to {viz_data['date'].iloc[-1]}")
        
        # Detect patterns
        candle_classifier = CandleClassifier(viz_data)
        classified_data = candle_classifier.classify_all_candles()
        zone_detector = ZoneDetector(candle_classifier)
        patterns = zone_detector.detect_all_patterns(classified_data)
        
        # Get all patterns and filter by distance
        all_patterns = (patterns['dbd_patterns'] + patterns['rbr_patterns'] + 
                       patterns['dbr_patterns'] + patterns['rbd_patterns'])
        
        # Calculate distance ratios and filter
        qualified_patterns = []
        for pattern in all_patterns:
            base_range = pattern['base']['range']
            leg_out_range = pattern['leg_out']['range']
            
            if base_range > 0:
                distance_ratio = leg_out_range / base_range
                if distance_ratio >= distance_threshold:
                    pattern['distance_ratio'] = distance_ratio
                    # Simulate trade outcome
                    trade_outcome = self.simulate_trade_quick(pattern, viz_data)
                    if trade_outcome:
                        pattern['trade_outcome'] = trade_outcome
                        qualified_patterns.append(pattern)
        
        print(f"🎯 Qualified patterns at {distance_threshold}x: {len(qualified_patterns)}")
        
        if not qualified_patterns:
            print("❌ No patterns meet the distance threshold")
            return
        
        # Sort by distance ratio (highest quality first)
        qualified_patterns.sort(key=lambda x: x['distance_ratio'], reverse=True)
        
        # Create visualizations for top patterns
        patterns_to_show = qualified_patterns[:max_samples]
        
        for i, pattern in enumerate(patterns_to_show):
            self.create_pattern_trade_chart(pattern, viz_data, i + 1, distance_threshold)
        
        # Create summary chart
        self.create_distance_summary_chart(qualified_patterns, distance_threshold)
    
    def simulate_trade_quick(self, pattern, data):
        """Quick trade simulation for visualization"""
        try:
            pattern_end_idx = pattern['end_idx']
            
            if pattern_end_idx + 25 >= len(data):
                return None
            
            # Entry at pattern completion
            entry_candle = data.iloc[pattern_end_idx + 1]
            entry_price = entry_candle['open']
            
            # Direction and targets
            zone_high = pattern['zone_high']
            zone_low = pattern['zone_low']
            pattern_type = pattern['type']
            
            if pattern_type in ['R-B-R', 'D-B-R']:  # Buy patterns
                direction = 'BUY'
                stop_loss = zone_low * 0.995
                risk_distance = abs(entry_price - stop_loss)
                take_profit = entry_price + (risk_distance * 2)
            else:  # Sell patterns
                direction = 'SELL'
                stop_loss = zone_high * 1.005
                risk_distance = abs(entry_price - stop_loss)
                take_profit = entry_price - (risk_distance * 2)
            
            # Check outcome over next 20 candles
            for j in range(pattern_end_idx + 2, min(pattern_end_idx + 22, len(data))):
                candle = data.iloc[j]
                
                if direction == 'BUY':
                    if candle['low'] <= stop_loss:
                        return {'outcome': 'LOSS', 'exit_idx': j, 'exit_price': stop_loss}
                    elif candle['high'] >= take_profit:
                        return {'outcome': 'WIN', 'exit_idx': j, 'exit_price': take_profit}
                else:
                    if candle['high'] >= stop_loss:
                        return {'outcome': 'LOSS', 'exit_idx': j, 'exit_price': stop_loss}
                    elif candle['low'] <= take_profit:
                        return {'outcome': 'WIN', 'exit_idx': j, 'exit_price': take_profit}
            
            return {'outcome': 'TIMEOUT', 'exit_idx': min(pattern_end_idx + 21, len(data) - 1), 'exit_price': entry_price}
            
        except:
            return None
    
    def create_pattern_trade_chart(self, pattern, data, chart_num, distance_threshold):
        """Create detailed chart for single pattern trade"""
        
        # Chart window around pattern
        pattern_start = pattern['start_idx']
        pattern_end = pattern['end_idx']
        trade_outcome = pattern['trade_outcome']
        
        chart_start = max(0, pattern_start - 10)
        chart_end = min(len(data), trade_outcome['exit_idx'] + 5)
        chart_data = data.iloc[chart_start:chart_end]
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot candlesticks
        self.plot_candlesticks(ax, chart_data, chart_start)
        
        # Plot pattern structure
        self.plot_pattern_structure(ax, pattern, chart_start)
        
        # Plot trade execution
        self.plot_trade_execution(ax, pattern, trade_outcome, chart_start)
        
        # Format chart
        self.format_trade_chart(ax, pattern, trade_outcome, chart_num, distance_threshold, chart_data)
        
        # Save chart
        self.save_trade_chart(fig, chart_num, pattern, distance_threshold)
        
        plt.show()
    
    def plot_candlesticks(self, ax, data, chart_start):
        """Plot candlesticks"""
        for i, row in data.iterrows():
            x = i - chart_start
            
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
            else:
                color = self.colors['bearish_candle']
                edge_color = '#C0392B'
                body_bottom = close_price
                body_top = open_price
            
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
                ax.plot([x - 0.4, x + 0.4], [open_price, open_price],
                       color=edge_color, linewidth=2.5)
    
    def plot_pattern_structure(self, ax, pattern, chart_start):
        """Plot pattern structure with zone"""
        
        # Pattern indices adjusted for chart
        leg_in_start = pattern['leg_in']['start_idx'] - chart_start
        leg_in_end = pattern['leg_in']['end_idx'] - chart_start
        base_start = pattern['base']['start_idx'] - chart_start
        base_end = pattern['base']['end_idx'] - chart_start
        leg_out_start = pattern['leg_out']['start_idx'] - chart_start
        leg_out_end = pattern['leg_out']['end_idx'] - chart_start
        
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        distance_ratio = pattern['distance_ratio']
        
        # Zone quality color based on distance
        if distance_ratio >= 4.0:
            zone_color = '#27AE60'  # High quality green
            alpha = 0.3
        elif distance_ratio >= 2.5:
            zone_color = '#4ECDC4'  # Medium quality teal
            alpha = 0.25
        else:
            zone_color = '#95A5A6'  # Lower quality gray
            alpha = 0.2
        
        # Plot zone rectangle
        zone_width = leg_out_end - leg_in_start + 2
        rect = patches.Rectangle((leg_in_start - 0.5, zone_low), zone_width, 
                               zone_high - zone_low,
                               facecolor=zone_color, alpha=alpha,
                               edgecolor=zone_color, linewidth=2)
        ax.add_patch(rect)
        
        # Zone boundaries
        ax.hlines(zone_high, leg_in_start - 0.5, leg_out_end + 0.5, 
                 colors=zone_color, linewidth=3, alpha=0.9)
        ax.hlines(zone_low, leg_in_start - 0.5, leg_out_end + 0.5, 
                 colors=zone_color, linewidth=3, alpha=0.9)
        
        # Pattern labels
        ax.text(leg_in_start + (leg_in_end - leg_in_start)/2, zone_high + 0.003,
               'LEG-IN', ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.8),
               color='white')
        
        ax.text(base_start + (base_end - base_start)/2, zone_high + 0.003,
               'BASE', ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.8),
               color='white')
        
        ax.text(leg_out_start + (leg_out_end - leg_out_start)/2, zone_high + 0.003,
               'LEG-OUT', ha='center', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8),
               color='white')
    
    def plot_trade_execution(self, ax, pattern, trade_outcome, chart_start):
        """Plot trade entry, stop, target, and outcome"""
        
        pattern_end = pattern['end_idx'] - chart_start
        entry_idx = pattern_end + 1
        exit_idx = trade_outcome['exit_idx'] - chart_start
        
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        pattern_type = pattern['type']
        outcome = trade_outcome['outcome']
        
        # Calculate trade levels
        if pattern_type in ['R-B-R', 'D-B-R']:  # Buy trade
            direction = 'BUY'
            stop_loss = zone_low * 0.995
            entry_price = (zone_high + zone_low) / 2  # Approximate entry
            risk_distance = abs(entry_price - stop_loss)
            take_profit = entry_price + (risk_distance * 2)
            
            entry_color = self.colors['win_trade'] if outcome == 'WIN' else self.colors['loss_trade']
        else:  # Sell trade
            direction = 'SELL'
            stop_loss = zone_high * 1.005
            entry_price = (zone_high + zone_low) / 2
            risk_distance = abs(entry_price - stop_loss)
            take_profit = entry_price - (risk_distance * 2)
            
            entry_color = self.colors['win_trade'] if outcome == 'WIN' else self.colors['loss_trade']
        
        # Entry marker
        ax.scatter(entry_idx, entry_price, marker='^' if direction == 'BUY' else 'v',
                  s=200, color=entry_color, edgecolor='white', linewidth=2,
                  zorder=10, label=f'{direction} Entry')
        
        # Exit marker
        if outcome != 'TIMEOUT':
            exit_marker = '^' if outcome == 'WIN' else 'v'
            exit_color = self.colors['win_trade'] if outcome == 'WIN' else self.colors['loss_trade']
            ax.scatter(exit_idx, trade_outcome['exit_price'], marker=exit_marker,
                      s=150, color=exit_color, edgecolor='white', linewidth=2,
                      zorder=10, label=f'{outcome} Exit')
        
        # Trade levels
        ax.hlines(stop_loss, entry_idx, exit_idx, colors=self.colors['stop_loss'],
                 linewidth=2, linestyle='--', alpha=0.8, label='Stop Loss')
        ax.hlines(take_profit, entry_idx, exit_idx, colors=self.colors['take_profit'],
                 linewidth=2, linestyle=':', alpha=0.8, label='Take Profit')
        
        # Trade path line
        path_color = self.colors['win_trade'] if outcome == 'WIN' else (
            self.colors['loss_trade'] if outcome == 'LOSS' else self.colors['timeout_trade'])
        ax.plot([entry_idx, exit_idx], [entry_price, trade_outcome['exit_price']], 
               color=path_color, linewidth=3, alpha=0.7, label=f'Trade Path ({outcome})')
    
    def format_trade_chart(self, ax, pattern, trade_outcome, chart_num, distance_threshold, data):
        """Format the trade chart"""
        
        pattern_type = pattern['type']
        distance_ratio = pattern['distance_ratio']
        base_candles = pattern['base']['candle_count']
        outcome = trade_outcome['outcome']
        
        # Chart title
        title = f'Trade #{chart_num}: {pattern_type} Pattern (Distance: {distance_ratio:.1f}x)\n'
        title += f'Base: {base_candles} candles | Outcome: {outcome} | '
        title += f'Threshold: {distance_threshold}x minimum'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Days', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Date labels
        if len(data) > 10:
            date_ticks = range(0, len(data), max(1, len(data)//8))
            date_labels = [data.iloc[i]['date'].strftime('%m-%d') for i in date_ticks if i < len(data)]
            ax.set_xticks(date_ticks[:len(date_labels)])
            ax.set_xticklabels(date_labels, rotation=45)
        
        # Pattern info box
        info_text = (
            f"📊 PATTERN ANALYSIS:\n"
            f"Distance Ratio: {distance_ratio:.1f}x (vs {distance_threshold}x min)\n"
            f"Base Quality: {base_candles} candles\n"
            f"Zone Range: {pattern['zone_range']/0.0001:.0f} pips\n"
            f"Pattern Strength: {pattern['strength']:.2f}"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95),
               fontsize=10, verticalalignment='top')
    
    def save_trade_chart(self, fig, chart_num, pattern, distance_threshold):
        """Save individual trade chart"""
        import os
        
        os.makedirs('results/distance_trades', exist_ok=True)
        
        pattern_type = pattern['type'].replace('-', '')
        distance_ratio = pattern['distance_ratio']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f'results/distance_trades/trade_{chart_num:02d}_{pattern_type}_{distance_ratio:.1f}x_{timestamp}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Trade chart saved: {filename}")
    
    def create_distance_summary_chart(self, patterns, distance_threshold):
        """Create summary chart showing all patterns by distance"""
        
        # Group patterns by distance ranges
        distance_ranges = {
            '1.0-1.9x': [],
            '2.0-2.9x': [],
            '3.0-3.9x': [],
            '4.0-4.9x': [],
            '5.0x+': []
        }
        
        for pattern in patterns:
            ratio = pattern['distance_ratio']
            outcome = pattern['trade_outcome']['outcome']
            
            if ratio < 2.0:
                distance_ranges['1.0-1.9x'].append(outcome)
            elif ratio < 3.0:
                distance_ranges['2.0-2.9x'].append(outcome)
            elif ratio < 4.0:
                distance_ranges['3.0-3.9x'].append(outcome)
            elif ratio < 5.0:
                distance_ranges['4.0-4.9x'].append(outcome)
            else:
                distance_ranges['5.0x+'].append(outcome)
        
        # Create summary chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ranges = list(distance_ranges.keys())
        win_rates = []
        trade_counts = []
        
        for range_name, outcomes in distance_ranges.items():
            if outcomes:
                wins = outcomes.count('WIN')
                total = len(outcomes)
                win_rate = (wins / total) * 100
                win_rates.append(win_rate)
                trade_counts.append(total)
            else:
                win_rates.append(0)
                trade_counts.append(0)
        
        # Bar chart
        bars = ax.bar(ranges, win_rates, alpha=0.8, color='#4ECDC4')
        
        # Add trade count labels on bars
        for bar, count in zip(bars, trade_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{count} trades', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Win Rate by Distance Range (≥{distance_threshold}x threshold)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_xlabel('Distance Ratio Range', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Save summary
        os.makedirs('results/distance_trades', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/distance_trades/distance_summary_{distance_threshold}x_{timestamp}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Summary chart saved: {filename}")
        
        plt.show()

def main():
    """Run distance trade visualization"""
    print("🎨 DISTANCE TRADE VISUALIZER")
    print("=" * 40)
    
    # Get user input for distance threshold
    print("Select distance threshold to visualize:")
    print("1. 1.5x (Current system)")
    print("2. 2.5x (Manual preference)")
    print("3. 3.0x (High quality)")
    print("4. 4.0x (Super quality)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    thresholds = {'1': 1.5, '2': 2.5, '3': 3.0, '4': 4.0}
    threshold = thresholds.get(choice, 2.5)
    
    visualizer = DistanceTradeVisualizer()
    visualizer.visualize_distance_samples(distance_threshold=threshold, max_samples=6)

if __name__ == "__main__":
    main()