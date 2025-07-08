"""
Manual Trade Chart Visualizer
Shows exact chart data around each manual trade for validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class ManualTradeChartVisualizer:
    def __init__(self):
        self.colors = {
            'bullish_candle': '#2ECC71',
            'bearish_candle': '#E74C3C',
            'doji_candle': '#95A5A6',
            'manual_zone': '#FFD700',
            'detected_zone': '#FF6B6B',
            'base_end_marker': '#FF0000'
        }
    
    def visualize_all_manual_trades(self, csv_file='manual_trades_validation.csv'):
        """Create charts for all manual trades"""
        
        # Load manual trades
        try:
            manual_trades = pd.read_csv(csv_file)
            print(f"📋 Loaded {len(manual_trades)} manual trades")
        except FileNotFoundError:
            print(f"❌ File {csv_file} not found!")
            return
        
        # Load EURUSD data
        data_loader = DataLoader()
        data = data_loader.load_pair_data('EURUSD', 'Daily')
        
        for index, trade in manual_trades.iterrows():
            # Process all trades (no limit)
                
            self.create_trade_chart(trade, data, index + 1)
    
    def create_trade_chart(self, manual_trade, data, trade_num):
        """Create detailed chart for single manual trade"""
        
        trade_id = manual_trade['trade_id']
        direction = str(manual_trade['direction']).upper().strip()
        zone_low = manual_trade['zone_low']
        zone_high = manual_trade['zone_high']
        base_end_date = manual_trade['base_end_date']
        
        # Check if trade is invalid
        entry_date = manual_trade['entry_date']
        is_invalid = (pd.isna(entry_date) or str(entry_date).strip() in ['N/A', '', 'Invalid'])
        
        print(f"\n📊 Creating chart for {trade_id}: {direction} | {zone_low:.4f}-{zone_high:.4f}")
        
        try:
            # Parse base end date
            try:
                parsed_date = pd.to_datetime(base_end_date, format='%d-%m-%Y')
            except:
                parsed_date = pd.to_datetime(base_end_date)
            
            # Handle missing dates (weekends/holidays)
            try:
                base_end_idx = data.index.get_loc(parsed_date)
            except KeyError:
                # Find nearest trading date
                nearest_date = data.index[data.index.get_indexer([parsed_date], method='nearest')[0]]
                base_end_idx = data.index.get_loc(nearest_date)
                print(f"   ⚠️  Date {parsed_date.date()} not found, using nearest: {nearest_date.date()}")
            
            # Create wider window for context (30 candles before, 15 after)
            window_start = max(0, base_end_idx - 30)
            window_end = min(len(data), base_end_idx + 15)
            chart_data = data.iloc[window_start:window_end].copy()
            chart_data = chart_data.reset_index()
            
            # Find base end marker position in chart
            base_end_chart_idx = base_end_idx - window_start
            
            # Run zone detection on this window
            candle_classifier = CandleClassifier(chart_data)
            classified_data = candle_classifier.classify_all_candles()
            zone_detector = ZoneDetector(candle_classifier)
            patterns = zone_detector.detect_all_patterns(classified_data)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(20, 10))
            
            # Plot candlesticks
            self.plot_candlesticks(ax, chart_data)
            
            # Mark manual zone
            self.plot_manual_zone(ax, zone_low, zone_high, len(chart_data), is_invalid)
            
            # Mark detected zones
            zone_type = 'rbr_patterns' if direction == 'BUY' else 'dbd_patterns'
            detected_zones = patterns[zone_type]
            self.plot_detected_zones(ax, detected_zones, len(chart_data))
            
            # Mark base end date
            self.mark_base_end_date(ax, base_end_chart_idx, chart_data, parsed_date)
            
            # Add candle classifications as text
            self.add_candle_info(ax, chart_data, candle_classifier, base_end_chart_idx)
            
            # Format chart
            self.format_chart(ax, trade_id, direction, zone_low, zone_high, 
                            parsed_date, is_invalid, len(detected_zones), chart_data)
            
            # Save chart
            self.save_chart(fig, trade_id)
            
            plt.close()  # Close to save memory
            
        except Exception as e:
            print(f"❌ Error creating chart for {trade_id}: {e}")
            import traceback
            traceback.print_exc()  # Show full error details
    
    def plot_candlesticks(self, ax, data):
        """Plot candlesticks with classifications"""
        
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
    
    def plot_manual_zone(self, ax, zone_low, zone_high, chart_length, is_invalid):
        """Plot manual zone as golden rectangle"""
        
        zone_color = '#FFD700' if not is_invalid else '#FFA500'
        alpha = 0.3 if not is_invalid else 0.2
        
        rect = patches.Rectangle((0, zone_low), chart_length, zone_high - zone_low,
                               facecolor=zone_color, alpha=alpha,
                               edgecolor=zone_color, linewidth=2,
                               label=f'Manual Zone {"(INVALID)" if is_invalid else ""}')
        ax.add_patch(rect)
        
        # Zone boundary lines
        ax.hlines(zone_high, 0, chart_length, colors=zone_color, linewidth=2, alpha=0.8)
        ax.hlines(zone_low, 0, chart_length, colors=zone_color, linewidth=2, alpha=0.8)
        
        # Zone label
        ax.text(chart_length * 0.02, (zone_high + zone_low) / 2, 
                f'MANUAL\n{zone_low:.4f}-{zone_high:.4f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.8),
                fontsize=10, fontweight='bold', ha='left')
    
    def plot_detected_zones(self, ax, detected_zones, chart_length):
        """Plot detected zones as red rectangles with proper numbering"""
        
        for i, zone in enumerate(detected_zones):
            zone_color = '#FF6B6B'
            zone_number = i + 1
            
            rect = patches.Rectangle((0, zone['zone_low']), chart_length, 
                                   zone['zone_high'] - zone['zone_low'],
                                   facecolor=zone_color, alpha=0.2,
                                   edgecolor=zone_color, linewidth=2, linestyle='--',
                                   label=f'Detected Zone {zone_number}' if i == 0 else '')
            ax.add_patch(rect)
            
            # Position labels to avoid overlap
            label_x_positions = [0.1, 0.5, 0.8]  # Different X positions for each zone
            label_x = chart_length * label_x_positions[i % len(label_x_positions)]
            
            # Zone label with consistent numbering
            ax.text(label_x, (zone['zone_high'] + zone['zone_low']) / 2, 
                    f'DETECTED {zone_number}\n{zone["zone_low"]:.4f}-{zone["zone_high"]:.4f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.9),
                    fontsize=9, fontweight='bold', ha='center', color='white')
    
    def highlight_best_match(self, ax, best_match_zone, chart_length):
        """Highlight the zone that was selected as best match"""
        if best_match_zone:
            # Add thick green border around the selected zone
            rect = patches.Rectangle((0, best_match_zone['zone_low']), chart_length, 
                                   best_match_zone['zone_high'] - best_match_zone['zone_low'],
                                   facecolor='none', edgecolor='#00FF00', 
                                   linewidth=4, linestyle='-', alpha=0.8,
                                   label='SYSTEM SELECTED MATCH')
            ax.add_patch(rect)
            
            # Add "SELECTED" label
            ax.text(chart_length * 0.95, (best_match_zone['zone_high'] + best_match_zone['zone_low']) / 2,
                    'SELECTED\nMATCH',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#00FF00', alpha=0.9),
                    fontsize=10, fontweight='bold', ha='center', color='black')
    
    def mark_base_end_date(self, ax, base_end_idx, chart_data, base_end_date):
        """Mark the base end date with arrow at top (doesn't hide candles)"""
        
        if 0 <= base_end_idx < len(chart_data):
            # Get price range for positioning
            price_range = chart_data['high'].max() - chart_data['low'].min()
            chart_top = chart_data['high'].max() + (price_range * 0.05)
            arrow_start = chart_top + (price_range * 0.08)
            
            # Draw arrow pointing down to base end
            ax.annotate('', xy=(base_end_idx, chart_top), 
                       xytext=(base_end_idx, arrow_start),
                       arrowprops=dict(arrowstyle='->', color=self.colors['base_end_marker'], 
                                     lw=3, alpha=0.9),
                       label=f'Base End: {base_end_date.strftime("%Y-%m-%d")}')
            
            # Add date label above arrow
            ax.text(base_end_idx, arrow_start + (price_range * 0.02), 
                    f'BASE END\n{base_end_date.strftime("%Y-%m-%d")}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=self.colors['base_end_marker'], 
                             alpha=0.9, edgecolor='white', linewidth=2),
                    color='white')
            
            # Add small marker dot at exact candle
            candle_price = (chart_data.iloc[base_end_idx]['high'] + chart_data.iloc[base_end_idx]['low']) / 2
            ax.scatter(base_end_idx, candle_price, color=self.colors['base_end_marker'], 
                      s=100, marker='o', edgecolor='white', linewidth=2, zorder=10,
                      alpha=0.9)
    
    def add_candle_info(self, ax, chart_data, candle_classifier, base_end_idx):
        """Add candle classification info around base end"""
        
        # Show classifications for 5 candles around base end
        start_info = max(0, base_end_idx - 2)
        end_info = min(len(chart_data), base_end_idx + 3)
        
        for i in range(start_info, end_info):
            if i < len(chart_data):
                candle = chart_data.iloc[i]
                classification = candle_classifier.classify_single_candle(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
                
                # Color based on classification
                if classification == 'base':
                    text_color = '#3498DB'
                elif classification == 'decisive':
                    text_color = '#F39C12'
                else:  # explosive
                    text_color = '#E74C3C'
                
                # Add classification below candle
                ax.text(i, candle['low'] - (chart_data['high'].max() - chart_data['low'].min()) * 0.08,
                       classification[0].upper(),  # First letter
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color=text_color)
    
    def format_chart(self, ax, trade_id, direction, zone_low, zone_high, 
                    base_end_date, is_invalid, detected_count, chart_data):
        """Format the chart with titles and labels"""
        
        invalid_text = " [INVALID]" if is_invalid else ""
        
        ax.set_title(f'{trade_id}: {direction} Trade{invalid_text}\n'
                    f'Manual Zone: {zone_low:.4f}-{zone_high:.4f} | '
                    f'Base End: {base_end_date.strftime("%Y-%m-%d")} | '
                    f'Detected Zones: {detected_count}',
                    fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Days (Red Arrow = Base End Date)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # Set y-limits to accommodate arrow above chart
        price_range = chart_data['high'].max() - chart_data['low'].min()
        ax.set_ylim(chart_data['low'].min() - (price_range * 0.15), 
                   chart_data['high'].max() + (price_range * 0.20))
        
        # Add classification legend
        legend_text = "Candle Types: B=Base(≤50%), D=Decisive(50-80%), E=Explosive(>80%)"
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, 
               fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def save_chart(self, fig, trade_id):
        """Save chart to results folder"""
        
        import os
        os.makedirs('results/manual_trade_charts', exist_ok=True)
        
        filename = f"results/manual_trade_charts/{trade_id}_chart.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Chart saved: {filename}")

def main():
    """Create charts for all manual trades"""
    print("📊 MANUAL TRADE CHART VISUALIZER")
    print("=" * 50)
    
    visualizer = ManualTradeChartVisualizer()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'manual_trades_validation.csv')
    
    visualizer.visualize_all_manual_trades(csv_path)
    
    print(f"\n✅ Charts created in: results/manual_trade_charts/")
    print(f"📋 Open the PNG files to see exact chart data around each manual trade")

if __name__ == "__main__":
    main()