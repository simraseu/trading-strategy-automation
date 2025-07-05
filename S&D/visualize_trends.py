"""
Triple EMA Trend Visualization Tool - Module 3 (WITH RANGING FILTER)
Visual validation of Triple EMA trend classification with ranging filter
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from modules.data_loader import DataLoader
from modules.trend_classifier import TrendClassifier
from config.settings import TREND_CONFIG

# Set professional styling
plt.style.use('default')
sns.set_palette("husl")

class TripleEMAVisualizer:
   def __init__(self):
       self.colors = {
           # Triple EMA colors
           'ema_50': '#3498DB',           # Blue (Fast)
           'ema_100': '#FF6B6B',          # Red (Medium)
           'ema_200': '#9B59B6',          # Purple (Slow)
           'price': '#34495E',            # Dark gray
           
           # Filtered trend background colors
           'strong_bullish': '#27AE60',   # Dark green
           'medium_bullish': '#2ECC71',   # Green
           'weak_bullish': '#58D68D',     # Light green
           'strong_bearish': '#C0392B',   # Dark red
           'medium_bearish': '#E74C3C',   # Red
           'weak_bearish': '#EC7063',     # Light red
           'ranging': '#95A5A6',          # Gray (NEW - for ranging markets)
           'neutral': '#BDC3C7',          # Light gray
           
           # UI colors
           'background': '#F8F9FA',       # Light gray
           'grid': '#BDC3C7',             # Light gray
           'text': '#2C3E50'              # Dark blue-gray
       }
       
   def plot_trend_analysis_filtered(self, data, sample_size=500):
       """
       Plot comprehensive Triple EMA trend analysis with ranging filter
       
       Args:
           data: DataFrame with filtered trend analysis
           sample_size: Number of recent candles to display
       """
       
       # Use recent data for relevance
       if len(data) > sample_size:
           plot_data = data.tail(sample_size).copy()
       else:
           plot_data = data.copy()
       
       # Reset index for plotting
       plot_data = plot_data.reset_index(drop=True)
       
       # Create figure with subplots
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14), 
                                     gridspec_kw={'height_ratios': [3, 1]})
       
       # Main price and EMA plot
       self.plot_price_and_triple_emas_filtered(ax1, plot_data)
       
       # EMA separation subplot
       self.plot_ema_separation(ax2, plot_data)
       
       # Add trend change markers
       self.mark_trend_changes_filtered(ax1, plot_data)
       
       plt.tight_layout()
       plt.show()

   def plot_price_and_triple_emas_filtered(self, ax, data):
       """Plot price, all 3 EMAs, and filtered trend background"""
       
       x = range(len(data))
       
       # Plot trend background zones using FILTERED trends
       self.plot_trend_background_filtered(ax, data)
       
       # Plot price line (thicker, on top)
       ax.plot(x, data['close'], color=self.colors['price'], 
               linewidth=3, label='Price', alpha=0.9, zorder=5)
       
       # Plot all 3 EMAs
       ax.plot(x, data['ema_50'], color=self.colors['ema_50'], 
               linewidth=2, label='EMA 50 (Fast)', alpha=0.9, zorder=4)
       ax.plot(x, data['ema_100'], color=self.colors['ema_100'], 
               linewidth=2, label='EMA 100 (Medium)', alpha=0.9, zorder=3)
       ax.plot(x, data['ema_200'], color=self.colors['ema_200'], 
               linewidth=2, label='EMA 200 (Slow)', alpha=0.9, zorder=2)
       
       # Formatting
       ax.set_title('EURUSD Daily - Triple EMA with Ranging Filter', 
                   fontsize=16, weight='bold', color=self.colors['text'])
       ax.set_ylabel('Price', fontsize=12, color=self.colors['text'])
       ax.grid(True, alpha=0.3, color=self.colors['grid'])
       ax.legend(loc='upper left', fontsize=11)
       
       # Set background
       ax.set_facecolor(self.colors['background'])

   def plot_trend_background_filtered(self, ax, data):
       """Plot colored background zones for filtered trend types"""
       
       # Use the FILTERED trend column
       current_trend = data['trend_filtered'].iloc[0]
       start_idx = 0
       
       for i in range(1, len(data)):
           if data['trend_filtered'].iloc[i] != current_trend:
               # End of current trend period
               self.add_trend_zone(ax, start_idx, i-1, current_trend, data)
               start_idx = i
               current_trend = data['trend_filtered'].iloc[i]
       
       # Add final trend zone
       self.add_trend_zone(ax, start_idx, len(data)-1, current_trend, data)

   def add_trend_zone(self, ax, start_idx, end_idx, trend, data):
       """Add colored background zone for specific trend type"""
       
       # Get color based on trend type
       if trend in self.colors:
           color = self.colors[trend]
       else:
           color = self.colors['neutral']
       
       alpha = 0.15  # Semi-transparent background
       
       # Get price range for the zone
       y_min = data['close'].min() * 0.999
       y_max = data['close'].max() * 1.001
       
       # Create rectangle
       rect = patches.Rectangle((start_idx, y_min), 
                              end_idx - start_idx, 
                              y_max - y_min,
                              facecolor=color, 
                              alpha=alpha,
                              edgecolor='none',
                              zorder=1)
       ax.add_patch(rect)

   def plot_ema_separation(self, ax, data):
       """Plot EMA separation over time"""
       
       x = range(len(data))
       
       # Plot EMA separation
       ax.plot(x, data['ema_separation'], color='#F39C12', 
               linewidth=2, label='EMA Separation')
       
       # Add separation threshold line
       ax.axhline(y=0.3, color='#E74C3C', linestyle='--', alpha=0.7, 
                  label='Ranging Threshold (0.3)')
       
       # Fill areas
       ax.fill_between(x, data['ema_separation'], alpha=0.3, color='#F39C12')
       
       # Highlight ranging periods
       ranging_mask = data['ema_separation'] < 0.3
       ax.fill_between(x, 0, 1, where=ranging_mask, alpha=0.2, color='#95A5A6',
                       label='Ranging Periods')
       
       # Formatting
       ax.set_title('EMA Separation (Trend Strength Filter)', fontsize=12, weight='bold')
       ax.set_xlabel('Candle Index', fontsize=10)
       ax.set_ylabel('Separation Score', fontsize=10)
       ax.set_ylim(0, 1)
       ax.grid(True, alpha=0.3)
       ax.legend(loc='upper right', fontsize=8)
       ax.set_facecolor(self.colors['background'])

   def mark_trend_changes_filtered(self, ax, data):
       """Mark filtered trend change points on the chart"""
       
       # Find trend changes using FILTERED trends
       trend_changes = []
       for i in range(1, len(data)):
           if data['trend_filtered'].iloc[i] != data['trend_filtered'].iloc[i-1]:
               trend_changes.append((i, data['close'].iloc[i], 
                                   data['trend_filtered'].iloc[i-1], 
                                   data['trend_filtered'].iloc[i]))
       
       # Mark each change with appropriate symbol
       for idx, price, old_trend, new_trend in trend_changes:
           # Determine marker based on new trend
           if 'bullish' in new_trend:
               marker_color = '#27AE60'  # Green
               marker = '^'  # Up arrow
           elif 'bearish' in new_trend:
               marker_color = '#C0392B'  # Red
               marker = 'v'  # Down arrow
           elif new_trend == 'ranging':
               marker_color = '#95A5A6'  # Gray
               marker = 'o'  # Circle
           else:
               marker_color = '#95A5A6'  # Gray
               marker = 'o'  # Circle
           
           ax.scatter(idx, price, color=marker_color, marker=marker, 
                     s=120, zorder=10, edgecolor='white', linewidth=2)

   def create_trend_summary_filtered(self, classifier):
       """Create text summary of filtered Triple EMA trend analysis"""
       
       current = classifier.get_current_trend()
       changes = classifier.detect_trend_changes()
       
       print("\n" + "="*70)
       print("📊 FILTERED TRIPLE EMA TREND ANALYSIS SUMMARY")
       print("="*70)
       
       print(f"📈 Filtered Trend Statistics:")
       print(f"   Total periods: {len(classifier.data)}")
       
       # Show filtered trend distribution
       for trend_type, count in classifier.trends_filtered.items():
           percentage = (count / len(classifier.data)) * 100
           print(f"   {trend_type}: {count} ({percentage:.1f}%)")
       
       print(f"\n🎯 Current Status:")
       print(f"   Original Trend: {current['trend'].upper()}")
       print(f"   Filtered Trend: {classifier.data['trend_filtered'].iloc[-1].upper()}")
       print(f"   EMA Separation: {classifier.data['ema_separation'].iloc[-1]:.3f}")
       print(f"   EMA50: {current['ema_50']:.5f}")
       print(f"   EMA100: {current['ema_100']:.5f}")
       print(f"   EMA200: {current['ema_200']:.5f}")
       print(f"   EMA Order: {current['ema_order']}")
       
       print(f"\n🔄 Trend Changes:")
       print(f"   Total changes detected: {len(changes)}")
       
       if changes:
           print(f"   Recent changes:")
           for change in changes[-8:]:  # Show last 8 changes
               print(f"      Index {change['index']:4d}: {change['from_trend']} → {change['to_trend']}")
       
       print("\n🎨 FILTERED TREND COLOR LEGEND:")
       print("   🟢 Strong Bullish  = EMA50 > EMA100 > EMA200 (Good separation)")
       print("   🟢 Medium Bullish  = EMA50 > EMA100, EMA100 < EMA200 (Good separation)")
       print("   🟢 Weak Bullish    = EMA50 > EMA100 & EMA200, EMA100 < EMA200 (Good separation)")
       print("   🔴 Strong Bearish  = EMA50 < EMA100 < EMA200 (Good separation)")
       print("   🔴 Medium Bearish  = EMA50 < EMA100, EMA100 > EMA200 (Good separation)")
       print("   🔴 Weak Bearish    = EMA50 < EMA100 & EMA200, EMA100 > EMA200 (Good separation)")
       print("   ⚪ Ranging         = EMAs too close together (Avoid trading)")
       
       print("="*70)

def visualize_eurusd_triple_ema():
   """Main function to visualize EURUSD Triple EMA trends with ranging filter"""
   
   print("🎨 EURUSD TRIPLE EMA TREND VISUALIZATION (WITH RANGING FILTER)")
   print("=" * 60)
   
   try:
       # Load data
       print("📊 Loading EURUSD data...")
       data_loader = DataLoader()
       data = data_loader.load_pair_data('EURUSD', 'Daily')
       
       print(f"✅ Loaded {len(data)} candles")
       
       # Classify trends WITH FILTER
       print("🔍 Analyzing Triple EMA trends with ranging filter...")
       classifier = TrendClassifier(data)
       results = classifier.classify_trend_with_filter(min_separation=0.3)
       
       print("✅ Triple EMA trend analysis complete")
       
       # Create visualizer
       visualizer = TripleEMAVisualizer()
       
       # Show summary
       visualizer.create_trend_summary_filtered(classifier)
       
       # Create visualization
       print("\n🖼️  Creating Triple EMA visualization with ranging filter...")
       visualizer.plot_trend_analysis_filtered(results, sample_size=500)
       
       print("✅ Visualization complete!")
       
       return True
       
   except Exception as e:
       print(f"❌ Error: {e}")
       import traceback
       traceback.print_exc()
       return False

if __name__ == "__main__":
   success = visualize_eurusd_triple_ema()
   
   if success:
       print("\n🎉 Triple EMA trend visualization successful!")
       print("📋 Visual validation points:")
       print("   • 3 EMA lines: Blue(50), Red(100), Purple(200)")
       print("   • Background colors show 6 different trend types + ranging")
       print("   • Gray background = Ranging periods (avoid trading)")
       print("   • Bottom chart shows EMA separation threshold")
       print("   • Arrows mark trend changes")
       print("   • Check that ranging periods align with close EMAs")
   else:
       print("\n❌ Visualization failed!")