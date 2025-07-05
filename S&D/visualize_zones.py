"""
Enhanced Zone Visualization Tool - Module 2 (DEBUG ONLY VERSION)
Shows base zones for debugging and testing - NO FILE CREATION
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector
from config.settings import ZONE_CONFIG

# Set professional styling
plt.style.use('default')
sns.set_palette("husl")

class EnhancedZoneVisualizer:
    def __init__(self):
        self.colors = {
            'dbd_zone': '#FF6B6B',      # Red for supply zones
            'rbr_zone': '#4ECDC4',      # Teal for demand zones
            'dbd_edge': '#CC0000',      # Dark red edge
            'rbr_edge': '#006666',      # Dark teal edge
            'bullish_candle': '#2ECC71', # Green
            'bearish_candle': '#E74C3C', # Red
            'neutral_candle': '#95A5A6', # Gray
            'background': '#F8F9FA',     # Light gray
            'text': '#2C3E50'           # Dark blue-gray
        }
        
    def plot_enhanced_candlesticks(self, ax, data, start_idx=0):
        """Plot professional candlesticks with better styling"""
        
        for i, (idx, candle) in enumerate(data.iterrows()):
            x = start_idx + i
            open_price = candle['open']
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']
            
            # Determine candle color and style
            if close_price > open_price:
                color = self.colors['bullish_candle']
                edge_color = '#27AE60'
                alpha = 0.8
                body_bottom = open_price
                body_top = close_price
            elif close_price < open_price:
                color = self.colors['bearish_candle']
                edge_color = '#C0392B'
                alpha = 0.8
                body_bottom = close_price
                body_top = open_price
            else:
                color = self.colors['neutral_candle']
                edge_color = '#7F8C8D'
                alpha = 0.6
                body_bottom = open_price
                body_top = close_price
            
            # Draw wick with better styling
            ax.plot([x, x], [low_price, high_price], 
                   color=edge_color, linewidth=1.2, alpha=0.9, solid_capstyle='round')
            
            # Draw body with better proportions
            body_height = abs(close_price - open_price)
            if body_height > 0:
                rect = patches.Rectangle((x - 0.3, body_bottom), 0.6, body_height, 
                                       facecolor=color, edgecolor=edge_color, 
                                       linewidth=0.8, alpha=alpha)
                ax.add_patch(rect)
            else:
                # Doji - horizontal line
                ax.plot([x - 0.3, x + 0.3], [open_price, open_price], 
                       color=edge_color, linewidth=2.5, alpha=0.9)

    def calculate_base_zone_boundaries(self, pattern, data):
        """
        Calculate BASE ZONE boundaries with correct candle direction logic
        R-B-R: Base wick-low to highest open/close within base (per candle direction)
        D-B-D: Base wick-high to lowest open/close within base (per candle direction)
        """
        base = pattern['base']
        base_data = data.iloc[base['start_idx']:base['end_idx'] + 1]
        
        if pattern['type'] == 'R-B-R':
            # R-B-R: From base wick-low to highest open/close per candle
            zone_low = base_data['low'].min()        # Base wick-low (unchanged)
            
            # For each candle, take the higher of open/close
            highest_points = []
            for idx in base_data.index:
                candle = base_data.loc[idx]
                if candle['close'] >= candle['open']:  # Bullish candle
                    highest_points.append(candle['close'])  # Take close
                else:  # Bearish candle  
                    highest_points.append(candle['open'])   # Take open
            
            zone_high = max(highest_points)  # Highest among all candle tops
            
        else:  # D-B-D
            # D-B-D: From base wick-high to lowest open/close per candle
            zone_high = base_data['high'].max()      # Base wick-high (unchanged)
            
            # For each candle, take the lower of open/close
            lowest_points = []
            for idx in base_data.index:
                candle = base_data.loc[idx]
                if candle['close'] <= candle['open']:  # Bearish candle
                    lowest_points.append(candle['close'])   # Take close
                else:  # Bullish candle
                    lowest_points.append(candle['open'])    # Take open
            
            zone_low = min(lowest_points)   # Lowest among all candle bottoms
        
        return zone_high, zone_low

    def calculate_zone_score(self, pattern):
        """Calculate comprehensive zone score (0-100)"""
        
        # Base scoring components
        leg_in_score = pattern['leg_in']['strength'] * 20      # 0-20 points
        base_score = pattern['base']['quality_score'] * 30     # 0-30 points  
        leg_out_score = pattern['leg_out']['strength'] * 25    # 0-25 points
        
        # Distance bonus (most important for momentum)
        distance_ratio = pattern['leg_out']['ratio_to_base']
        distance_score = min(distance_ratio / 2.0, 1.0) * 20  # 0-20 points
        
        # Base candle bonus (1-2 candles optimal)
        base_candles = pattern['base']['candle_count']
        if base_candles <= 2:
            base_bonus = 5
        elif base_candles == 3:
            base_bonus = 3
        else:
            base_bonus = 0
            
        total_score = leg_in_score + base_score + leg_out_score + distance_score + base_bonus
        return min(total_score, 100)  # Cap at 100

    def create_zone_label(self, ax, pattern, x, y, pattern_num, zone_range):
        """Create professional zone label with comprehensive info"""
        
        zone_score = self.calculate_zone_score(pattern)
        
        # Create label content
        pattern_type = pattern['type']
        strength = pattern['strength']
        leg_out_ratio = pattern['leg_out']['ratio_to_base']
        base_candles = pattern['base']['candle_count']
        zone_pips = (zone_range / 0.0001)
        
        # Create compact multi-line label
        label_text = f"{pattern_type} #{pattern_num}\n"
        label_text += f"Score: {zone_score:.0f}/100\n"
        label_text += f"Str: {strength:.2f} | Out: {leg_out_ratio:.1f}x\n"
        label_text += f"Base: {base_candles}C | {zone_pips:.0f}p"
        
        # Color based on score
        if zone_score >= 80:
            bbox_color = '#2ECC71'  # Green - excellent
            text_color = 'white'
        elif zone_score >= 60:
            bbox_color = '#F39C12'  # Orange - good
            text_color = 'white'
        else:
            bbox_color = '#E74C3C'  # Red - weak
            text_color = 'white'
            
        # Create label with better styling
        ax.text(x, y, label_text, 
               ha='center', va='center', fontsize=9, color=text_color, 
               weight='bold', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color, 
                        alpha=0.95, edgecolor='white', linewidth=1.5))

    def visualize_zones_debug(self, sample_size=100, timeframe='Daily'):
        """DEBUG VISUALIZATION - Shows base zones only, NO FILE CREATION"""
        
        print("🎨 DEBUG BASE ZONE VISUALIZATION")
        print("=" * 50)
        
        try:
            # Load and prepare data
            print(f"📊 Loading EURUSD {timeframe} data...")
            data_loader = DataLoader()
            data = data_loader.load_pair_data('EURUSD', timeframe)
            
            # Use recent data for relevance
            sample_data = data.tail(sample_size).copy()
            sample_data.reset_index(drop=True, inplace=True)
            
            print(f"✅ Loaded {len(sample_data)} candles")
            
            # Initialize components
            candle_classifier = CandleClassifier(sample_data)
            zone_detector = ZoneDetector(candle_classifier, ZONE_CONFIG)
            
            # Detect patterns
            print("🔍 Detecting patterns...")
            patterns = zone_detector.detect_all_patterns(sample_data)
            
            total_patterns = patterns['total_patterns']
            print(f"📈 Found {total_patterns} patterns")
            
            if total_patterns == 0:
                print("⚠️  No patterns found to display")
                return True
            
            # Create figure for display only
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_facecolor(self.colors['background'])
            
            # Plot candlesticks
            self.plot_enhanced_candlesticks(ax, sample_data)
            
            # Prepare data for positioning
            data_high = sample_data['high'].max()
            data_low = sample_data['low'].min()
            data_range = data_high - data_low
            
            pattern_counter = 0
            
            # Collect all patterns with scores for sorting
            all_patterns = []
            
            for pattern in patterns['dbd_patterns']:
                score = self.calculate_zone_score(pattern)
                all_patterns.append((pattern, score, 'D-B-D'))
                
            for pattern in patterns['rbr_patterns']:
                score = self.calculate_zone_score(pattern)
                all_patterns.append((pattern, score, 'R-B-R'))
            
            # Sort by score (highest first)
            all_patterns.sort(key=lambda x: x[1], reverse=True)
            
            # Plot BASE ZONES ONLY
            for pattern, score, pattern_type in all_patterns:
                pattern_counter += 1
                
                # Use BASE boundaries only
                base = pattern['base']
                start_idx = base['start_idx']
                end_idx = base['end_idx']
                
                # Calculate BASE zone boundaries
                zone_high, zone_low = self.calculate_base_zone_boundaries(pattern, sample_data)
                zone_range = zone_high - zone_low
                
                width = end_idx - start_idx + 1
                
                if pattern_type == 'D-B-D':
                    zone_color = self.colors['dbd_zone']
                    edge_color = self.colors['dbd_edge']
                else:
                    zone_color = self.colors['rbr_zone']
                    edge_color = self.colors['rbr_edge']
                
                # Plot BASE zone rectangle only
                rect = patches.Rectangle((start_idx - 0.5, zone_low), width, zone_high - zone_low, 
                                       facecolor=zone_color, 
                                       alpha=0.25, edgecolor=edge_color, 
                                       linewidth=2, linestyle='-')
                ax.add_patch(rect)
                
                # Zone boundaries
                ax.hlines(zone_high, start_idx - 0.5, end_idx + 0.5, 
                         colors=edge_color, linewidth=2, alpha=0.8)
                ax.hlines(zone_low, start_idx - 0.5, end_idx + 0.5, 
                         colors=edge_color, linewidth=2, alpha=0.8)
                
                # Simple label positioning (no complex overlap detection for debug)
                zone_center_x = (start_idx + end_idx) / 2
                if pattern_type == 'D-B-D':
                    label_y = zone_high + data_range * 0.05
                else:
                    label_y = zone_low - data_range * 0.05
                
                self.create_zone_label(ax, pattern, zone_center_x, label_y, pattern_counter, zone_range)
            
            # Simple formatting for debug
            ax.set_title(f'DEBUG: EURUSD {timeframe} Base Zones | {total_patterns} Patterns', 
                        fontsize=16, weight='bold', color=self.colors['text'])
            ax.set_xlabel('Candle Index', fontsize=12, color=self.colors['text'])
            ax.set_ylabel('Price Level', fontsize=12, color=self.colors['text'])
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Show summary in console (no text on chart)
            if all_patterns:
                all_scores = [score for _, score, _ in all_patterns]
                print(f"📊 SUMMARY: Avg Score: {np.mean(all_scores):.1f} | Best: {max(all_scores):.1f}")
                
                excellent = len([s for s in all_scores if s >= 80])
                good = len([s for s in all_scores if 60 <= s < 80])
                weak = len([s for s in all_scores if s < 60])
                print(f"🎯 Quality Distribution: {excellent} Excellent | {good} Good | {weak} Weak")
            
            plt.tight_layout()
            
            # DISPLAY ONLY - NO SAVING
            print("🖼️  Displaying visualization (close window to continue)...")
            plt.show()
            
            print("✅ Debug visualization completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in debug visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Main execution for debugging
def run_debug_visualization():
    """Run DEBUG visualization - display only, no files"""
    
    print("🔧 DEBUG BASE ZONE VISUALIZATION")
    print("=" * 40)
    
    visualizer = EnhancedZoneVisualizer()
    
    # Debug visualization
    success = visualizer.visualize_zones_debug(
        sample_size=150,   # Smaller for debugging
        timeframe='Weekly'
    )
    
    if success:
        print("✅ Debug visualization completed!")
    else:
        print("❌ Debug visualization failed!")

if __name__ == "__main__":
    run_debug_visualization()