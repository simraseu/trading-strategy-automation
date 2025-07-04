"""
Enhanced Zone Visualization Tool - Module 2 (CLEAN VERSION)
Beautiful, non-overlapping visualization with comprehensive zone scoring
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from datetime import datetime
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

    def get_smart_label_position(self, pattern, data_high, data_low, existing_labels, candle_positions):
        """Calculate smart non-overlapping label position that avoids candles"""
        
        zone_center_x = (pattern['start_idx'] + pattern['end_idx']) / 2
        zone_high = pattern['zone_high']
        zone_low = pattern['zone_low']
        data_range = data_high - data_low
        
        # Calculate safe distance from candles
        safe_distance = data_range * 0.08  # 8% of range
        
        # Starting positions
        if pattern['type'] == 'D-B-D':
            # Supply zone - try positions above zone
            candidate_positions = [
                zone_high + safe_distance,           # First choice
                zone_high + safe_distance * 1.8,    # Second choice
                zone_high + safe_distance * 2.6,    # Third choice
                zone_low - safe_distance,           # Fallback below
            ]
            va_options = ['bottom', 'bottom', 'bottom', 'top']
        else:
            # Demand zone - try positions below zone
            candidate_positions = [
                zone_low - safe_distance,           # First choice
                zone_low - safe_distance * 1.8,    # Second choice
                zone_low - safe_distance * 2.6,    # Third choice
                zone_high + safe_distance,          # Fallback above
            ]
            va_options = ['top', 'top', 'top', 'bottom']
        
        # Find best position
        label_width = 25  # Width in candles
        label_height = data_range * 0.06
        
        for i, y_pos in enumerate(candidate_positions):
            position_good = True
            
            # Check overlap with existing labels
            for existing in existing_labels:
                x_overlap = abs(zone_center_x - existing['x']) < label_width
                y_overlap = abs(y_pos - existing['y']) < label_height
                if x_overlap and y_overlap:
                    position_good = False
                    break
            
            # Check overlap with high-volatility candles
            if position_good:
                for candle_x in range(max(0, int(zone_center_x - label_width//2)), 
                                    min(len(candle_positions), int(zone_center_x + label_width//2))):
                    if candle_x < len(candle_positions):
                        candle_high = candle_positions[candle_x]['high']
                        candle_low = candle_positions[candle_x]['low']
                        
                        # Check if label would overlap with candle
                        if pattern['type'] == 'D-B-D' and i < 3:  # Above zone
                            if y_pos - label_height < candle_high:
                                position_good = False
                                break
                        elif pattern['type'] == 'R-B-R' and i < 3:  # Below zone
                            if y_pos + label_height > candle_low:
                                position_good = False
                                break
            
            if position_good:
                return zone_center_x, y_pos, va_options[i]
        
        # If no good position found, use the safest fallback
        fallback_y = data_high + safe_distance * 2 if pattern['type'] == 'D-B-D' else data_low - safe_distance * 2
        return zone_center_x, fallback_y, 'bottom' if pattern['type'] == 'D-B-D' else 'top'

    def create_zone_label(self, ax, pattern, position, pattern_num):
        """Create professional zone label with comprehensive info"""
        
        x, y, va = position
        zone_score = self.calculate_zone_score(pattern)
        
        # Create label content
        pattern_type = pattern['type']
        strength = pattern['strength']
        leg_out_ratio = pattern['leg_out']['ratio_to_base']
        base_candles = pattern['base']['candle_count']
        zone_pips = (pattern['zone_range'] / 0.0001)
        
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
               ha='center', va=va, fontsize=9, color=text_color, 
               weight='bold', family='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color, 
                        alpha=0.95, edgecolor='white', linewidth=1.5))

    def visualize_zones_professional(self, sample_size=200, timeframe='H12'):
        """Create professional zone visualization with comprehensive analysis"""
        
        print("🎨 PROFESSIONAL ZONE VISUALIZATION")
        print("=" * 60)
        
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
            
            # Create single large figure
            fig, ax = plt.subplots(figsize=(24, 12))
            ax.set_facecolor(self.colors['background'])
            
            # Plot candlesticks
            self.plot_enhanced_candlesticks(ax, sample_data)
            
            # Prepare data for smart positioning
            data_high = sample_data['high'].max()
            data_low = sample_data['low'].min()
            candle_positions = []
            
            for i, (idx, candle) in enumerate(sample_data.iterrows()):
                candle_positions.append({
                    'high': candle['high'],
                    'low': candle['low']
                })
            
            existing_labels = []
            pattern_counter = 0
            
            # Plot zones with smart positioning
            all_patterns = []
            
            # Collect all patterns with scores for sorting
            for pattern in patterns['dbd_patterns']:
                score = self.calculate_zone_score(pattern)
                all_patterns.append((pattern, score, 'D-B-D'))
                
            for pattern in patterns['rbr_patterns']:
                score = self.calculate_zone_score(pattern)
                all_patterns.append((pattern, score, 'R-B-R'))
            
            # Sort by score (highest first) for better label placement
            all_patterns.sort(key=lambda x: x[1], reverse=True)
            
            # Plot zones
            for pattern, score, pattern_type in all_patterns:
                pattern_counter += 1
                
                # Zone rectangle
                start_idx = pattern['start_idx']
                end_idx = pattern['end_idx']
                zone_high = pattern['zone_high']
                zone_low = pattern['zone_low']
                
                width = end_idx - start_idx + 1
                
                if pattern_type == 'D-B-D':
                    zone_color = self.colors['dbd_zone']
                    edge_color = self.colors['dbd_edge']
                else:
                    zone_color = self.colors['rbr_zone']
                    edge_color = self.colors['rbr_edge']
                
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
                
                # Smart label positioning
                position = self.get_smart_label_position(pattern, data_high, data_low, existing_labels, candle_positions)
                self.create_zone_label(ax, pattern, position, pattern_counter)
                existing_labels.append({'x': position[0], 'y': position[1]})
            
            # Professional formatting
            ax.set_title(f'EURUSD {timeframe} - Zone Analysis | {total_patterns} Patterns Detected', 
                        fontsize=18, weight='bold', color=self.colors['text'], pad=20)
            ax.set_xlabel('Candle Index', fontsize=14, color=self.colors['text'])
            ax.set_ylabel('Price Level', fontsize=14, color=self.colors['text'])
            
            # Professional X-axis
            major_ticks = list(range(0, len(sample_data), max(10, len(sample_data)//15)))
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f'{x}' for x in major_ticks])
            
            # Grid and styling
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Legend
            legend_elements = [
                patches.Patch(color=self.colors['dbd_zone'], alpha=0.7, label='Supply Zones (D-B-D)'),
                patches.Patch(color=self.colors['rbr_zone'], alpha=0.7, label='Demand Zones (R-B-R)'),
                patches.Patch(color='#2ECC71', alpha=0.7, label='Excellent (80-100)'),
                patches.Patch(color='#F39C12', alpha=0.7, label='Good (60-79)'),
                patches.Patch(color='#E74C3C', alpha=0.7, label='Weak (<60)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(0.02, 0.98), fontsize=12)
            
            # Add summary statistics as text
            if all_patterns:
                all_scores = [score for _, score, _ in all_patterns]
                stats_text = f"📊 SUMMARY: Avg Score: {np.mean(all_scores):.1f} | "
                stats_text += f"Best: {max(all_scores):.1f} | "
                
                excellent = len([s for s in all_scores if s >= 80])
                good = len([s for s in all_scores if 60 <= s < 80])
                weak = len([s for s in all_scores if s < 60])
                
                stats_text += f"Quality: {excellent}E/{good}G/{weak}W"
                
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                       fontsize=11, weight='bold', color=self.colors['text'],
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/zone_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
            
            print(f"✅ Professional visualization saved: {filename}")
            plt.show()
            
            # Generate detailed report
            self.generate_zone_report(patterns, sample_data, timeframe)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def generate_zone_report(self, patterns, data, timeframe):
        """Generate detailed CSV report for validation"""
        
        report_data = []
        
        # Process all patterns
        for pattern_type in ['dbd_patterns', 'rbr_patterns']:
            for i, pattern in enumerate(patterns[pattern_type]):
                zone_score = self.calculate_zone_score(pattern)
                
                report_data.append({
                    'Pattern_ID': f"{pattern_type}_{i+1}",
                    'Type': pattern['type'],
                    'Score': zone_score,
                    'Strength': pattern['strength'],
                    'Start_Candle': pattern['start_idx'],
                    'End_Candle': pattern['end_idx'],
                    'Zone_High': pattern['zone_high'],
                    'Zone_Low': pattern['zone_low'],
                    'Zone_Pips': pattern['zone_range'] / 0.0001,
                    'Base_Candles': pattern['base']['candle_count'],
                    'Leg_Out_Ratio': pattern['leg_out']['ratio_to_base'],
                    'Leg_In_Strength': pattern['leg_in']['strength'],
                    'Leg_Out_Strength': pattern['leg_out']['strength'],
                    'Base_Quality': pattern['base']['quality_score'],
                    'Validation_Status': 'PENDING'  # For manual validation
                })
        
        # Save report
        report_df = pd.DataFrame(report_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/zone_validation_report_{timestamp}.csv'
        report_df.to_csv(filename, index=False)
        
        print(f"📋 Validation report saved: {filename}")
        print(f"📊 Total patterns for validation: {len(report_data)}")

# Main execution
def run_enhanced_visualization():
    """Run the enhanced visualization system"""
    
    print("🎨 ENHANCED ZONE VISUALIZATION SYSTEM")
    print("=" * 60)
    
    visualizer = EnhancedZoneVisualizer()
    
    # Create professional visualization
    success = visualizer.visualize_zones_professional(
        sample_size=150,  # Adjust as needed
        timeframe='Daily'  # Change to H12, H4, etc.
    )
    
    if success:
        print("\n✅ Enhanced visualization completed successfully!")
        print("📁 Check results/ folder for:")
        print("   - Professional chart image")
        print("   - Detailed validation CSV")
        print("   - Ready for manual validation")
    else:
        print("\n❌ Visualization failed - check error messages above")

if __name__ == "__main__":
    run_enhanced_visualization()