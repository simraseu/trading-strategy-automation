"""
VALIDATION TEST 1: Zone Detection Historical Accuracy
TEST OBJECTIVE: Prove zone detection ≥95% accuracy on real market data
SUCCESS CRITERIA: Manual validation matches automated detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from modules.data_loader import DataLoader
from modules.candle_classifier import CandleClassifier
from modules.zone_detector import ZoneDetector

class ZoneAccuracyValidator:
    """Historical zone detection accuracy validation"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.results = {}
        
    def test_zone_detection_historical_accuracy(self):
        """
        Core validation test for zone detection accuracy
        Tests multiple market conditions for robustness
        """
        
        print("🎯 ZONE DETECTION ACCURACY VALIDATION")
        print("=" * 60)
        print("OBJECTIVE: Prove ≥95% accuracy on real market data")
        print("METHOD: Multi-period validation across market conditions")
        print("=" * 60)
        
        # Test periods covering different market conditions
        test_periods = [
            ('EURUSD', '3D', 730, 'Recent 2-year validation'),
            ('GBPUSD', 'Daily', 1095, 'Brexit aftermath period'),  
            ('USDJPY', '2D', 547, 'COVID volatility test'),
            ('AUDUSD', '3D', 365, 'Commodity correlation test')
        ]
        
        validation_results = {}
        
        for pair, timeframe, days_back, description in test_periods:
            print(f"\n📊 TESTING: {pair} {timeframe} - {description}")
            
            try:
                # Load historical data
                data = self.data_loader.load_pair_data(pair, timeframe)
                if data is None or len(data) < 100:
                    print(f"❌ Insufficient data for {pair} {timeframe}")
                    continue
                
                # Debug data structure
                print(f"   📊 Data columns: {list(data.columns)}")
                print(f"   📅 Data index type: {type(data.index)}")
                
                # Check if we have 'time' column and fix datetime index
                if 'time' in data.columns:
                    print(f"   🔧 Found 'time' column, converting to datetime index...")
                    try:
                        # Convert UNIX timestamp to datetime
                        data['datetime'] = pd.to_datetime(data['time'], unit='s')
                        data.set_index('datetime', inplace=True)
                        print(f"   ✅ Datetime index set successfully")
                        print(f"   📅 First date: {data.index[0].strftime('%d/%m/%Y')}")
                        print(f"   📅 Last date: {data.index[-1].strftime('%d/%m/%Y')}")
                    except Exception as e:
                        print(f"   ⚠️ Error setting datetime index: {str(e)}")
                        # Try ISO format
                        try:
                            data['datetime'] = pd.to_datetime(data['time'])
                            data.set_index('datetime', inplace=True)
                            print(f"   ✅ ISO datetime index set successfully")
                        except Exception as e2:
                            print(f"   ❌ Failed to convert time column: {str(e2)}")
                
                # Limit to test period AFTER fixing datetime
                if days_back < len(data):
                    data = data.iloc[-days_back:]
                
                # Run zone detection with 2.5x validation
                candle_classifier = CandleClassifier(data)
                classified_data = candle_classifier.classify_all_candles()
                
                zone_detector = ZoneDetector(candle_classifier)
                zones = zone_detector.detect_all_patterns(classified_data)
                
                # Calculate metrics including validation data
                total_patterns = zones['total_patterns']
                dbd_count = len(zones['dbd_patterns'])
                rbr_count = len(zones['rbr_patterns'])
                dbr_count = len(zones.get('dbr_patterns', []))
                rbd_count = len(zones.get('rbd_patterns', []))
                
                # NEW: 2.5x validation metrics
                validated_count = zones.get('validated_zones', 0)
                invalidated_count = zones.get('invalidated_zones', 0)
                pending_count = zones.get('pending_zones', 0)
                
                # Export for manual validation
                validation_file = f"validation_{pair}_{timeframe}_{days_back}d.csv"
                self.export_zones_for_manual_check(zones, validation_file, data, pair, timeframe)
                
                # Store results
                validation_results[f"{pair}_{timeframe}"] = {
                    'pair': pair,
                    'timeframe': timeframe,
                    'days_tested': days_back,
                    'candles_processed': len(data),
                    'total_patterns': total_patterns,
                    'dbd_patterns': dbd_count,
                    'rbr_patterns': rbr_count,
                    'dbr_patterns': dbr_count,
                    'rbd_patterns': rbd_count,
                    
                    # NEW: 2.5x validation metrics
                    'validated_zones': validated_count,
                    'invalidated_zones': invalidated_count,
                    'pending_zones': pending_count,
                    'validation_rate': f"{(validated_count/total_patterns)*100:.1f}%" if total_patterns > 0 else "0%",
                    
                    'validation_file': validation_file,
                    'status': 'READY_FOR_MANUAL_VALIDATION'
                }
                
                print(f"   ✅ Patterns detected: {total_patterns}")
                print(f"   📊 D-B-D: {dbd_count}, R-B-R: {rbr_count}, D-B-R: {dbr_count}, R-B-D: {rbd_count}")
                print(f"   🎯 2.5x Validation: {validated_count} validated, {invalidated_count} invalidated, {pending_count} pending")
                print(f"   📈 Validation rate: {(validated_count/total_patterns)*100:.1f}%" if total_patterns > 0 else "   📈 Validation rate: 0%")
                print(f"   📁 Validation file: {validation_file}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                validation_results[f"{pair}_{timeframe}"] = {
                    'pair': pair,
                    'timeframe': timeframe,
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        
        return validation_results
    
    def export_zones_for_manual_check(self, zones, filename, data, pair, timeframe):
        """Export zones with complete context for manual validation"""
        
        all_zones = []
        zone_types = ['dbd_patterns', 'rbr_patterns', 'dbr_patterns', 'rbd_patterns']
        
        for zone_type in zone_types:
            if zone_type in zones:
                for zone in zones[zone_type]:
                    try:
                        # Get zone formation dates in DD/MM/YYYY format
                        try:
                            # Now we should have proper datetime index
                            start_date = data.index[zone['start_idx']].strftime('%d/%m/%Y')
                            end_date = data.index[zone['end_idx']].strftime('%d/%m/%Y')
                            base_start = data.index[zone['base']['start_idx']].strftime('%d/%m/%Y')
                            base_end = data.index[zone['base']['end_idx']].strftime('%d/%m/%Y')
                            
                        except Exception as e:
                            print(f"   ⚠️ Date conversion error: {str(e)}")
                            # Fallback to index numbers
                            start_date = f"Index_{zone['start_idx']}"
                            end_date = f"Index_{zone['end_idx']}"
                            base_start = f"Index_{zone['base']['start_idx']}"
                            base_end = f"Index_{zone['base']['end_idx']}"
                        
                        zone_info = {
                            'pair': pair,
                            'timeframe': timeframe,
                            'pattern_type': zone['type'],
                            'zone_start_date': start_date,
                            'zone_end_date': end_date,
                            'base_start_date': base_start,
                            'base_end_date': base_end,
                            'zone_high': zone['zone_high'],
                            'zone_low': zone['zone_low'],
                            'zone_range': zone['zone_range'],
                            'leg_in_candles': zone['leg_in']['candle_count'],
                            'base_candles': zone['base']['candle_count'],
                            'leg_out_candles': zone['leg_out']['candle_count'],                            
                            # NEW: 2.5x validation data
                            'immediate_leg_out_ratio': zone.get('immediate_leg_out_ratio', zone['leg_out']['ratio_to_base']),
                            'maximum_distance_ratio': zone.get('maximum_distance_ratio', 'N/A'),
                            'target_2_5x_price': zone.get('target_2_5x_price', 'N/A'),
                            'target_2_5x_hit': zone.get('target_2_5x_hit', False),
                            'target_2_5x_date': zone.get('target_2_5x_date', 'N/A'),
                            'validation_status': zone.get('zone_validation_status', 'UNKNOWN'),
                            'invalidation_date': zone.get('invalidation_date', 'N/A'),
                            'monitoring_candles': zone.get('monitoring_candles_count', 0),
                        }
                        
                        all_zones.append(zone_info)
                        
                    except Exception as e:
                        print(f"   ⚠️  Error exporting zone: {str(e)}")
        
        # Save to CSV
        if all_zones:
            df = pd.DataFrame(all_zones)
            os.makedirs('validation_tests/exports', exist_ok=True)
            filepath = f"validation_tests/exports/{filename}"
            df.to_csv(filepath, index=False)
            print(f"   📁 Manual validation export: {filepath}")
        

def main():
    """Main execution function"""
    try:
        validator = ZoneAccuracyValidator()
        results = validator.test_zone_detection_historical_accuracy()
        
        print(f"\n🎯 ZONE ACCURACY VALIDATION COMPLETE!")
        print(f"📊 Tests completed: {len(results)}")
        
        successful = len([r for r in results.values() if r.get('status') == 'READY_FOR_MANUAL_VALIDATION'])
        print(f"✅ Successful validations: {successful}/{len(results)}")
        
        if successful > 0:
            print(f"\n📋 NEXT STEPS:")
            print(f"1. Check validation_tests/exports/ folder for CSV files")
            print(f"2. Manually validate the detected zones")
            print(f"3. Mark zones as CORRECT/INCORRECT in the CSV files")
            print(f"4. Calculate final accuracy percentage")
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()