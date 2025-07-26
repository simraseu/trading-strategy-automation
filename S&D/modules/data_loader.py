"""
Data Loading for MetaTrader CSV Format
Handles your specific file structure and format
"""

import pandas as pd
import os
import glob
from typing import Optional, List, Dict, Tuple
from config.settings import PATHS, COLUMN_MAPPING, FILE_PATTERNS

class DataLoader:
    def __init__(self):
        self.raw_path = PATHS['raw_data']
        self.processed_path = PATHS['processed_data']
        self.column_mapping = COLUMN_MAPPING
        
    def load_pair_data(self, pair: str, timeframe: str = 'Daily') -> pd.DataFrame:
        """
        Enhanced load method with smart OANDA format detection
        """
        print(f"📊 Loading {pair} {timeframe} data...")
        
# First try direct OANDA format matching
        files = self.list_available_files()
        target_file = None
        
        for filename in files:
            parsed = self.parse_oanda_filename(filename)
            if parsed:
                file_pair, file_timeframe = parsed
                if file_pair == pair and file_timeframe == timeframe:
                    target_file = filename
                    break
        
        if target_file:
            filepath = os.path.join(self.raw_path, target_file)
            print(f"📁 Found direct match: {target_file}")
        else:
            # Fallback to pattern matching for backward compatibility
            timeframe_mapping = {
                'daily': ['Daily', '1D'],
                '2daily': ['2Daily', '2D'], 
                '3daily': ['3Daily', '3D'],
                '4daily': ['4Daily', '4D'],
                '5daily': ['5Daily', '5D'],
                'weekly': ['Weekly', '1W'],
                '2weekly': ['2Weekly', '2W'],
                '3weekly': ['3Weekly', '3W'], 
                'monthly': ['Monthly', '1M'],
                'h12': ['H12', '12H'],
                'h4': ['H4', '4H']
            }
            
            timeframe_lower = timeframe.lower()
            possible_tf_names = timeframe_mapping.get(timeframe_lower, [timeframe])
            
            patterns_to_try = []
            for tf_name in possible_tf_names:
                patterns_to_try.extend([
                    f"{pair}.raw_{tf_name}_*.csv",
                    f"OANDA_{pair}_{tf_name}_*",
                    f"OANDA_{pair}_{tf_name}_*.csv",
                    f"OANDA_{pair}, {tf_name}_*",
                    f"OANDA_{pair}, {tf_name}_*.csv",
                    f"{pair}_{tf_name}.csv",
                    f"{pair}_{tf_name}_*.csv"
                ])
            
            matching_files = []
            for pattern in patterns_to_try:
                search_pattern = os.path.join(self.raw_path, pattern)
                files = glob.glob(search_pattern)
                if files:
                    matching_files.extend(files)
                    break
            
            if not matching_files:
                available_files = [f for f in os.listdir(self.raw_path) if f.endswith('.csv')]
                raise FileNotFoundError(
                    f"No files found for {pair} {timeframe}.\n"
                    f"Available files: {available_files[:10]}..." if len(available_files) > 10 else f"Available files: {available_files}"
                )
            
            filepath = matching_files[0]
            print(f"📁 Found via pattern matching: {os.path.basename(filepath)}")
        
        # Load and process the file
        try:
            data = pd.read_csv(filepath)
            
            if len(data.columns) == 1 or '<DATE>' in str(data.columns):
                data = pd.read_csv(filepath, sep='\t')
            
            data = self._clean_data(data)
            
            if self.validate_data_quality(data):
                print(f"✅ Loaded {len(data)} candles for {pair} {timeframe}")
                return data
            else:
                raise ValueError("Data quality validation failed")
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the loaded data
        """
        # Create a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Rename columns to standard format
        rename_map = {
            '<DATE>': 'date',
            '<OPEN>': 'open',
            '<HIGH>': 'high',
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<TICKVOL>': 'volume'
        }
        
        # Only rename columns that exist
        existing_renames = {old: new for old, new in rename_map.items() if old in cleaned_data.columns}
        cleaned_data.rename(columns=existing_renames, inplace=True)
        
        # Convert datetime with robust format handling
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = self._parse_datetime_robust(cleaned_data['date'])
            cleaned_data.set_index('date', inplace=True)
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        # Remove any rows with NaN values
        cleaned_data.dropna(subset=price_columns, inplace=True)
        
        return cleaned_data
    
    def _parse_alternative_format(self, filepath: str) -> pd.DataFrame:
        """
        Alternative parsing method for problematic files
        """
        print("🔄 Trying alternative parsing method...")
        
        # Read as plain text first
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        if lines and '<DATE>' in lines[0]:
            lines = lines[1:]
        
        # Parse each line
        parsed_data = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 5:  # Need at least date, open, high, low, close
                try:
                    parsed_data.append({
                        'date': parts[0],
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4])
                    })
                except (ValueError, IndexError):
                    continue
        
        if not parsed_data:
            raise ValueError("Could not parse any data from the file")
        
        # Convert to DataFrame
        data = pd.DataFrame(parsed_data)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        return data
    
    def _parse_datetime_robust(self, date_series: pd.Series) -> pd.Series:
        """
        Robust datetime parsing handling multiple formats:
        - ISO format: '2024-01-15 09:00:00'
        - UNIX timestamps: 1642233600
        - Date only: '2024-01-15'
        """
        print("🕐 Parsing datetime with robust format detection...")
        
        def parse_single_date(date_val):
            if pd.isna(date_val):
                return pd.NaT
            
            # Convert to string for processing
            date_str = str(date_val).strip()
            
            # Try UNIX timestamp first (all digits)
            if date_str.replace('.', '').isdigit():
                try:
                    # Handle both seconds and milliseconds timestamps
                    timestamp = float(date_str)
                    if timestamp > 1e10:  # Milliseconds
                        timestamp = timestamp / 1000
                    return pd.to_datetime(timestamp, unit='s')
                except (ValueError, OSError):
                    pass
            
            # Try standard pandas parsing for ISO formats
            try:
                return pd.to_datetime(date_str)
            except (ValueError, TypeError):
                pass
            
            # Try specific format patterns
            format_patterns = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y.%m.%d %H:%M:%S',
                '%Y.%m.%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y'
            ]
            
            for pattern in format_patterns:
                try:
                    return pd.to_datetime(date_str, format=pattern)
                except (ValueError, TypeError):
                    continue
            
            # If all fails, return NaT
            print(f"⚠️  Could not parse date: {date_str}")
            return pd.NaT
        
        # Apply robust parsing
        parsed_dates = date_series.apply(parse_single_date)
        
        # Validate results
        successful_parses = parsed_dates.notna().sum()
        total_dates = len(date_series)
        success_rate = (successful_parses / total_dates) * 100
        
        print(f"   ✅ Parsed {successful_parses}/{total_dates} dates ({success_rate:.1f}% success)")
        
        if success_rate < 95:
            print(f"   ⚠️  Low success rate - check date formats in source data")
        
        return parsed_dates

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness (RELAXED for real-world data)
        """
        print("🔍 Validating data quality...")
        
        required_cols = ['open', 'high', 'low', 'close']
        
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check for missing values
        if data[required_cols].isnull().any().any():
            print("❌ Data contains missing values")
            return False
        
        # Check for negative prices
        if (data[required_cols] <= 0).any().any():
            print("❌ Data contains negative/zero prices")
            return False
        
        # Check high >= low (CRITICAL validation)
        if (data['high'] < data['low']).any():
            print("❌ Data contains invalid high/low relationships")
            return False
        
        # RELAXED VALIDATION: Check for extreme outliers instead of strict range validation
        # Allow small violations due to data feed differences, but catch major errors
        
        # Check for extreme open price violations (more than 1% outside range)
        high_low_range = data['high'] - data['low']
        open_violations = ((data['open'] > data['high'] + 0.01 * high_low_range) | 
                          (data['open'] < data['low'] - 0.01 * high_low_range))
        
        if open_violations.any():
            violation_count = open_violations.sum()
            total_count = len(data)
            violation_pct = (violation_count / total_count) * 100
            
            if violation_pct > 5:  # Only fail if >5% of data has violations
                print(f"❌ Too many open price violations: {violation_count}/{total_count} ({violation_pct:.1f}%)")
                return False
            else:
                print(f"⚠️  Minor open price violations: {violation_count}/{total_count} ({violation_pct:.1f}%) - acceptable")
        
        # Check for extreme close price violations (more than 1% outside range)  
        close_violations = ((data['close'] > data['high'] + 0.01 * high_low_range) | 
                           (data['close'] < data['low'] - 0.01 * high_low_range))
        
        if close_violations.any():
            violation_count = close_violations.sum()
            total_count = len(data)
            violation_pct = (violation_count / total_count) * 100
            
            if violation_pct > 5:  # Only fail if >5% of data has violations
                print(f"❌ Too many close price violations: {violation_count}/{total_count} ({violation_pct:.1f}%)")
                return False
            else:
                print(f"⚠️  Minor close price violations: {violation_count}/{total_count} ({violation_pct:.1f}%) - acceptable")
        
        print("✅ Data quality validation passed (relaxed validation)")
        return True
    
    def list_available_files(self) -> List[str]:
        """
        List all available data files
        """
        if not os.path.exists(self.raw_path):
            return []
        
        files = []
        for file in os.listdir(self.raw_path):
            if file.endswith('.csv'):
                files.append(file)
        
        return files
    
    def parse_oanda_filename(self, filename: str) -> Optional[Tuple[str, str]]:
        """
        Extract pair and timeframe from OANDA format filename
        
        Args:
            filename: OANDA format filename (e.g., "OANDA_AUDCAD, 1D_0d7fe.csv")
            
        Returns:
            Tuple of (pair, timeframe) or None if parsing fails
        """
        try:
            # Remove .csv extension
            clean_name = filename.replace('.csv', '')
            
            # Handle OANDA format: OANDA_AUDCAD, 1D_0d7fe
            if 'OANDA_' in clean_name and ', ' in clean_name:
                # Remove OANDA_ prefix
                without_prefix = clean_name.replace('OANDA_', '')
                # Split on comma-space
                parts = without_prefix.split(', ')
                
                if len(parts) >= 2:
                    pair = parts[0].strip()  # AUDCAD
                    timeframe_part = parts[1].strip()  # 1D_0d7fe
                    
                    # Extract timeframe (everything before underscore)
                    if '_' in timeframe_part:
                        timeframe = timeframe_part.split('_')[0]  # 1D
                    else:
                        timeframe = timeframe_part
                    
                    # Validate pair format (6 characters for forex)
                    if len(pair) == 6 and pair.isalpha():
                        return (pair, timeframe)
            
            return None
            
        except Exception:
            return None
    
    def discover_all_pairs(self) -> List[str]:
        """
        Auto-discover all available currency pairs from data files
        
        Returns:
            List of unique currency pairs found
        """
        print(f"🔍 Searching for pairs in: {self.raw_path}")
        
        files = self.list_available_files()
        print(f"📁 Found {len(files)} CSV files")
        
        pairs = set()
        
        for filename in files:
            parsed = self.parse_oanda_filename(filename)
            if parsed:
                pair, timeframe = parsed
                pairs.add(pair)
        
        pairs_list = sorted(list(pairs))
        print(f"📊 Found {len(pairs_list)} currency pairs: {', '.join(pairs_list)}")
        
        return pairs_list
    
    def discover_all_timeframes(self) -> List[str]:
        """
        Auto-discover all available timeframes from data files
        
        Returns:
            List of unique timeframes found
        """
        print(f"🔍 Searching for timeframes in: {self.raw_path}")
        
        files = self.list_available_files()
        timeframes = set()
        
        for filename in files:
            parsed = self.parse_oanda_filename(filename)
            if parsed:
                pair, timeframe = parsed
                timeframes.add(timeframe)
        
        timeframes_list = sorted(list(timeframes))
        print(f"⏰ Found {len(timeframes_list)} timeframes: {', '.join(timeframes_list)}")
        
        return timeframes_list
    
    def get_available_data(self) -> Dict[str, List[str]]:
        """
        Get complete inventory of available data
        
        Returns:
            Dictionary mapping pairs to their available timeframes
        """
        print(f"📋 Building complete data inventory...")
        
        files = self.list_available_files()
        data_inventory = {}
        
        for filename in files:
            parsed = self.parse_oanda_filename(filename)
            if parsed:
                pair, timeframe = parsed
                
                if pair not in data_inventory:
                    data_inventory[pair] = []
                
                if timeframe not in data_inventory[pair]:
                    data_inventory[pair].append(timeframe)
        
        # Sort timeframes for each pair
        for pair in data_inventory:
            data_inventory[pair].sort()
        
        total_files = sum(len(timeframes) for timeframes in data_inventory.values())
        print(f"✅ Inventory complete: {len(data_inventory)} pairs, {total_files} datasets")
        
        return data_inventory
    
    def load_all_pairs(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available pairs for a specific timeframe
        
        Args:
            timeframe: Target timeframe (e.g., '1D', '2D', 'Weekly')
            
        Returns:
            Dictionary mapping pair names to DataFrames
        """
        print(f"📊 Loading all pairs for timeframe: {timeframe}")
        
        inventory = self.get_available_data()
        results = {}
        successful_loads = 0
        
        for pair, available_timeframes in inventory.items():
            if timeframe in available_timeframes:
                try:
                    data = self.load_pair_data(pair, timeframe)
                    if data is not None and len(data) > 0:
                        results[pair] = data
                        successful_loads += 1
                        print(f"   ✅ {pair}: {len(data)} candles")
                    else:
                        print(f"   ❌ {pair}: No data loaded")
                except Exception as e:
                    print(f"   ❌ {pair}: {str(e)}")
        
        print(f"📈 Loaded {successful_loads}/{len([p for p, tf in inventory.items() if timeframe in tf])} pairs successfully")
        return results
    
    def load_all_timeframes(self, pair: str) -> Dict[str, pd.DataFrame]:
        """
        Load all available timeframes for a specific pair
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        print(f"📊 Loading all timeframes for pair: {pair}")
        
        inventory = self.get_available_data()
        results = {}
        
        if pair not in inventory:
            print(f"❌ Pair {pair} not found in data inventory")
            return results
        
        available_timeframes = inventory[pair]
        successful_loads = 0
        
        for timeframe in available_timeframes:
            try:
                data = self.load_pair_data(pair, timeframe)
                if data is not None and len(data) > 0:
                    results[timeframe] = data
                    successful_loads += 1
                    print(f"   ✅ {timeframe}: {len(data)} candles")
                else:
                    print(f"   ❌ {timeframe}: No data loaded")
            except Exception as e:
                print(f"   ❌ {timeframe}: {str(e)}")
        
        print(f"📈 Loaded {successful_loads}/{len(available_timeframes)} timeframes successfully")
        return results
    
    def load_complete_dataset(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all available data (all pairs, all timeframes)
        
        Returns:
            Nested dictionary: {pair: {timeframe: DataFrame}}
        """
        print(f"🚀 Loading complete dataset...")
        
        inventory = self.get_available_data()
        results = {}
        total_datasets = 0
        successful_loads = 0
        
        for pair, timeframes in inventory.items():
            print(f"\n📊 Loading {pair}...")
            results[pair] = {}
            
            for timeframe in timeframes:
                total_datasets += 1
                try:
                    data = self.load_pair_data(pair, timeframe)
                    if data is not None and len(data) > 0:
                        results[pair][timeframe] = data
                        successful_loads += 1
                        print(f"   ✅ {timeframe}: {len(data)} candles")
                    else:
                        print(f"   ❌ {timeframe}: No data loaded")
                except Exception as e:
                    print(f"   ❌ {timeframe}: {str(e)}")
        
        print(f"\n🎉 Complete dataset loaded: {successful_loads}/{total_datasets} datasets successful")
        print(f"📈 Final structure: {len(results)} pairs with data")
        
        return results