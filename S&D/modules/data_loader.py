"""
Data Loading for MetaTrader CSV Format
Handles your specific file structure and format
"""

import pandas as pd
import os
import glob
from typing import Optional, List
from config.settings import PATHS, COLUMN_MAPPING, FILE_PATTERNS

class DataLoader:
    def __init__(self):
        self.raw_path = PATHS['raw_data']
        self.processed_path = PATHS['processed_data']
        self.column_mapping = COLUMN_MAPPING
        
    def load_pair_data(self, pair: str, timeframe: str = 'Daily') -> pd.DataFrame:
        """
        Load OHLC data for a currency pair from multiple formats
        """
        print(f"📊 Loading {pair} {timeframe} data...")
        
        # Map timeframes to file patterns
        timeframe_mapping = {
            'daily': ['Daily', '1D'],
            '2daily': ['2Daily', '2D'], 
            '3daily': ['3Daily', '3D'],
            '4daily': ['4Daily', '4D'],
            '5daily': ['5Daily', '5D'],
            'weekly': ['Weekly', '1W'],
            'h12': ['H12', '12H'],
            'h4': ['H4', '4H']
        }
        
        timeframe_lower = timeframe.lower()
        possible_tf_names = timeframe_mapping.get(timeframe_lower, [timeframe])
        
        # Try multiple file patterns
        patterns_to_try = []
        
        for tf_name in possible_tf_names:
            # Standard MetaTrader format
            patterns_to_try.append(f"{pair}.raw_{tf_name}_*.csv")
            
            # OANDA format with underscores
            patterns_to_try.append(f"OANDA_{pair}_{tf_name}_*")
            patterns_to_try.append(f"OANDA_{pair}_{tf_name}_*.csv")
            
            # OANDA format with COMMA AND SPACE (your actual format)
            patterns_to_try.append(f"OANDA_{pair}, {tf_name}_*")
            patterns_to_try.append(f"OANDA_{pair}, {tf_name}_*.csv")
            
            # Simple formats
            patterns_to_try.append(f"{pair}_{tf_name}.csv")
            patterns_to_try.append(f"{pair}_{tf_name}_*.csv")
        
        matching_files = []
        for pattern in patterns_to_try:
            search_pattern = os.path.join(self.raw_path, pattern)
            files = glob.glob(search_pattern)
            if files:
                matching_files.extend(files)
                print(f"📁 Found files with pattern '{pattern}': {[os.path.basename(f) for f in files]}")
                break
        
        if not matching_files:
            # Show available files for debugging
            available_files = [f for f in os.listdir(self.raw_path) if f.endswith('.csv') or not '.' in f]
            raise FileNotFoundError(
                f"No files found for {pair} {timeframe}.\n"
                f"Searched patterns: {patterns_to_try}\n"
                f"Available files: {available_files}"
            )
        
        # Use the first matching file
        filepath = matching_files[0]
        print(f"📁 Using file: {os.path.basename(filepath)}")
        
        # Load the CSV
        try:
            # Try reading as CSV first
            data = pd.read_csv(filepath)
            
            # If no headers or wrong format, try tab-separated
            if len(data.columns) == 1 or '<DATE>' in str(data.columns):
                data = pd.read_csv(filepath, sep='\t')
            
            # Clean and standardize the data
            data = self._clean_data(data)
            
            print(f"✅ Loaded {len(data)} candles for {pair}")
            return data
            
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
        
        # Convert datetime
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
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
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness
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
        
        # Check high >= low
        if (data['high'] < data['low']).any():
            print("❌ Data contains invalid high/low relationships")
            return False
        
        # Check open/close within high/low range
        if ((data['open'] > data['high']) | (data['open'] < data['low'])).any():
            print("❌ Data contains open prices outside high/low range")
            return False
        
        if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
            print("❌ Data contains close prices outside high/low range")
            return False
        
        print("✅ Data quality validation passed")
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

# Convenience function
def load_pair_data(pair: str, timeframe: str = 'Daily') -> pd.DataFrame:
    """
    Convenience function to load pair data
    """
    loader = DataLoader()
    return loader.load_pair_data(pair, timeframe)