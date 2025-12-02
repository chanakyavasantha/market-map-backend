#!/usr/bin/env python3
"""
Simple script for fetching major market indices data using yfinance and saving to MongoDB.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
@dataclass
class Config:
    mongodb_uri: str = os.getenv("MONGODB_URI")
    database_name: str = os.getenv("DB_NAME")
    collection_name: str = os.getenv("COLLECTION_NAME")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

@dataclass
class IndexData:
    symbol: str
    name: str
    current_price: float
    change: float
    change_percent: float
    timestamp: datetime
    source: str = "yfinance"
    volume: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None

class MarketDataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self.client = None
        self.db = None
        self.collection = None
        
        # Load indices from metadata file instead of hardcoding
        # Get the path relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(current_dir, "..", "data", "indices_metadata.json")
        metadata_path = os.path.abspath(metadata_path)
        try:
            with open(metadata_path, 'r') as f:
                self.metadata: Dict[str, Dict] = json.load(f)
                self.indices = list(self.metadata.keys())
                self.logger.info(f"Loaded {len(self.indices)} symbols from metadata")
        except Exception as e:
            self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            # Fallback to empty list; run won't fetch anything
            self.metadata = {}
            self.indices = []

    def _setup_logging(self) -> logging.Logger:
        """Set up simple logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    def connect_to_mongodb(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.config.mongodb_uri)
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.config.database_name]
            self.collection = self.db[self.config.collection_name]
            
            # Create index for better performance
            self.collection.create_index([("symbol", 1), ("timestamp", -1)])
            
            self.logger.info("Connected to MongoDB successfully")
            
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")

    def fetch_single_ticker(self, symbol: str) -> Optional[IndexData]:
        """Fetch data for a single ticker."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current info
            info = ticker.info
            
            # Get recent price data
            hist = ticker.history(period="2d")
            
            if hist.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Get the most recent data
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            
            # Calculate change
            if len(hist) >= 2:
                previous_close = float(hist.iloc[-2]['Close'])
            else:
                previous_close = info.get('previousClose', current_price)
            
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            # Extract data safely
            def safe_float(value):
                try:
                    return float(value) if value and not pd.isna(value) else None
                except (ValueError, TypeError):
                    return None
            
            name_from_meta = self.metadata.get(symbol, {}).get('name')
            
            return IndexData(
                symbol=symbol,
                name=name_from_meta or info.get('shortName', symbol),
                current_price=current_price,
                change=change,
                change_percent=change_percent,
                timestamp=datetime.now(),
                volume=safe_float(latest.get('Volume')),
                high=safe_float(latest.get('High')),
                low=safe_float(latest.get('Low')),
                open=safe_float(latest.get('Open')),
                previous_close=previous_close,
                market_cap=info.get('marketCap')
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None

    def fetch_all_data(self) -> List[IndexData]:
        """Fetch data for all symbols."""
        self.logger.info("Fetching market data...")
        
        results = []
        for symbol in self.indices:
            data = self.fetch_single_ticker(symbol)
            if data:
                results.append(data)
                self.logger.info(f"Fetched {symbol}: ${data.current_price:.2f} ({data.change_percent:+.2f}%)")
            else:
                self.logger.warning(f"Failed to fetch data for {symbol}")
        
        self.logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results

    def save_data(self, data_list: List[IndexData]):
        """Save data to MongoDB."""
        self.logger.info("Saving data to MongoDB...")
        
        saved_count = 0
        for data in data_list:
            try:
                document = asdict(data)
                # Create unique ID to prevent duplicates
                document['_id'] = f"{data.symbol}_{data.timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                self.collection.insert_one(document)
                saved_count += 1
                
            except DuplicateKeyError:
                self.logger.debug(f"Duplicate entry for {data.symbol}")
            except Exception as e:
                self.logger.error(f"Error saving {data.symbol}: {e}")
        
        self.logger.info(f"Saved {saved_count} records to MongoDB")

    def run(self):
        """Main execution method."""
        try:
            # Connect to database
            self.connect_to_mongodb()
            
            # Fetch all data
            data = self.fetch_all_data()
            
            if data:
                # Save to database
                self.save_data(data)
                self.logger.info("Data fetch and save completed successfully")
            else:
                self.logger.warning("No data was fetched")
                
        except Exception as e:
            self.logger.error(f"Error in main execution: {e}")
            raise
        finally:
            self.close_connection()

    def get_stats(self):
        """Get simple database statistics."""
        try:
            self.connect_to_mongodb()
            
            total_docs = self.collection.count_documents({})
            unique_symbols = len(self.collection.distinct("symbol"))
            
            # Get latest timestamp
            latest = self.collection.find_one({}, sort=[("timestamp", -1)])
            latest_time = latest.get("timestamp") if latest else None
            
            print(f"Database Statistics:")
            print(f"Total documents: {total_docs}")
            print(f"Unique symbols: {unique_symbols}")
            print(f"Latest data: {latest_time}")
            
            # Show some recent data
            recent_data = self.collection.find({}, sort=[("timestamp", -1)]).limit(5)
            print(f"\nRecent entries:")
            for doc in recent_data:
                print(f"  {doc['symbol']}: ${doc['current_price']:.2f} at {doc['timestamp']}")
                
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
        finally:
            self.close_connection()

    def list_symbols(self):
        """List all tracked symbols."""
        print(f"Tracked symbols ({len(self.indices)}):")
        for symbol in self.indices:
            name = self.metadata.get(symbol, {}).get('name', symbol)
            country = self.metadata.get(symbol, {}).get('country', 'Unknown')
            print(f"  {symbol}: {name} ({country})")

def main():
    """Main entry point."""
    config = Config()
    fetcher = MarketDataFetcher(config)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--stats":
            fetcher.get_stats()
        elif command == "--list":
            fetcher.list_symbols()
        elif command == "--help":
            print("Usage:")
            print("  python market_data_fetcher.py          # Fetch and save data")
            print("  python market_data_fetcher.py --stats  # Show database stats")
            print("  python market_data_fetcher.py --list   # List tracked symbols")
            print("  python market_data_fetcher.py --help   # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Default: fetch and save data
        fetcher.run()

if __name__ == "__main__":
    main()