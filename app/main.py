from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Market Map API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load indices metadata
def load_indices_metadata() -> Dict:
    metadata_path = os.path.join(os.path.dirname(__file__), "data", "indices_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Cache metadata in memory since it rarely changes
INDICES_METADATA = load_indices_metadata()

# MongoDB connection
def get_mongodb_collection():
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("DB_NAME")]
        collection = db[os.getenv("COLLECTION_NAME")]
        return collection
    except ConnectionFailure as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Market Map API", "version": "1.0.0"}

@app.get("/api/indices/locations")
async def get_indices_locations():
    """Get static metadata for all indices (locations, exchanges, etc.)"""
    return {
        "data": list(INDICES_METADATA.values()),
        "count": len(INDICES_METADATA)
    }

@app.get("/api/indices/status")
async def get_indices_status():
    """Get latest market data joined with metadata for all indices"""
    try:
        collection = get_mongodb_collection()

        # Get latest data for each symbol from the DB (if any)
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$symbol",
                "latest_doc": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest_doc"}}
        ]
        latest_data = list(collection.aggregate(pipeline))

        # Build a map for quick lookup
        latest_by_symbol = {doc.get("symbol"): doc for doc in latest_data if doc.get("symbol")}

        # Always include all indices from metadata; if no market data, fill with nulls
        result = []
        for symbol, meta in INDICES_METADATA.items():
            data = latest_by_symbol.get(symbol)
            combined = {
                **meta,
                "current_price": data.get("current_price") if data else None,
                "change": data.get("change") if data else None,
                "change_percent": data.get("change_percent") if data else None,
                "volume": data.get("volume") if data else None,
                "last_updated": data.get("timestamp") if data else None,
                "source": data.get("source", "yfinance") if data else None
            }
            result.append(combined)

        return {
            "data": result,
            "count": len(result),
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market status: {str(e)}")

@app.get("/api/indices/{symbol}")
async def get_index_details(symbol: str):
    """Get detailed information for a specific index"""
    if symbol not in INDICES_METADATA:
        raise HTTPException(status_code=404, detail=f"Index {symbol} not found")
    
    try:
        collection = get_mongodb_collection()
        
        # Get latest data for this symbol
        latest = collection.find_one(
            {"symbol": symbol},
            sort=[("timestamp", -1)]
        )
        
        if not latest:
            # Return metadata only if no market data available
            return {
                **INDICES_METADATA[symbol],
                "current_price": None,
                "change": None,
                "change_percent": None,
                "last_updated": None,
                "message": "No market data available"
            }
        
        # Combine metadata with latest market data
        result = {
            **INDICES_METADATA[symbol],
            "current_price": latest.get("current_price"),
            "change": latest.get("change"),
            "change_percent": latest.get("change_percent"),
            "volume": latest.get("volume"),
            "high": latest.get("high"),
            "low": latest.get("low"),
            "open": latest.get("open"),
            "previous_close": latest.get("previous_close"),
            "market_cap": latest.get("market_cap"),
            "last_updated": latest.get("timestamp"),
            "source": latest.get("source", "yfinance")
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching index details: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)