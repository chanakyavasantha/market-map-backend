from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict
import json
import os
from ..dependencies import get_mongodb_collection

router = APIRouter(prefix="/api/indices", tags=["indices"])

# Load indices metadata
def load_indices_metadata() -> Dict:
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "data", "indices_metadata.json")
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Cache metadata in memory since it rarely changes
INDICES_METADATA = load_indices_metadata()

@router.get("/locations")
async def get_indices_locations():
    """Get static metadata for all indices (locations, exchanges, etc.)"""
    return {
        "data": list(INDICES_METADATA.values()),
        "count": len(INDICES_METADATA)
    }

@router.get("/status")
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

@router.get("/last-updated")
async def get_last_updated():
    """Get the timestamp of when the market data was last updated"""
    try:
        collection = get_mongodb_collection()
        
        # Get the most recent timestamp across all data
        latest = collection.find_one(
            {},
            sort=[("timestamp", -1)]
        )
        
        if not latest:
            return {
                "last_updated": None,
                "message": "No market data available"
            }
        
        return {
            "last_updated": latest.get("timestamp"),
            "last_updated_formatted": latest.get("timestamp").strftime("%Y-%m-%d %H:%M:%S UTC") if latest.get("timestamp") else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching last updated time: {str(e)}")

@router.get("/{symbol}")
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