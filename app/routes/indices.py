from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
from ..dependencies import get_mongodb_collection
from ..agent.financial_news_agent import FinancialNewsAgent
from ..services.data_fetcher import MarketDataFetcher, Config

router = APIRouter(prefix="/api/indices", tags=["indices"])

# Load indices metadata
def load_indices_metadata() -> Dict:
    # Get the path relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "..", "data", "indices_metadata.json")
    metadata_path = os.path.abspath(metadata_path)
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

@router.post("/fetch-data")
async def fetch_market_data(background_tasks: BackgroundTasks):
    """
    Trigger market indices data fetching (runs in background)
    """
    try:
        config = Config()
        fetcher = MarketDataFetcher(config)
        
        # Run in background
        background_tasks.add_task(fetcher.run)
        
        return {
            "status": "started",
            "message": "Market data fetching started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting market data fetch: {str(e)}")

@router.post("/fetch-all")
async def fetch_all_data(
    background_tasks: BackgroundTasks,
    max_news_results: int = Query(default=10, ge=1, le=50, description="Maximum number of news articles to fetch")
):
    """
    Trigger both market indices data and financial news fetching simultaneously (runs in background)
    """
    try:
        # Fetch market indices data
        config = Config()
        fetcher = MarketDataFetcher(config)
        background_tasks.add_task(fetcher.run)
        
        # Fetch financial news
        agent = FinancialNewsAgent()
        background_tasks.add_task(agent.fetch_and_process_news, max_results=max_news_results)
        
        return {
            "status": "started",
            "message": f"Market data and news fetching started (max {max_news_results} articles)",
            "timestamp": datetime.now().isoformat(),
            "tasks": ["market_data", "financial_news"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting data fetch: {str(e)}")

@router.get("/with-news")
async def get_indices_with_news():
    """Get list of indices that have associated news articles"""
    try:
        agent = FinancialNewsAgent()
        
        # Get all unique index symbols from mappings
        pipeline = [
            {"$group": {
                "_id": "$index_symbol",
                "article_count": {"$sum": 1}
            }},
            {"$project": {
                "index_symbol": "$_id",
                "article_count": 1,
                "_id": 0
            }}
        ]
        
        mappings = list(agent.mappings_collection.aggregate(pipeline))
        
        return {
            "indices": [m["index_symbol"] for m in mappings],
            "indices_with_counts": {m["index_symbol"]: m["article_count"] for m in mappings},
            "count": len(mappings),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching indices with news: {str(e)}")

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

@router.get("/{symbol}/news")
async def get_index_news(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of news articles to return")
):
    """Get financial news articles for a specific index"""
    if symbol not in INDICES_METADATA:
        raise HTTPException(status_code=404, detail=f"Index {symbol} not found")
    
    try:
        agent = FinancialNewsAgent()
        articles = agent.get_news_by_index(symbol, limit=limit)
        
        # Convert MongoDB documents to JSON-serializable format
        for article in articles:
            if "_id" in article:
                article["_id"] = str(article["_id"])
            if "published_at" in article and isinstance(article["published_at"], datetime):
                article["published_at"] = article["published_at"].isoformat()
            if "fetched_at" in article and isinstance(article["fetched_at"], datetime):
                article["fetched_at"] = article["fetched_at"].isoformat()
            # Remove embedding from response (too large)
            if "embedding" in article:
                del article["embedding"]
        
        return {
            "index_symbol": symbol,
            "index_name": INDICES_METADATA[symbol].get("name"),
            "articles": articles,
            "count": len(articles),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news for index {symbol}: {str(e)}")