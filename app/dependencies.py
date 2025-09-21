import os
from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def get_mongodb_collection():
    """Get MongoDB collection for market data"""
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise HTTPException(status_code=500, detail="MongoDB URI not configured")
        
        client = MongoClient(mongodb_uri)
        db_name = os.getenv("DB_NAME", "marketmap")
        collection_name = os.getenv("COLLECTION_NAME", "market_data")
        
        return client[db_name][collection_name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")