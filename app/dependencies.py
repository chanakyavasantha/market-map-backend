import os
import logging
from pymongo import MongoClient
from fastapi import HTTPException

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mongodb_collection():
    try:
        # Debug: Log environment variables (without exposing sensitive data)
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME", "marketmap")
        collection_name = os.getenv("COLLECTION_NAME", "indices")
        
        logger.info(f"MongoDB URI exists: {bool(mongodb_uri)}")
        logger.info(f"DB Name: {db_name}")
        logger.info(f"Collection Name: {collection_name}")
        
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable is not set")
            raise HTTPException(status_code=500, detail="Database configuration error")
        
        # Log connection attempt
        logger.info("Attempting to connect to MongoDB...")
        client = MongoClient(mongodb_uri)
        
        # Test the connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        db = client[db_name]
        collection = db[collection_name]
        return collection
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")