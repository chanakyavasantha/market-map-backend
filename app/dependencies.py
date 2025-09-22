import os
import logging
from pymongo import MongoClient
from fastapi import HTTPException
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mongodb_collection():
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME", "marketmap")
        collection_name = os.getenv("COLLECTION_NAME", "indices")
        
        logger.info(f"MongoDB URI exists: {bool(mongodb_uri)}")
        logger.info(f"DB Name: {db_name}")
        logger.info(f"Collection Name: {collection_name}")
        
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable is not set")
            raise HTTPException(status_code=500, detail="Database configuration error")
        
        logger.info("Attempting to connect to MongoDB...")
        
        # Add SSL configuration for Azure compatibility
        client = MongoClient(
            mongodb_uri,
            ssl=True,
            ssl_cert_reqs=ssl.CERT_NONE,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        
        # Test the connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        db = client[db_name]
        collection = db[collection_name]
        return collection
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")